"""Um laço para os quatro estágios: linha de base dos pontos, aluno com rótulos, professor e
aluno destilado.

Cada passo sorteia um frame por pausa (`PauseSampler`), aumenta o lote na GPU e
atualiza o modelo em bf16. A avaliação usa a média exponencial dos pesos (EMA).
Na destilação, professor e aluno recebem a mesma vista e a mesma mistura, e o
professor roda congelado a cada passo (Beyer et al., 2022).
"""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from research.data.augment import EvalView, TrainAugment, mixup
from research.data.splits import PauseSampler
from research.training.data import InMemorySplit
from research.training.inputs import model_inputs
from research.training.losses import kd_kl, mixed_cross_entropy
from research.training.metrics import summarize

STAGES = ("landmarks", "student_ce", "teacher", "student_kd")


def build_model(stage: str, config: dict, pretrained: bool) -> nn.Module:
    from research.networks.student import build_landmark_baseline, build_student
    from research.networks.teacher import build_teacher

    model_cfg = config["model"]
    classes = int(model_cfg["num_classes"])
    if stage == "landmarks":
        return build_landmark_baseline(model_cfg["landmark_baseline"], classes)
    if stage in ("student_ce", "student_kd"):
        return build_student(model_cfg["student"], classes, pretrained)
    if stage == "teacher":
        teacher = model_cfg["teacher"]
        return build_teacher(teacher["arch"], classes, bool(teacher["depth"]), pretrained)
    raise ValueError(f"estágio desconhecido: {stage}")


def _param_groups(model: nn.Module, stage_cfg: dict) -> list[dict]:
    """Sem weight decay em normalização e bias; cabeça nova mais rápida no professor."""
    lr = float(stage_cfg["lr"])
    head_ids = set()
    if hasattr(model, "head_parameters") and "head_lr_mult" in stage_cfg:
        head_ids = {id(p) for p in model.head_parameters()}
    groups: dict[tuple[bool, bool], list] = {}
    for param in model.parameters():
        if not param.requires_grad:
            continue
        key = (id(param) in head_ids, param.ndim <= 1)
        groups.setdefault(key, []).append(param)
    mult = float(stage_cfg.get("head_lr_mult", 1.0))
    decay = float(stage_cfg["weight_decay"])
    return [
        {
            "params": params,
            "lr": lr * (mult if head else 1.0),
            "weight_decay": 0.0 if flat else decay,
        }
        for (head, flat), params in groups.items()
    ]


def _schedule(warmup: int, total: int):
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return factor


def _batches(sampler: PauseSampler, batch_size: int):
    """Lotes sem fim; cada época é uma permutação nova das pausas."""
    epoch, buffer = 0, []
    while True:
        sampler.set_epoch(epoch)
        for index in sampler:
            buffer.append(index)
            if len(buffer) == batch_size:
                yield buffer, epoch
                buffer = []
        epoch += 1


class Trainer:
    """Treina `model` num estágio por `steps` passos e fica com o último checkpoint (EMA).

    `evals` são medidos durante o treino, só para acompanhar a curva; nada é
    escolhido por eles. `holdout` (o C6) só é medido uma vez, depois do último passo.
    """

    def __init__(
        self,
        stage: str,
        config: dict,
        model: nn.Module,
        train: InMemorySplit,
        evals: dict[str, InMemorySplit],
        hand_side_px: float,
        device: torch.device,
        seed: int,
        run_dir: Path,
        teacher: nn.Module | None = None,
        backgrounds: torch.Tensor | None = None,
        steps: int | None = None,
        holdout: dict[str, InMemorySplit] | None = None,
        caps: dict[int, int] | None = None,
        on_step=None,
        aux_teacher: nn.Module | None = None,
    ) -> None:
        """`on_step(trainer, step)` roda depois de cada passo (a poda gradual muda a máscara)."""
        if stage not in STAGES:
            raise ValueError(f"estágio desconhecido: {stage}")
        if (stage == "student_kd") != (teacher is not None):
            raise ValueError("só a destilação usa professor, e ela precisa de um")
        self.stage = stage
        self.config = config
        self.stage_cfg = config["stages"][stage]
        self.run_cfg = config["run"]
        self.device = device
        self.train = train
        self.evals = evals
        self.holdout = dict(holdout or {})
        self.on_step = on_step
        self.run_dir = run_dir
        self.steps = int(steps or self.stage_cfg["steps"])
        self.range_mm = float(config["depth"]["range_mm"])
        self.model = model.to(device).to(memory_format=torch.channels_last)
        self.teacher = None
        if teacher is not None:
            self.teacher = teacher.to(device).to(memory_format=torch.channels_last).eval()
            for param in self.teacher.parameters():
                param.requires_grad_(False)
        # Professor só da cabeça auxiliar (o modelo de pontos treinado à parte); sem
        # ele, a cabeça auxiliar aprende do mesmo professor da principal.
        self.aux_teacher = None
        if aux_teacher is not None:
            self.aux_teacher = aux_teacher.to(device).eval()
            for param in self.aux_teacher.parameters():
                param.requires_grad_(False)
        view = config["view"]
        self.augment = TrainAugment(
            config["augment"],
            hand_side_px,
            view["view_context"],
            view["input_size"],
            device,
            seed,
            backgrounds,
        )
        self.eval_view = EvalView(hand_side_px, view["view_context"], view["input_size"], device)
        self.sampler = PauseSampler(
            train.keys, _samples_of(train), seed, train.labels.tolist(), caps
        )
        self.mix_gen = torch.Generator(device=device).manual_seed(seed + 1)
        torch.manual_seed(seed)
        decay = float(self.run_cfg["ema_decay"])
        self.ema = torch.optim.swa_utils.AveragedModel(
            self.model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay),
            use_buffers=True,
        )
        # Alinhamento de features (cosine KD): uma projeção aluno → professor por
        # professor, só no treino; não entra no last.pt nem no app.
        self.feature_kd = float(self.stage_cfg.get("feature_kd", 0.0))
        self.aux_weight = float(self.stage_cfg.get("aux_weight", 0.0))
        self.projections = None
        groups = _param_groups(self.model, self.stage_cfg)
        if self.feature_kd > 0:
            members = getattr(self.teacher, "members", None) or [self.teacher]
            fused = self.model.head.in_features
            self.projections = nn.ModuleList(
                [nn.Linear(fused, t.head.in_features) for t in members]
            ).to(device)
            groups.append(
                {
                    "params": list(self.projections.parameters()),
                    "lr": float(self.stage_cfg["lr"]),
                    "weight_decay": float(self.stage_cfg["weight_decay"]),
                }
            )
        self.optim = torch.optim.AdamW(groups)
        # Snapshots da média EMA em frações dos passos, para a sopa de pesos.
        self.snapshots = {
            max(1, round(f * self.steps)) for f in self.stage_cfg.get("soup_snapshots", [])
        }
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.optim, _schedule(int(self.stage_cfg["warmup_steps"]), self.steps)
        )
        dtype = getattr(torch, str(self.run_cfg["amp_dtype"]))
        self.autocast = (
            (lambda: torch.autocast("cuda", dtype=dtype)) if device.type == "cuda" else nullcontext
        )

    def _view(self, rgb, depth, landmarks, augment: bool, flip: bool = False):
        if self.stage == "landmarks":
            if augment:
                return self.augment.landmarks_only(landmarks)
            return self.eval_view.landmarks_only(landmarks, flip)
        view = (
            self.augment(rgb, depth, landmarks)
            if augment
            else self.eval_view(rgb, depth, landmarks, flip)
        )
        if self.stage == "student_ce" or (self.stage == "teacher" and not self._uses_depth()):
            view.pop("depth")
        return view

    def _uses_depth(self) -> bool:
        return bool(self.config["model"]["teacher"]["depth"])

    def _loss(self, inputs: dict, labels: torch.Tensor) -> torch.Tensor:
        smoothing = float(self.stage_cfg.get("label_smoothing", 0.0))
        if self.stage != "student_kd":
            with self.autocast():
                logits = self.model(**inputs)
            return mixed_cross_entropy(logits, labels, smoothing=smoothing)
        alpha = float(self.stage_cfg["mixup_alpha"])
        lam, perm = 1.0, None
        if alpha > 0:
            inputs, lam, perm = mixup(inputs, alpha, self.mix_gen, dominant=("landmarks",))
        temperature = float(self.stage_cfg["temperature"])
        with self.autocast():
            with torch.no_grad():
                if self.projections is not None:
                    teacher_logits, teacher_feats = self.teacher(**inputs, return_features=True)
                else:
                    teacher_logits = self.teacher(**inputs)
            if self.aux_weight > 0 or self.projections is not None:
                logits, fused, aux = self.model(**inputs, return_parts=True)
            else:  # receita padrão: exatamente o caminho de antes
                logits, fused, aux = self.model(**inputs), None, None
        loss = kd_kl(logits, teacher_logits, temperature)
        if self.aux_weight > 0 and aux is not None:
            # A cabeça só dos pontos aprende do professor dela (o modelo de pontos
            # externo, se houver) ou do mesmo professor da principal.
            target = teacher_logits
            if self.aux_teacher is not None:
                with torch.no_grad(), self.autocast():
                    target = self.aux_teacher(**inputs)
            loss = loss + self.aux_weight * kd_kl(aux, target, temperature)
        if self.projections is not None:
            feats = teacher_feats if isinstance(teacher_feats, list) else [teacher_feats]
            align = [
                1 - F.cosine_similarity(p(fused.float()), t.float(), dim=1).mean()
                for p, t in zip(self.projections, feats, strict=True)
            ]
            loss = loss + self.feature_kd * torch.stack(align).mean()
        alpha_hard = float(self.stage_cfg["alpha_hard"])
        if alpha_hard > 0:
            hard = mixed_cross_entropy(logits, labels, lam, perm, smoothing)
            loss = (1 - alpha_hard) * loss + alpha_hard * hard
        return loss

    @torch.no_grad()
    def predict(self, model: nn.Module, split: InMemorySplit, flip: bool = False) -> np.ndarray:
        model.eval()
        size = int(self.run_cfg["eval_batch_size"])
        out = []
        for start in range(0, len(split), size):
            index = range(start, min(start + size, len(split)))
            rgb, depth, landmarks, _ = split.batch(list(index), self.device)
            view = self._view(rgb, depth, landmarks, augment=False, flip=flip)
            with self.autocast():
                out.append(model(**model_inputs(view, self.range_mm)).float().cpu())
        return torch.cat(out).numpy()

    def evaluate(self, model: nn.Module, splits: dict, flips: bool = False) -> dict:
        results = {}
        for name, split in splits.items():
            labels, pauses = split.labels.numpy(), split.pause_ids.numpy()
            results[name] = summarize(self.predict(model, split), labels, pauses)
            if flips:
                results[f"{name}_espelhado"] = summarize(
                    self.predict(model, split, flip=True), labels, pauses
                )
        return results

    def save_predictions(self, model: nn.Module, splits: dict) -> None:
        for name, split in splits.items():
            for flip in (False, True):
                logits = self.predict(model, split, flip=flip)
                suffix = "_espelhado" if flip else ""
                np.savez_compressed(
                    self.run_dir / f"pred_{name}{suffix}.npz",
                    keys=np.array(split.keys),
                    logits=logits.astype(np.float16),
                    labels=split.labels.numpy(),
                    codes=split.codes.numpy(),
                    pause_ids=split.pause_ids.numpy(),
                )

    def fit(self) -> dict:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(self.run_dir / "tb"))
        curve = (self.run_dir / "curve.jsonl").open("w", encoding="utf-8")
        batches = _batches(self.sampler, int(self.stage_cfg["batch_size"]))
        clip = float(self.run_cfg["grad_clip"])
        every = int(self.run_cfg["eval_every"])
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        seen, started, eval_s, losses = 0, time.perf_counter(), 0.0, []
        for step in range(1, self.steps + 1):
            self.model.train()
            index, epoch = next(batches)
            rgb, depth, landmarks, labels = self.train.batch(index, self.device)
            with torch.no_grad():
                view = self._view(rgb, depth, landmarks, augment=True)
                inputs = model_inputs(view, self.range_mm)
            loss = self._loss(inputs, labels)
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optim.step()
            self.sched.step()
            self.ema.update_parameters(self.model)
            if step in self.snapshots:
                torch.save(self.ema.module.state_dict(), self.run_dir / f"ema_step{step}.pt")
            if self.on_step is not None:
                self.on_step(self, step)
            seen += len(index)
            losses.append(loss.item())
            if step % every and step != self.steps:
                continue
            elapsed = time.perf_counter() - started - eval_s
            row = {
                "step": step,
                "epoch": epoch,
                "loss": round(float(np.mean(losses)), 4),
                "lr": self.sched.get_last_lr()[0],
                "imgs_per_s": round(seen / elapsed, 1),
                "elapsed_s": round(elapsed, 1),
            }
            if self.device.type == "cuda":
                row["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(self.device) / 2**30, 2)
            tick = time.perf_counter()
            row["eval"] = self.evaluate(self.ema.module, self.evals)
            eval_s += time.perf_counter() - tick
            losses = []
            for name, value in row.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"train/{name}", value, step)
            for split, metrics in row["eval"].items():
                for name, value in metrics.items():
                    writer.add_scalar(f"{split}/{name}", value, step)
            curve.write(json.dumps(row) + "\n")
            curve.flush()
            print(_progress(row), flush=True)
        train_s = time.perf_counter() - started - eval_s
        curve.close()
        writer.close()
        torch.save(self.ema.module.state_dict(), self.run_dir / "last.pt")
        everything = {**self.evals, **self.holdout}
        final = self.evaluate(self.ema.module, everything, flips=True)
        self.save_predictions(self.ema.module, everything)
        return {
            "steps": self.steps,
            "checkpoint": "last.pt",
            "metrics": final,
            "imgs_per_s": round(seen / train_s, 1),
            "train_s": round(train_s, 1),
            "peak_mem_gb": row.get("peak_mem_gb"),
        }


def _samples_of(split: InMemorySplit) -> dict[str, dict]:
    """Clipe e pausa de cada chave, no formato que o `PauseSampler` espera."""
    pauses = split.pause_ids.tolist()
    return {key: {"clip": str(pause), "pause": 0} for key, pause in zip(split.keys, pauses)}


def _progress(row: dict) -> str:
    parts = [f"passo {row['step']}", f"perda {row['loss']:.3f}", f"{row['imgs_per_s']:.0f} img/s"]
    if "peak_mem_gb" in row:
        parts.append(f"{row['peak_mem_gb']:.1f} GB")
    for split, metrics in row["eval"].items():
        parts.append(f"{split} pausa {metrics['pause_acc']:.3f} frame {metrics['frame_acc']:.3f}")
    return " | ".join(parts)
