"""Poda de conexões (não estruturada) do aluno e o fine-tuning que recupera a acurácia.

Só as camadas que aguentam: as pointwise 1×1 (groups=1) do MobileNetV4 e as lineares
do ramo dos pontos, com `prune_amount` dos pesos delas zerados. Três critérios:
`l1_uniform` (a mesma fração em cada camada, o baseline), `lamp` (Lee et al., 2021) e
`taylor` (|w·∇w| acumulado no próprio fine-tuning), os dois últimos com limiar global.
O corte é `oneshot` ou `gradual` (curva cúbica de Zhu & Gupta, 2017). Ficam inteiras
as depthwise (poucos pesos por canal, sem redundância), as convs k>1 do stem e dos
blocos fused, as normas, os biases e a cabeça.

O fine-tuning é a receita do run de origem (KD com o professor do mesmo fold), por
`finetune_epochs` épocas de pausas. O resultado vai para `paths.pruned_dir`, fora da
pasta que a agregação LOSO lê, com os zeros já gravados nos pesos: um `last.pt`
comum, que o quantizador lê como qualquer aluno.

Zeros soltos não encolhem o TFLite nem aceleram o XNNPACK (Int8 denso). O ganho é o
arquivo comprimido; tamanho e latência pedem poda estruturada (canais inteiros).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune

from research.config import load_config


def zero_pruned(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Devolve uma cópia dos pesos com zero onde a máscara é falsa.

    `mask` é booleana e tem o mesmo shape de `weights`. True conserva o peso.
    """
    if weights.shape != mask.shape:
        raise ValueError(f"máscara e pesos com shapes diferentes: {mask.shape} e {weights.shape}")
    pruned = np.array(weights, copy=True)
    pruned[~np.asarray(mask, dtype=bool)] = 0
    return pruned


def sparsity(weights: np.ndarray) -> float:
    """Fração de elementos exatamente zero no intervalo [0, 1]."""
    if weights.size == 0:
        raise ValueError("sparsity exige um tensor com ao menos um elemento")
    return float(np.count_nonzero(weights == 0) / weights.size)


CRITERIA = ("l1_uniform", "lamp", "taylor")
SCHEDULES = ("oneshot", "gradual")


def prunable(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Pointwise 1×1 densas e as lineares do ramo dos pontos; a cabeça fica de fora."""
    chosen = []
    for name, module in model.named_modules():
        pointwise = (
            isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1) and module.groups == 1
        )
        if pointwise or (isinstance(module, nn.Linear) and name.startswith("landmarks.")):
            chosen.append((name, module))
    return chosen


def attach_masks(model: nn.Module) -> nn.Module:
    """Reparametriza as camadas podáveis com máscara toda 1 (`weight = weight_orig × mask`)."""
    for _, module in prunable(model):
        if not prune.is_pruned(module):
            prune.custom_from_mask(module, "weight", torch.ones_like(module.weight))
            # O hook recalcula `weight` a cada forward; solto do grafo, o EMA consegue copiar.
            module.weight = module.weight.detach()
    return model


def effective_weight(module: nn.Module) -> torch.Tensor:
    if prune.is_pruned(module):
        return (module.weight_orig * module.weight_mask).detach()
    return module.weight.detach()


def lamp_scores(weight: torch.Tensor) -> torch.Tensor:
    """LAMP (Lee et al., ICLR 2021): w_u² / Σ_{v ≥ u} w_v², com os pesos em ordem crescente.

    O maior peso de cada camada vale 1, e a escala da camada some: um limiar global
    distribui a esparsidade entre as camadas sem esvaziar nenhuma.
    """
    squared = weight.flatten().float() ** 2
    order = squared.argsort()
    ranked = squared[order]
    tail = ranked.flip(0).cumsum(0).flip(0)
    scores = torch.empty_like(squared)
    scores[order] = ranked / tail.clamp_min(1e-12)
    return scores.view_as(weight)


def _drop_lowest(scores: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Máscara 0/1 que zera exatamente `sparsity` dos elementos de menor score."""
    flat = scores.flatten()
    mask = torch.ones_like(flat)
    mask[flat.argsort()[: round(sparsity * flat.numel())]] = 0
    return mask.view_as(scores)


def compute_masks(
    modules: list[nn.Module], sparsity: float, criterion: str, taylor=None
) -> list[torch.Tensor]:
    """Máscaras para `sparsity` das camadas juntas.

    `l1_uniform` corta a mesma fração em cada camada; `lamp` e `taylor` usam um limiar
    global sobre o score. Peso já zerado tem score zero e continua podado.
    """
    weights = [effective_weight(m) for m in modules]
    if criterion == "l1_uniform":
        return [_drop_lowest(w.abs(), sparsity) for w in weights]
    if criterion == "lamp":
        scores = [lamp_scores(w) for w in weights]
    elif criterion == "taylor":
        if taylor is None:
            raise ValueError("taylor precisa dos gradientes acumulados")
        scores = [t * (w != 0) for t, w in zip(taylor, weights, strict=True)]
    else:
        raise ValueError(f"critério desconhecido: {criterion}")
    flat = _drop_lowest(torch.cat([s.flatten() for s in scores]), sparsity)
    return list(flat.split([s.numel() for s in scores]))


def set_masks(model: nn.Module, masks: list[torch.Tensor]) -> None:
    for (_, module), mask in zip(prunable(model), masks, strict=True):
        module.weight_mask.copy_(mask.view_as(module.weight_mask))


def schedule_events(steps: int, amount: float, criterion: str, schedule: str) -> dict[int, float]:
    """{passo: esparsidade}. O passo 0 é aplicado antes do treino.

    oneshot: tudo de uma vez; Taylor espera 10% dos passos para acumular gradiente.
    gradual: curva cúbica (Zhu & Gupta, 2017) até 60% dos passos, em 10 cortes; o
    resto do fine-tuning já roda com a máscara final.
    """
    if schedule == "oneshot":
        return {max(1, steps // 10) if criterion == "taylor" else 0: amount}
    if schedule != "gradual":
        raise ValueError(f"cronograma desconhecido: {schedule}")
    end = max(1, round(0.6 * steps))
    marks = sorted({max(1, round(end * i / 10)) for i in range(1, 11)})
    return {t: amount * (1 - (1 - t / end) ** 3) for t in marks}


class MaskSchedule:
    """Gancho do Trainer: acumula |w·∇w| (Taylor) e corta nos passos do cronograma.

    A máscara nova vai para o modelo treinado e para a cópia EMA, a que é salva e medida.
    """

    def __init__(self, criterion: str, events: dict[int, float]) -> None:
        self.criterion = criterion
        self.events = events
        self.taylor: list[torch.Tensor] | None = None
        self.log: list[dict] = []

    def __call__(self, trainer, step: int) -> None:
        modules = [m for _, m in prunable(trainer.model)]
        if self.criterion == "taylor":
            saliency = [
                (effective_weight(m) * m.weight_orig.grad).abs().float()
                if m.weight_orig.grad is not None
                else torch.zeros_like(m.weight_orig, dtype=torch.float32)
                for m in modules
            ]
            if self.taylor is None:
                self.taylor = saliency
            else:
                self.taylor = [a + b for a, b in zip(self.taylor, saliency, strict=True)]
        if step in self.events:
            masks = compute_masks(modules, self.events[step], self.criterion, self.taylor)
            set_masks(trainer.model, masks)
            set_masks(trainer.ema.module, masks)
            self.log.append({"step": step, "sparsity": round(self.events[step], 4)})


def bake(model: nn.Module) -> nn.Module:
    """Grava a máscara nos pesos e tira a reparametrização (state_dict comum)."""
    for _, module in prunable(model):
        if prune.is_pruned(module):
            prune.remove(module, "weight")
    return model


def sparsity_report(model: nn.Module) -> dict:
    """Esparsidade nas camadas podadas, por camada e no modelo inteiro."""
    layers = {name: module.weight.detach() for name, module in prunable(model)}
    pruned_zero = sum(int((w == 0).sum()) for w in layers.values())
    pruned_size = sum(w.numel() for w in layers.values())
    total_zero = sum(int((p == 0).sum()) for p in model.parameters())
    total = sum(p.numel() for p in model.parameters())
    per_layer = {name: round(float((w == 0).float().mean()), 3) for name, w in layers.items()}
    return {
        "layers": len(layers),
        "pruned_params": pruned_size,
        "share_of_model": round(pruned_size / total, 4),
        "sparsity_pruned_layers": round(pruned_zero / pruned_size, 4),
        "sparsity_model": round(total_zero / total, 4),
        "layer_sparsity_min": min(per_layer.values()),
        "layer_sparsity_max": max(per_layer.values()),
        "per_layer": per_layer,
    }


def finetune_config(record: dict, compression: dict) -> dict:
    """A config do run de origem com passos, LR e EMA do fine-tuning."""
    config = copy.deepcopy(record["config"])
    stage = config["stages"][record["stage"]]
    epochs = float(compression["finetune_epochs"])
    steps = max(1, math.ceil(epochs * record["sizes"]["train_pauses"] / stage["batch_size"]))
    stage["steps"] = steps
    stage["warmup_steps"] = max(1, steps // 10)
    stage["lr"] = float(compression["finetune_lr"])
    # A média EMA do treino (0,999) mal sairia do ponto de partida em poucos passos.
    config["run"]["ema_decay"] = 1.0 - 1.0 / max(10.0, steps / 10)
    config["run"]["eval_every"] = max(1, steps // 4)
    config["compression"] = compression
    return config


def _pruned_name(run_dir: Path, compression: dict) -> str:
    amount = round(float(compression["prune_amount"]) * 100)
    return f"{run_dir.name}_poda{amount}_{compression['criterion']}_{compression['schedule']}"


def prune_post_training(settings: dict, run_dir: Path, record: dict, compression: dict) -> Path:
    """Poda sem fine-tuning (`finetune_epochs: 0`): sem dados, sem professor, sem GPU.

    Só para critérios de magnitude; Taylor precisa de gradiente. O run.json do aluno é
    copiado com a poda anotada, para o quantizador e o eval lerem o diretório como um run.
    """
    from research.train import _repo_path
    from research.training.loop import build_model

    criterion = compression["criterion"]
    if criterion == "taylor" or compression["schedule"] != "oneshot":
        raise ValueError("sem fine-tuning, só l1_uniform ou lamp em oneshot")
    out = _repo_path(settings["paths"]["pruned_dir"]) / f"{_pruned_name(run_dir, compression)}_ft0"
    out.mkdir(parents=True, exist_ok=False)
    model = build_model(record["stage"], record["config"], pretrained=False)
    model.load_state_dict(torch.load(run_dir / "last.pt", map_location="cpu"))
    attach_masks(model)
    modules = [m for _, m in prunable(model)]
    set_masks(model, compute_masks(modules, float(compression["prune_amount"]), criterion))
    bake(model)
    torch.save(model.state_dict(), out / "last.pt")
    record = {**record, "pruned_from": str(run_dir), "pruning": compression}
    (out / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), "utf-8")
    report = sparsity_report(model)
    (out / "pruning.json").write_text(json.dumps(report, indent=1), "utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_layer"}), flush=True)
    return out


def prune_run(
    config_path: Path, run_dir: Path, compression: dict, device: str | None = None
) -> Path:
    """Poda o `last.pt` de `run_dir` (com ou sem fine-tuning) e devolve o diretório novo."""
    from research.train import _repo_path, run
    from research.training.loop import build_model

    settings = load_config(config_path)
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if record["stage"] not in ("student_ce", "student_kd"):
        raise ValueError(f"{run_dir} não é um aluno ({record['stage']})")
    criterion, schedule = compression["criterion"], compression["schedule"]
    amount = float(compression["prune_amount"])
    if criterion not in CRITERIA or schedule not in SCHEDULES or not 0.0 < amount < 1.0:
        raise ValueError(f"poda inválida: {criterion}, {schedule}, {amount}")
    if float(compression["finetune_epochs"]) == 0:
        return prune_post_training(settings, run_dir, record, compression)
    config = finetune_config(record, compression)
    steps = config["stages"][record["stage"]]["steps"]
    events = schedule_events(steps, amount, criterion, schedule)

    model = build_model(record["stage"], config, pretrained=False)
    model.load_state_dict(torch.load(run_dir / "last.pt", map_location="cpu"))
    attach_masks(model)
    hook = MaskSchedule(criterion, events)
    if 0 in events:
        modules = [m for _, m in prunable(model)]
        set_masks(model, compute_masks(modules, events.pop(0), criterion))
        hook.log.append({"step": 0, "sparsity": amount})
    lr = float(compression["finetune_lr"])
    teacher = record["provenance"].get("teacher_run")
    out = run(
        config_path,
        record["stage"],
        record["fold"],
        record["seed"],
        None
        if teacher is None
        else [_repo_path(t) for t in teacher]
        if isinstance(teacher, list)
        else _repo_path(teacher),
        name=f"{_pruned_name(run_dir, compression)}_lr{lr:g}",
        device=device,
        vocab=record["vocab"]["name"],
        n_train_actors=record.get("n_train_actors"),
        config=config,
        model=model,
        runs_dir=_repo_path(settings["paths"]["pruned_dir"]),
        extra={"pruned_from": str(run_dir), "pruning": compression},
        on_step=hook,
    )

    # O Trainer salva a reparametrização (weight_orig e weight_mask); grava os zeros.
    final = attach_masks(build_model(record["stage"], config, pretrained=False))
    final.load_state_dict(torch.load(out / "last.pt", map_location="cpu"))
    bake(final)
    torch.save(final.state_dict(), out / "last.pt")
    report = {**sparsity_report(final), "events": hook.log}
    (out / "pruning.json").write_text(json.dumps(report, indent=1), "utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_layer"}), flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    from research.train import _repo_path

    parser = argparse.ArgumentParser(description="Poda o aluno e faz o fine-tuning.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="diretório do run do aluno")
    parser.add_argument("--amount", type=float, help="sobrepõe compression.prune_amount")
    parser.add_argument("--epochs", type=float, help="sobrepõe compression.finetune_epochs")
    parser.add_argument("--lr", type=float, help="sobrepõe compression.finetune_lr")
    parser.add_argument("--criterion", choices=CRITERIA, help="sobrepõe compression.criterion")
    parser.add_argument("--schedule", choices=SCHEDULES, help="sobrepõe compression.schedule")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    compression = dict(load_config(args.config)["compression"])
    for key, value in (
        ("prune_amount", args.amount),
        ("finetune_epochs", args.epochs),
        ("finetune_lr", args.lr),
        ("criterion", args.criterion),
        ("schedule", args.schedule),
    ):
        if value is not None:
            compression[key] = value
    prune_run(args.config, _repo_path(args.run), compression, args.device)


if __name__ == "__main__":
    main()
