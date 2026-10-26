"""Orquestra um estágio de treino: dados, modelo, procedência e resultado.

Dois modos, e nenhum escolhe checkpoint pelo conjunto medido (passos fixos, último):
- fold (`--fold k`): treina nos atores dos folds menos um e mede esse um. A média
  dos folds é a estimativa de generalização para uma pessoa nova e a régua dos
  hiperparâmetros.
- todos (`--all-actors`): treina em todos os atores dos folds com a receita já
  congelada e mede o C6 uma única vez, no fim.

A destilação exige um professor do mesmo fold e modo: um professor que viu o ator
medido vazaria esse ator para o aluno.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from research.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_path(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_sha256() -> str:
    """Hash do código e das configs; o projeto ainda não é um repositório git."""
    digest = hashlib.sha256()
    files = sorted((REPO_ROOT / "research").rglob("*.py")) + sorted(
        (REPO_ROOT / "configs").glob("*.yaml")
    )
    for path in files:
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _versions() -> dict:
    import torch

    versions = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "timm", "numpy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    versions["cuda"] = torch.version.cuda
    versions["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return versions


def _load_teacher(run_dir: Path, fold: int | None, device, stage: str = "teacher"):
    from research.training.loop import build_model

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if record["stage"] != stage:
        raise ValueError(f"{run_dir} não é um {stage} ({record['stage']})")
    if record["fold"] != fold:
        raise ValueError(f"professor do fold {record['fold']} não serve ao fold {fold}")
    import torch

    teacher = build_model(stage, record["config"], pretrained=False)
    teacher.load_state_dict(torch.load(run_dir / "last.pt", map_location=device))
    return teacher, record


def ensemble_teacher(teachers: list):
    """Média dos logits de vários professores do mesmo fold (KD multi-professor)."""
    import torch
    from torch import nn

    class EnsembleTeacher(nn.Module):
        def __init__(self, members):
            super().__init__()
            self.members = nn.ModuleList(members)

        def forward(self, return_features: bool = False, **inputs):
            if not return_features:
                return torch.stack([t(**inputs) for t in self.members]).mean(dim=0)
            outs = [t(return_features=True, **inputs) for t in self.members]
            # Os espaços de features dos professores não se alinham: vão em lista.
            return torch.stack([o[0] for o in outs]).mean(dim=0), [o[1] for o in outs]

    return EnsembleTeacher(teachers)


def _fewer_actors(fold: dict, samples: dict, n: int, seed: int) -> dict:
    """O fold treinando em `n` dos seus atores; a semente gira quais (curva por sinalizantes)."""
    actors = fold["train_actors"]
    if not 1 <= n <= len(actors):
        raise ValueError(f"n_train_actors={n} fora de 1..{len(actors)}")
    kept = [actors[(seed + i) % len(actors)] for i in range(n)]
    return {
        **fold,
        "train_actors": kept,
        "train": [key for key in fold["train"] if samples[key]["actor"] in kept],
    }


def run(
    config_path: Path,
    stage: str,
    fold: int | None,
    seed: int | None = None,
    teacher_run: Path | None = None,
    steps: int | None = None,
    name: str | None = None,
    device: str | None = None,
    n_train_actors: int | None = None,
    *,
    config: dict | None = None,
    model=None,
    runs_dir: Path | None = None,
    extra: dict | None = None,
    on_step=None,
    aux_teacher_run: Path | None = None,
) -> Path:
    """Treina e avalia um estágio. `fold=None` é o modo todos (mede o C6). Devolve o diretório.

    Para o fine-tuning da poda: `config` substitui o lido de `config_path` (que só fica
    na procedência), `model` substitui o modelo novo, `runs_dir` tira a execução da pasta
    que a agregação lê, `extra` entra no run.json e `on_step` vai para o Trainer.
    """
    import torch

    from research.data.augment import load_backgrounds
    from research.data.splits import all_actors_split, loso_folds
    from research.data.vocab import resolve_vocab
    from research.ingestion.lmdb_engine import LMDBStore
    from research.training.data import InMemorySplit
    from research.training.loop import Trainer, build_model

    config_path = Path(config_path)
    config = load_config(config_path) if config is None else config
    vocab_info = resolve_vocab(config)
    config["model"]["num_classes"] = vocab_info["num_classes"]
    seed = int(config["run"]["seed"] if seed is None else seed)
    data_config_path = _repo_path(config["paths"]["data_config"])
    data_config = load_config(data_config_path)
    lmdb_path = _repo_path(data_config["paths"]["lmdb_path"])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    where = "todos" if fold is None else f"f{fold}"
    if n_train_actors is not None:
        where += f"_n{n_train_actors}"
    run_dir = _repo_path(runs_dir or config["paths"]["runs_dir"]) / (
        name or f"{stamp}_{vocab_info['name']}_{stage}_{where}_s{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    teacher, teacher_record = None, None
    if stage == "student_kd":
        if not teacher_run:
            raise ValueError("a destilação precisa de --teacher-run")
        runs = list(teacher_run) if isinstance(teacher_run, (list, tuple)) else [teacher_run]
        loaded = [_load_teacher(_repo_path(r), fold, device) for r in runs]
        teacher_record = loaded[0][1]
        teacher = loaded[0][0] if len(loaded) == 1 else ensemble_teacher([t for t, _ in loaded])
        teacher_run = str(runs[0]) if len(runs) == 1 else [str(r) for r in runs]
    aux_teacher = None
    if aux_teacher_run is not None:
        # Professor da cabeça auxiliar: o modelo só de pontos do mesmo fold.
        if stage != "student_kd":
            raise ValueError("--aux-teacher-run só vale na destilação")
        aux_teacher, _ = _load_teacher(_repo_path(aux_teacher_run), fold, device, stage="landmarks")

    label_of = vocab_info["label_of"]
    with LMDBStore(lmdb_path, int(data_config["lmdb"]["map_size"]), readonly=True) as store:
        meta = store.read_meta()
        split = config["split"]
        holdout = {}
        if fold is None:
            chosen = all_actors_split(meta["samples"], split["actors"], split["ood"])
            evals = {}
            holdout = {"ood": InMemorySplit(store, chosen["ood"], meta, label_of)}
            roles = {"train_actors": chosen["train_actors"], "ood_actors": chosen["ood_actors"]}
        else:
            chosen = loso_folds(meta["samples"], split["actors"], split["ood"])[fold]
            if n_train_actors is not None:
                chosen = _fewer_actors(chosen, meta["samples"], n_train_actors, seed)
            evals = {"val": InMemorySplit(store, chosen["val"], meta, label_of)}
            roles = {"train_actors": chosen["train_actors"], "val_actor": chosen["val_actor"]}
        train = InMemorySplit(store, chosen["train"], meta, label_of)
    hand_side = int(meta["layout"]["size"]) / float(meta["crop_rule"]["context_scale"])

    record = {
        "stage": stage,
        "fold": fold,
        "seed": seed,
        "vocab": {k: v for k, v in vocab_info.items() if k != "label_of"},
        **roles,
        "n_train_actors": n_train_actors,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": sys.argv,
        "config": config,
        "provenance": {
            "config_sha256": _sha256(config_path),
            "data_config_sha256": _sha256(data_config_path),
            "lmdb_manifest_sha256": _sha256(lmdb_path / "manifest.json"),
            "lmdb_config_sha256": meta.get("config_sha256"),
            "code_sha256": code_sha256(),
            "teacher_run": None if teacher_record is None else teacher_run,
            "aux_teacher_run": None if aux_teacher_run is None else str(aux_teacher_run),
            "versions": _versions(),
        },
        **(extra or {}),
        "sizes": {
            "train_frames": len(train),
            "train_pauses": train.n_pauses,
            **{f"{n}_pauses": s.n_pauses for n, s in {**evals, **holdout}.items()},
        },
    }
    (run_dir / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), "utf-8")
    (run_dir / "labels.txt").write_text("\n".join(vocab_info["names"]) + "\n", "utf-8")
    print(f"{run_dir.name}: {record['sizes']}", flush=True)

    pretrained = bool(config["model"].get("pretrained", True))
    if model is None:
        model = build_model(stage, config, pretrained)
    photos = load_backgrounds(
        _repo_path(config["augment"]["background"]["photo_dir"]), config["view"]["input_size"]
    )
    trainer = Trainer(
        stage,
        config,
        model,
        train,
        evals,
        hand_side,
        device,
        seed,
        run_dir,
        teacher=teacher,
        backgrounds=photos,
        steps=steps,
        holdout=holdout,
        on_step=on_step,
        aux_teacher=aux_teacher,
    )
    result = trainer.fit()
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False), flush=True)
    return run_dir


def main(argv: list[str] | None = None) -> None:
    from research.training.loop import STAGES

    parser = argparse.ArgumentParser(description="Treina um estágio num fold LOSO.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--fold", type=int, help="ator de validação: índice em split.actors")
    where.add_argument(
        "--all-actors", action="store_true", help="treina em todos os atores e mede o C6, uma vez"
    )
    parser.add_argument("--n-train-actors", type=int, help="treina só com N atores do fold")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--teacher-run",
        type=Path,
        nargs="+",
        help="diretório(s) do professor; vários = média dos logits (KD multi-professor)",
    )
    parser.add_argument("--steps", type=int, help="sobrepõe stages.<estágio>.steps")
    parser.add_argument("--name", help="nome do diretório da execução")
    parser.add_argument(
        "--aux-teacher-run",
        type=Path,
        help="modelo só de pontos (estágio landmarks) que ensina a cabeça auxiliar do aluno",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help="pasta das execuções (padrão: paths.runs_dir); pilotos fora da agregação LOSO",
    )
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    run(
        args.config,
        args.stage,
        None if args.all_actors else args.fold,
        args.seed,
        args.teacher_run,
        args.steps,
        args.name,
        args.device,
        args.n_train_actors,
        runs_dir=args.runs_dir,
        aux_teacher_run=args.aux_teacher_run,
    )


if __name__ == "__main__":
    main()
