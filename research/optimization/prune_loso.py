"""Poda pós-treino nos 5 folds do LOSO, sem ajuste fino: L1 uniforme e LAMP a 50%.

Denso e podados são medidos pelo mesmo caminho do `eval_tflite` (vista do app, FP32
na CPU, T = 1), para o McNemar comparar as mesmas pausas. Os logits por quadro ficam
em `<out>/<modelo>_f<fold>.npz` e servem de cache; o resumo traz média ± dp por
pausa, quadro e F1, o McNemar contra o denso e a temperatura ajustada depois da poda
nas 1.160 pausas fora do fold.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

from research.config import load_config
from research.optimization.calibration import fit_temperature, nll
from research.optimization.pruner import prune_run
from research.training.aggregate import find_runs
from research.training.metrics import (
    expected_calibration_error,
    pause_logits,
    softmax,
    summarize,
)

CRITERIA = ("l1_uniform", "lamp")


def pruned_dir(pruned_root: Path, run: Path, criterion: str, config_path: Path) -> Path:
    """O diretório podado do run (ft0, corte único), criado se ainda não existe."""
    out = pruned_root / f"{run.name}_poda50_{criterion}_oneshot_ft0"
    if not out.exists():
        compression = {
            **load_config(config_path)["compression"],
            "prune_amount": 0.5,
            "finetune_epochs": 0,
            "criterion": criterion,
            "schedule": "oneshot",
        }
        out = prune_run(config_path, run, compression)
    return out


def fold_logits(run_dir: Path, cache: Path) -> dict:
    """Logits por quadro do split de validação pela vista do app, com cache em npz."""
    from research.optimization.eval_tflite import torch_logits
    from research.optimization.quantizer import app_views, load_split, load_student

    model, record = load_student(run_dir)
    split, hand_side = load_split(record, "val")
    if cache.exists():
        logits = np.load(cache)["logits"]
    else:
        logits = torch_logits(model, list(app_views(split, hand_side, record["config"], False)))
        np.savez_compressed(
            cache,
            logits=logits,
            labels=split.labels.numpy(),
            codes=split.codes.numpy(),
            pause_ids=split.pause_ids.numpy(),
        )
    return {
        "logits": logits,
        "labels": split.labels.numpy(),
        "pause_ids": split.pause_ids.numpy(),
    }


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """(só a acerta, só b acerta, p bilateral exato) sobre as mesmas pausas."""
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return only_a, only_b, p


def calibration(logits: np.ndarray, labels: np.ndarray, dense_t: float) -> dict:
    t = fit_temperature(logits, labels)
    row = {
        "T": t,
        "nll_t1": nll(logits, labels, 1.0),
        "nll_t": nll(logits, labels, t),
        "ece_t1": expected_calibration_error(softmax(logits), labels),
        "ece_t": expected_calibration_error(softmax(logits / t), labels),
        "ece_t_denso": expected_calibration_error(softmax(logits / dense_t), labels),
        "nll_t_denso": nll(logits, labels, dense_t),
    }
    return {k: round(float(v), 4) for k, v in row.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Poda pós-treino nos 5 folds do LOSO.")
    parser.add_argument("--config", type=Path, default=Path("configs/distillation.yaml"))
    parser.add_argument("--runs-dir", type=Path, default=Path("research/experiments/runs"))
    parser.add_argument("--pruned-dir", type=Path, default=Path("research/experiments/pruned"))
    parser.add_argument(
        "--out", type=Path, default=Path("research/experiments/aggregate/poda_loso_ft0")
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    dense = find_runs(args.runs_dir, "student_kd", 0, None)
    if sorted(dense) != [0, 1, 2, 3, 4]:
        raise SystemExit(f"faltam folds do student_kd: {sorted(dense)}")
    models = {"denso": dense}
    for criterion in CRITERIA:
        models[criterion] = {
            f: pruned_dir(args.pruned_dir, run, criterion, args.config) for f, run in dense.items()
        }

    folds, pooled = {}, {}
    for name, runs in models.items():
        per_fold = [fold_logits(d, args.out / f"{name}_f{f}.npz") for f, d in runs.items()]
        metrics = [summarize(p["logits"], p["labels"], p["pause_ids"]) for p in per_fold]
        folds[name] = {
            k: [m[k] for m in metrics] for k in ("pause_acc", "frame_acc", "pause_macro_f1")
        }
        parts = [pause_logits(p["logits"], p["labels"], p["pause_ids"]) for p in per_fold]
        pooled[name] = (
            np.concatenate([z for z, _ in parts]),
            np.concatenate([y for _, y in parts]),
        )

    correct = {n: z.argmax(1) == y for n, (z, y) in pooled.items()}
    comparisons = {
        c: dict(zip(("so_denso", "so_podado", "p"), mcnemar(correct["denso"], correct[c])))
        for c in CRITERIA
    }
    dense_t = fit_temperature(*pooled["denso"])
    report = {
        "folds": folds,
        "mcnemar": comparisons,
        "calibracao": {n: calibration(*pooled[n], dense_t) for n in ("denso", "lamp")},
        "pruned_dirs": {f"{c}_f{f}": str(d) for c in CRITERIA for f, d in models[c].items()},
    }
    (args.out / "resumo.json").write_text(json.dumps(report, indent=1), "utf-8")
    for name, values in folds.items():
        acc = np.array(values["pause_acc"])
        print(f"{name:11s} {acc.mean():.2%} ± {acc.std(ddof=1):.2%}")
    print(json.dumps({"mcnemar": comparisons, "calibracao": report["calibracao"]}))


if __name__ == "__main__":
    main()
