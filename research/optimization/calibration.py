"""Temperatura de calibração ajustada nas predições fora do fold do LOSO.

O modelo final (`--all-actors`) não tem conjunto de validação e o C6 não pode ser
usado para escolher nada. Cada `pred_val.npz` do LOSO, porém, é de um ator que aquele
modelo nunca viu: juntos, os 5 folds dão predições honestas para ajustar T por NLL.

O ajuste é por pausa (média dos logits dos frames), a mesma unidade que o limiar do
app enxerga. O quantizador dobra 1/T nos pesos e no bias da cabeça: o app recebe os
logits já calibrados.

Com T fixo, o limiar padrão do app sai das mesmas predições: o maior recall nas CMs
da demo com falso aceite (pausa de fora do vocabulário aceita como da demo) até
`--max-false-accept`, com a regra do app (argmax nas 61, aceita se é da demo e passa
do limiar).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.config import load_config
from research.training.aggregate import find_runs
from research.training.metrics import expected_calibration_error, pause_logits, softmax


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = softmax(logits / temperature)
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """T que minimiza a NLL, numa grade de 0,30 a 4,00 em passos de 0,01."""
    grid = np.arange(0.30, 4.0, 0.01)
    return float(round(grid[np.argmin([nll(logits, labels, t) for t in grid])], 2))


def threshold_sweep(
    probs: np.ndarray, codes: np.ndarray, demo: list[int], grid: np.ndarray
) -> list[dict]:
    """Recall nas CMs da demo e falso aceite das de fora, para cada limiar."""
    pred, conf = probs.argmax(1), probs.max(1)
    inside = np.isin(codes, demo)
    rows = []
    for t in grid:
        accepted = np.isin(pred, demo) & (conf >= t)
        rows.append(
            {
                "threshold": round(float(t), 2),
                "recall": round(
                    float((accepted & inside & (pred == codes)).sum() / inside.sum()), 4
                ),
                "false_accept": round(float((accepted & ~inside).sum() / (~inside).sum()), 4),
            }
        )
    return rows


def out_of_fold(runs_dir: Path, stage: str, vocab: str = "full") -> tuple[np.ndarray, np.ndarray]:
    """Logits e rótulos por pausa dos 5 folds juntos."""
    runs = find_runs(runs_dir, stage, vocab)
    if not runs:
        raise SystemExit(f"nenhuma execução LOSO de {stage} ({vocab})")
    pooled, labels = [], []
    for run_dir in runs.values():
        data = np.load(run_dir / "pred_val.npz")
        logits, pause_labels = pause_logits(
            data["logits"].astype(np.float64), data["labels"], data["pause_ids"]
        )
        pooled.append(logits)
        labels.append(pause_labels)
    return np.concatenate(pooled), np.concatenate(labels)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ajusta T nas predições fora do fold do LOSO.")
    parser.add_argument("--runs-dir", type=Path, default=Path("research/experiments/runs"))
    parser.add_argument("--stage", default="student_kd")
    parser.add_argument("--vocab", default="full")
    parser.add_argument("--out", type=Path, default=Path("research/experiments/aggregate"))
    parser.add_argument("--config", type=Path, default=Path("configs/distillation.yaml"))
    parser.add_argument("--max-false-accept", type=float, default=0.03)
    args = parser.parse_args(argv)
    logits, labels = out_of_fold(args.runs_dir, args.stage, args.vocab)
    temperature = fit_temperature(logits, labels)
    report = {
        "stage": args.stage,
        "vocab": args.vocab,
        "pauses": len(labels),
        "temperature": temperature,
        "ece_t1": round(expected_calibration_error(softmax(logits), labels), 4),
        "ece_t": round(expected_calibration_error(softmax(logits / temperature), labels), 4),
        "nll_t1": round(nll(logits, labels, 1.0), 4),
        "nll_t": round(nll(logits, labels, temperature), 4),
    }
    if args.vocab == "full":
        # Com o vocabulário completo, rótulo = código da base.
        demo = [int(c) for c in load_config(args.config)["vocabularies"]["demo"]["codes"]]
        sweep = threshold_sweep(
            softmax(logits / temperature), labels, demo, np.arange(0.30, 0.951, 0.05)
        )
        ok = [r for r in sweep if r["false_accept"] <= args.max_false_accept]
        report["max_false_accept"] = args.max_false_accept
        report["threshold"] = max(ok, key=lambda r: r["recall"]) if ok else None
        report["threshold_sweep"] = sweep
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"temperature_{args.vocab}_{args.stage}.json"
    path.write_text(json.dumps(report, indent=1), "utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
