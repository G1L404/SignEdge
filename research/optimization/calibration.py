"""Temperatura de calibração e decisão do app, nas predições fora do fold do LOSO.

O modelo final (`--all-actors`) não tem validação e o C6 não escolhe nada; cada
`pred_val.npz` do LOSO, porém, é de um ator que aquele modelo nunca viu, e os 5
folds juntos dão predições honestas para ajustar T por NLL, por pausa, a unidade que
o limiar do app enxerga. O quantizador dobra 1/T na cabeça.

`--decisao <dir>` mede a decisão do app sobre os logits do modelo podado (cache do
`prune_loso`): cobertura e acerto por limiar, top-k, recall por CM e pares trocados.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.training.aggregate import find_runs
from research.training.metrics import expected_calibration_error, pause_logits, softmax

THRESHOLDS = (0.30, 0.45, 0.70)
CITED_PAIRS = ((46, 47), (42, 43), (15, 38), (19, 32), (27, 28))


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = softmax(logits / temperature)
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """T que minimiza a NLL, numa grade de 0,30 a 4,00 em passos de 0,01."""
    grid = np.arange(0.30, 4.0, 0.01)
    return float(round(grid[np.argmin([nll(logits, labels, t) for t in grid])], 2))


def out_of_fold(runs_dir: Path, stage: str) -> tuple[np.ndarray, np.ndarray]:
    """Logits e rótulos por pausa dos 5 folds juntos."""
    runs = find_runs(runs_dir, stage)
    if not runs:
        raise SystemExit(f"nenhuma execução LOSO de {stage}")
    pooled, labels = [], []
    for run_dir in runs.values():
        data = np.load(run_dir / "pred_val.npz")
        logits, pause_labels = pause_logits(
            data["logits"].astype(np.float64), data["labels"], data["pause_ids"]
        )
        pooled.append(logits)
        labels.append(pause_labels)
    return np.concatenate(pooled), np.concatenate(labels)


def _pooled(cache: Path, model: str, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    parts = []
    for fold in range(5):
        data = np.load(cache / f"{model}_f{fold}.npz")
        logits = data["logits"].astype(np.float32) / temperature
        parts.append(pause_logits(logits, data["labels"], data["pause_ids"]))
    return np.concatenate([z for z, _ in parts]), np.concatenate([y for _, y in parts])


def decision_report(probs: np.ndarray, labels: np.ndarray) -> dict:
    """A regra do app nas 61 classes: aceita a top-1 se ela passa do limiar."""
    pred, conf = probs.argmax(1), probs.max(1)
    ranks = np.argsort(-probs, axis=1)
    confusion = np.zeros((61, 61), dtype=int)
    np.add.at(confusion, (labels, pred), 1)
    np.fill_diagonal(confusion, 0)
    swaps = confusion + confusion.T
    recall = {c + 1: round(float((pred[labels == c] == c).mean()), 4) for c in range(61)}
    pairs = sorted(
        ((int(swaps[i, j]), i + 1, j + 1) for i in range(61) for j in range(i + 1, 61)),
        reverse=True,
    )
    return {
        "pausas": int(len(labels)),
        "limiares": {
            f"{t:.2f}": {
                "cobertura": round(float((conf >= t).mean()), 4),
                "acerto_entre_respondidas": round(float((pred == labels)[conf >= t].mean()), 4),
            }
            for t in THRESHOLDS
        },
        "top_k": {
            f"top{k}": round(float((ranks[:, :k] == labels[:, None]).any(1).mean()), 4)
            for k in (1, 2, 3)
        },
        "recall_por_cm": recall,
        "recall_abaixo_de_50": [c for c, r in recall.items() if r < 0.5],
        "pares_citados_no_capitulo": {
            f"{a}/{b}": int(swaps[a - 1, b - 1]) for a, b in sorted(CITED_PAIRS)
        },
        "pares_mais_trocados": [{"cms": [a, b], "pausas": n} for n, a, b in pairs[:8]],
    }


def logit_scale(cache: Path) -> dict:
    """Quanto a poda encolheu os logits: inclinação da regressão podado ~ denso."""
    slopes, ratios = [], []
    for fold in range(5):
        dense = np.load(cache / f"denso_f{fold}.npz")["logits"].astype(np.float64)
        pruned = np.load(cache / f"lamp_f{fold}.npz")["logits"].astype(np.float64)
        dense -= dense.mean(1, keepdims=True)
        pruned -= pruned.mean(1, keepdims=True)
        slopes.append((pruned * dense).sum() / (dense * dense).sum())
        ratios.append(pruned.std() / dense.std())
    return {
        "inclinacao_regressao": round(float(np.mean(slopes)), 3),
        "razao_desvios": round(float(np.mean(ratios)), 3),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ajusta T nas predições fora do fold do LOSO.")
    parser.add_argument("--runs-dir", type=Path, default=Path("research/experiments/runs"))
    parser.add_argument("--stage", default="student_kd")
    parser.add_argument("--out", type=Path, default=Path("research/experiments/aggregate"))
    parser.add_argument(
        "--decisao", type=Path, help="cache do prune_loso: mede a decisão do modelo podado"
    )
    args = parser.parse_args(argv)

    if args.decisao:
        summary = json.loads((args.decisao / "resumo.json").read_text("utf-8"))
        temperature = summary["calibracao"]["lamp"]["T"]
        probs_logits, labels = _pooled(args.decisao, "lamp", temperature)
        report = {
            "fonte": str(args.decisao / "lamp_f*.npz"),
            "temperatura": temperature,
            "unidade": "pausa (média dos logits)",
            **decision_report(softmax(probs_logits), labels),
            "escala_logits_podado_vs_denso": {
                **logit_scale(args.decisao),
                "razao_temperaturas": round(temperature / summary["calibracao"]["denso"]["T"], 3),
            },
        }
        path = args.decisao / "decisao_61.json"
    else:
        logits, labels = out_of_fold(args.runs_dir, args.stage)
        temperature = fit_temperature(logits, labels)
        report = {
            "stage": args.stage,
            "pauses": len(labels),
            "temperature": temperature,
            "ece_t1": round(expected_calibration_error(softmax(logits), labels), 4),
            "ece_t": round(expected_calibration_error(softmax(logits / temperature), labels), 4),
            "nll_t1": round(nll(logits, labels, 1.0), 4),
            "nll_t": round(nll(logits, labels, temperature), 4),
        }
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"temperature_full_{args.stage}.json"
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False), "utf-8")
    print(path)


if __name__ == "__main__":
    main()
