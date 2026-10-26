"""Agrega os 5 folds LOSO de um estágio: média ± desvio e confusão somada.

A unidade é a pausa (média dos logits dos seus frames). Para cada fold vale a
execução mais recente com `result.json` do estágio e da semente pedidos, nas 61 classes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.training.metrics import pause_logits


def find_runs(
    runs_dir: Path,
    stage: str,
    seed: int | None = None,
    n_train_actors: int | None = None,
) -> dict:
    """{fold: diretório} com a execução mais recente de cada fold."""
    found: dict[int, Path] = {}
    for run_json in sorted(Path(runs_dir).glob("*/run.json")):
        run_dir = run_json.parent
        if not (run_dir / "result.json").is_file():
            continue
        record = json.loads(run_json.read_text(encoding="utf-8"))
        if "vocab" not in record or record["stage"] != stage or record["fold"] is None:
            continue
        if record["vocab"]["name"] != "full" or (seed is not None and record["seed"] != seed):
            continue
        if record.get("n_train_actors") != n_train_actors:
            continue
        found[int(record["fold"])] = run_dir
    return dict(sorted(found.items()))


def load_pauses(run_dir: Path, split: str = "val", flip: bool = False) -> dict:
    """Logits médios, rótulo, código da base e ator de cada pausa."""
    suffix = "_espelhado" if flip else ""
    data = np.load(run_dir / f"pred_{split}{suffix}.npz")
    logits = data["logits"].astype(np.float32)
    pooled, labels = pause_logits(logits, data["labels"], data["pause_ids"])
    codes = pause_logits(logits, data["codes"], data["pause_ids"])[1]
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return {
        "logits": pooled,
        "labels": labels,
        "codes": codes,
        "vocab": record["vocab"],
        "actor": record.get("val_actor"),
    }


def confusion(folds: list[dict], n: int) -> np.ndarray:
    """Matriz (verdadeiro, previsto) de pausas somada nos folds, no espaço dos rótulos."""
    matrix = np.zeros((n, n), dtype=np.int64)
    for pauses in folds:
        np.add.at(matrix, (pauses["labels"], pauses["logits"].argmax(axis=1)), 1)
    return matrix


def _mean_std(values: list[float]) -> dict:
    array = np.array([v for v in values if v is not None], dtype=float)
    return {"mean": round(float(array.mean()), 4), "std": round(float(array.std(ddof=1)), 4)}


def summarize_stage(runs: dict) -> dict:
    per_fold = {}
    for fold, run_dir in runs.items():
        metrics = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))["metrics"]
        record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        per_fold[fold] = {
            "actor": record["val_actor"],
            "run": run_dir.name,
            **{k: metrics["val"][k] for k in ("pause_acc", "frame_acc", "pause_macro_f1")},
            "frame_ece": metrics["val"]["frame_ece"],
            "pause_acc_espelhado": metrics["val_espelhado"]["pause_acc"],
        }
    keys = [k for k in next(iter(per_fold.values())) if k not in ("actor", "run")]
    return {
        "folds": per_fold,
        "mean": {k: _mean_std([f[k] for f in per_fold.values()]) for k in keys},
    }


def top_confusions(matrix: np.ndarray, names: list[str], top: int) -> list[dict]:
    """Pares (a, b) com mais pausas trocadas nos dois sentidos."""
    pairs = []
    for a in range(len(matrix)):
        for b in range(a + 1, len(matrix)):
            swapped = int(matrix[a, b] + matrix[b, a])
            if swapped:
                pairs.append({"a": names[a], "b": names[b], "a_as_b": int(matrix[a, b]),
                              "b_as_a": int(matrix[b, a]), "total": swapped})  # fmt: skip
    return sorted(pairs, key=lambda p: -p["total"])[:top]


def plot_confusion(matrix: np.ndarray, names: list[str], out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    side = max(6, len(names) * 0.22)
    fig, ax = plt.subplots(figsize=(side, side))
    ax.imshow(rates, cmap="magma_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("previsto")
    ax.set_ylabel("verdadeiro")
    ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            if matrix[i, j] and i != j:
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=5, color="b")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


NAMES = [f"CM{c + 1}" for c in range(61)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Agrega os folds LOSO de um estágio.")
    parser.add_argument("--runs-dir", type=Path, default=Path("research/experiments/runs"))
    parser.add_argument("--stage", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-train-actors", type=int, help="só execuções da curva com N atores")
    parser.add_argument("--compare", help="outro estágio: diferença pareada por fold")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("research/experiments/aggregate"))
    args = parser.parse_args(argv)

    runs = find_runs(args.runs_dir, args.stage, args.seed, args.n_train_actors)
    if not runs:
        raise SystemExit(f"nenhuma execução de {args.stage}")
    folds = [load_pauses(run_dir) for run_dir in runs.values()]
    report = {"stage": args.stage, "n_folds": len(runs)}
    report.update(summarize_stage(runs))
    matrix = confusion(folds, len(NAMES))
    report["pooled_pause_acc"] = round(float(np.trace(matrix) / matrix.sum()), 4)
    report["per_class_recall"] = {
        name: round(float(matrix[i, i] / max(matrix[i].sum(), 1)), 3)
        for i, name in enumerate(NAMES)
    }
    report["top_confusions"] = top_confusions(matrix, NAMES, args.top)

    if args.compare:
        other_runs = find_runs(args.runs_dir, args.compare, args.seed, args.n_train_actors)
        common = sorted(set(runs) & set(other_runs))
        diffs = []
        for fold in common:
            a = json.loads((runs[fold] / "result.json").read_text())["metrics"]["val"]
            b = json.loads((other_runs[fold] / "result.json").read_text())["metrics"]["val"]
            diffs.append(a["pause_acc"] - b["pause_acc"])
        report["compare"] = {
            "against": args.compare,
            "folds": common,
            "pause_acc_diff": [round(d, 4) for d in diffs],
            "mean_diff": round(float(np.mean(diffs)), 4) if diffs else None,
            "wins": int(sum(d > 0 for d in diffs)),
        }

    stem = f"full_{args.stage}"
    if args.n_train_actors is not None:
        stem += f"_n{args.n_train_actors}"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    pooled = report["pooled_pause_acc"]
    title = f"{args.stage}, {len(runs)} folds: pausa {pooled:.3f}"
    plot_confusion(matrix, NAMES, args.out / f"{stem}_confusao.png", title)
    print(json.dumps({k: report[k] for k in ("mean", "pooled_pause_acc")}, ensure_ascii=False))
    print(args.out / f"{stem}.json")


if __name__ == "__main__":
    main()
