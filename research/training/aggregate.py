"""Agrega os 5 folds LOSO de um estágio: média ± desvio, confusão somada e vocabulários.

A unidade é a pausa (média dos logits dos seus frames). Para cada fold vale a
execução mais recente com `result.json` do estágio, vocabulário e semente pedidos.

Vocabulário candidato (`--cms`), medido de dois jeitos num modelo de 61 classes:
- fechado: só pausas do vocabulário, argmax restrito às colunas dele (o teto, se
  a mão nunca fizesse outra configuração);
- aberto: argmax nas 61; uma configuração de fora prevista como de dentro é um
  falso aceite. É a referência para o modelo treinado com a classe "outra".
Num modelo treinado com vocabulário, só a medida aberta faz sentido.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.training.metrics import pause_logits

OTHER = -1


def find_runs(
    runs_dir: Path,
    stage: str,
    vocab: str = "full",
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
        if record["vocab"]["name"] != vocab or (seed is not None and record["seed"] != seed):
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


def predicted_codes(pauses: dict) -> np.ndarray:
    """Código previsto por pausa; OTHER quando o modelo responde "outra"."""
    pred = pauses["logits"].argmax(axis=1)
    codes = list(pauses["vocab"]["codes"]) + ([OTHER] if pauses["vocab"]["other"] else [])
    return np.array([codes[p] for p in pred])


def confusion(folds: list[dict], n: int) -> np.ndarray:
    """Matriz (verdadeiro, previsto) de pausas somada nos folds, no espaço dos rótulos."""
    matrix = np.zeros((n, n), dtype=np.int64)
    for pauses in folds:
        np.add.at(matrix, (pauses["labels"], pauses["logits"].argmax(axis=1)), 1)
    return matrix


def closed_accuracy(pauses: dict, codes: list[int]) -> float:
    if pauses["vocab"]["name"] != "full":
        raise ValueError("a medida fechada pede um modelo de 61 classes")
    inside = np.isin(pauses["codes"], codes)
    restricted = pauses["logits"][inside][:, codes]
    truth = pauses["codes"][inside]
    return float((np.array(codes)[restricted.argmax(axis=1)] == truth).mean())


def open_report(pauses: dict, codes: list[int]) -> dict:
    """Acertos dentro do vocabulário, rejeição do que é de fora e falsos aceites."""
    pred = predicted_codes(pauses)
    pred = np.where(np.isin(pred, codes), pred, OTHER)
    truth = np.where(np.isin(pauses["codes"], codes), pauses["codes"], OTHER)
    inside, outside, accepted = truth != OTHER, truth == OTHER, pred != OTHER
    recalls = [float((pred[truth == c] == c).mean()) for c in codes if (truth == c).any()]
    if outside.any():
        recalls.append(float((pred[outside] == OTHER).mean()))
    return {
        "vocab_recall": float((pred[inside] == truth[inside]).mean()),
        "other_rejected": float((pred[outside] == OTHER).mean()) if outside.any() else None,
        "false_accept": float(accepted[outside].mean()) if outside.any() else None,
        "precision": float((pred[accepted] == truth[accepted]).mean()) if accepted.any() else None,
        "balanced_acc": float(np.mean(recalls)),
    }


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


def greedy_vocab(folds: list[dict], keep: list[int], size: int, exclude: list[int]) -> list:
    """Adiciona a classe que mais mantém a acurácia fechada média dos folds."""
    chosen = list(keep)
    path = []
    while len(chosen) < size:
        best = None
        for code in range(61):
            if code in chosen or code in exclude:
                continue
            trial = chosen + [code]
            score = float(np.mean([closed_accuracy(f, trial) for f in folds]))
            if best is None or score > best[1]:
                best = (code, score)
        chosen.append(best[0])
        path.append({"cm": best[0] + 1, "closed_acc": round(best[1], 4)})
    return path


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


def _names(vocab: dict) -> list[str]:
    if vocab["name"] == "full":
        return [f"CM{c + 1}" for c in vocab["codes"]]
    return [f"CM{c + 1}" for c in vocab["codes"]] + (["outra"] if vocab["other"] else [])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Agrega os folds LOSO de um estágio.")
    parser.add_argument("--runs-dir", type=Path, default=Path("research/experiments/runs"))
    parser.add_argument("--stage", required=True)
    parser.add_argument("--vocab", default="full", help="vocabulário com que o modelo treinou")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-train-actors", type=int, help="só execuções da curva com N atores")
    parser.add_argument("--cms", type=int, nargs="+", help="vocabulário candidato, em números CM")
    parser.add_argument("--greedy", type=int, help="completa --cms até esse tamanho, gulosamente")
    parser.add_argument("--exclude", type=int, nargs="*", default=[], help="CMs fora da busca")
    parser.add_argument("--compare", help="outro estágio: diferença pareada por fold")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("research/experiments/aggregate"))
    args = parser.parse_args(argv)

    runs = find_runs(args.runs_dir, args.stage, args.vocab, args.seed, args.n_train_actors)
    if not runs:
        raise SystemExit(f"nenhuma execução de {args.stage} ({args.vocab})")
    folds = [load_pauses(run_dir) for run_dir in runs.values()]
    vocab = folds[0]["vocab"]
    names = _names(vocab)
    report = {"stage": args.stage, "vocab": args.vocab, "n_folds": len(runs)}
    report.update(summarize_stage(runs))
    matrix = confusion(folds, vocab["num_classes"])
    report["pooled_pause_acc"] = round(float(np.trace(matrix) / matrix.sum()), 4)
    report["per_class_recall"] = {
        name: round(float(matrix[i, i] / max(matrix[i].sum(), 1)), 3)
        for i, name in enumerate(names)
    }
    report["top_confusions"] = top_confusions(matrix, names, args.top)

    if args.cms:
        codes = [cm - 1 for cm in args.cms]
        candidate = {"cms": args.cms}
        if vocab["name"] == "full":
            closed = [closed_accuracy(f, codes) for f in folds]
            candidate["closed_acc"] = _mean_std(closed)
            candidate["closed_acc_folds"] = [round(v, 4) for v in closed]
            inside = [confusion_inside(f, codes) for f in folds]
            sub = np.sum(inside, axis=0)
            candidate["inside_confusions"] = top_confusions(
                sub, [f"CM{c + 1}" for c in codes], args.top
            )
            candidate["per_class_closed_recall"] = {
                f"CM{c + 1}": round(float(sub[i, i] / max(sub[i].sum(), 1)), 3)
                for i, c in enumerate(codes)
            }
        opened = [open_report(f, codes) for f in folds]
        candidate["open"] = {k: _mean_std([o[k] for o in opened]) for k in opened[0]}
        candidate["open_folds"] = opened
        intruders: dict[str, int] = {}
        for pauses in folds:
            pred = predicted_codes(pauses)
            for true, guess in zip(pauses["codes"], pred, strict=True):
                if true not in codes and guess in codes:
                    pair = f"CM{true + 1}->CM{guess + 1}"
                    intruders[pair] = intruders.get(pair, 0) + 1
        candidate["false_accepts"] = dict(sorted(intruders.items(), key=lambda kv: -kv[1])[:20])
        report["candidate"] = candidate
        if args.greedy and vocab["name"] == "full":
            report["greedy"] = greedy_vocab(
                folds, codes, args.greedy, [cm - 1 for cm in args.exclude]
            )

    if args.compare:
        other_runs = find_runs(
            args.runs_dir, args.compare, args.vocab, args.seed, args.n_train_actors
        )
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

    stem = f"{args.vocab}_{args.stage}"
    if args.n_train_actors is not None:
        stem += f"_n{args.n_train_actors}"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    pooled = report["pooled_pause_acc"]
    title = f"{args.stage} ({args.vocab}), {len(runs)} folds: pausa {pooled:.3f}"
    plot_confusion(matrix, names, args.out / f"{stem}_confusao.png", title)
    print(json.dumps({k: report[k] for k in ("mean", "pooled_pause_acc")}, ensure_ascii=False))
    if "candidate" in report:
        print(json.dumps(report["candidate"], ensure_ascii=False)[:3000])
    if "greedy" in report:
        print(json.dumps(report["greedy"], ensure_ascii=False))
    print(args.out / f"{stem}.json")


def confusion_inside(pauses: dict, codes: list[int]) -> np.ndarray:
    """Confusão fechada no vocabulário: só pausas dele, argmax nas colunas dele."""
    inside = np.isin(pauses["codes"], codes)
    restricted = pauses["logits"][inside][:, codes].argmax(axis=1)
    index = {c: i for i, c in enumerate(codes)}
    truth = np.array([index[c] for c in pauses["codes"][inside]])
    matrix = np.zeros((len(codes), len(codes)), dtype=np.int64)
    np.add.at(matrix, (truth, restricted), 1)
    return matrix


if __name__ == "__main__":
    main()
