"""Quanto o contexto fora da mão revela da classe, sem olhar a mão.

Dentro de cada sessão, o brilho e a cor do fundo acompanham a ordem de gravação
das classes. A sonda mede se isso é um atalho aprendível: uma regressão
logística que só vê o que fica fora da mão (a superfície próxima da
profundidade, dilatada), em três protocolos:

- LOSO: treina em 4 atores, testa no 5º (atalho que transfere entre pessoas);
- entre tomadas: treina na tomada 1 de todos, testa na 2 (e o inverso);
- treino: acerto no próprio conjunto de treino (o fundo como chave para decorar).

A mesma sonda com os 21 pontos normalizados é o controle positivo: mostra que
ela aprende quando o sinal existe. Uma pausa = um exemplo (o primeiro frame).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.config import load_config
from research.ingestion.lmdb_engine import LMDBStore, unpack_sample

REPO_ROOT = Path(__file__).resolve().parents[2]
GRID = 4


def _repo_path(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def context_features(rgb: np.ndarray, depth: np.ndarray, band_mm: float, grow_px: int):
    """Média de cor por célula de uma grade 4×4 e desvio global, só fora da mão.

    Devolve (features, fração do recorte que sobrou como contexto).
    """
    import cv2

    valid = (depth > 0) & (depth < 2500)
    if valid.any():
        z_near = np.percentile(depth[valid], 2)
        near = (valid & (depth <= z_near + band_mm)).astype(np.uint8)
    else:
        near = np.zeros(depth.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1,) * 2)
    context = cv2.dilate(near, kernel) == 0
    image = rgb.astype(np.float32) / 255.0
    size = depth.shape[0]
    cell = size // GRID
    overall = image[:, context].mean(axis=1) if context.any() else image.mean(axis=(1, 2))
    feats = []
    for gy in range(GRID):
        for gx in range(GRID):
            window = context[gy * cell : (gy + 1) * cell, gx * cell : (gx + 1) * cell]
            pixels = image[:, gy * cell : (gy + 1) * cell, gx * cell : (gx + 1) * cell]
            feats.append(pixels[:, window].mean(axis=1) if window.sum() > 20 else overall)
    spread = image[:, context].std(axis=1) if context.any() else np.zeros(3)
    return np.concatenate([*feats, spread]), float(context.mean())


def shape_features(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    return (centered / (np.sqrt((centered**2).sum(axis=1).mean()) + 1e-6)).ravel()


def fit_logistic(x: np.ndarray, y: np.ndarray, n_classes: int, l2: float = 1e-3, steps: int = 300):
    """Regressão logística multinomial (LBFGS, padronização pelo treino)."""
    import torch

    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-6
    features = torch.tensor((x - mean) / std, dtype=torch.float64)
    labels = torch.tensor(y)
    weight = torch.zeros(x.shape[1], n_classes, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(n_classes, dtype=torch.float64, requires_grad=True)
    optim = torch.optim.LBFGS([weight, bias], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        optim.zero_grad()
        loss = torch.nn.functional.cross_entropy(features @ weight + bias, labels)
        loss = loss + l2 * (weight**2).sum()
        loss.backward()
        return loss

    optim.step(closure)
    w, b = weight.detach().numpy(), bias.detach().numpy()
    return lambda z: (((z - mean) / std) @ w + b).argmax(axis=1)


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def _score(predict, x, y) -> dict:
    hits = int((predict(x) == y).sum())
    low, high = wilson(hits, len(y))
    return {"acc": round(hits / len(y), 4), "ci95": [round(low, 4), round(high, 4)], "n": len(y)}


def probe(train_config: Path, out: Path | None) -> dict:
    config = load_config(train_config)
    data_config = load_config(_repo_path(config["paths"]["data_config"]))
    band = float(config["augment"]["background"]["band_mm"])
    actors = list(config["split"]["actors"])
    rows = defaultdict(list)
    with LMDBStore(_repo_path(data_config["paths"]["lmdb_path"]), 1 << 30, readonly=True) as store:
        meta = store.read_meta()
        size = int(meta["layout"]["size"])
        first: dict[tuple[str, int], str] = {}
        for key in sorted(meta["samples"]):
            sample = meta["samples"][key]
            first.setdefault((sample["clip"], sample["pause"]), key)
        for key in first.values():
            sample = meta["samples"][key]
            if sample["actor"] not in actors:
                continue
            rgb, depth, points = unpack_sample(store.read_sample(key.encode("utf-8")), size)
            context, kept = context_features(rgb, depth, band, grow_px=12)
            rows["context"].append(context)
            rows["shape"].append(shape_features(points))
            rows["kept"].append(kept)
            rows["cls"].append(sample["cls"])
            rows["actor"].append(sample["actor"])
            rows["take"].append(sample["take"])
    x = {name: np.stack(rows[name]) for name in ("context", "shape")}
    y = np.array(rows["cls"])
    actor, take = np.array(rows["actor"]), np.array(rows["take"])
    n_classes = int(y.max()) + 1
    report = {
        "n_pausas": len(y),
        "acaso": round(1 / n_classes, 4),
        "fracao_contexto": round(float(np.mean(rows["kept"])), 3),
    }
    for name, features in x.items():
        loso, train_fit = [], []
        for held in actors:
            train, test = actor != held, actor == held
            predict = fit_logistic(features[train], y[train], n_classes)
            loso.append({"ator": held, **_score(predict, features[test], y[test])})
            train_fit.append(_score(predict, features[train], y[train])["acc"])
        takes = []
        for source, target in ((1, 2), (2, 1)):
            predict = fit_logistic(features[take == source], y[take == source], n_classes)
            mask = take == target
            takes.append(
                {"treino": source, "teste": target, **_score(predict, features[mask], y[mask])}
            )
        report[name] = {
            "loso": loso,
            "loso_media": round(float(np.mean([r["acc"] for r in loso])), 4),
            "entre_tomadas": takes,
            "acerto_no_treino": round(float(np.mean(train_fit)), 4),
        }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sonda do atalho de fundo.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(probe(args.config, args.out), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
