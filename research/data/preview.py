"""Mosaico de vistas aumentadas, para conferir a augmentation a olho antes de treinar.

Cada linha é uma amostra: a vista de avaliação e `--views` sorteios do treino,
com os 21 pontos desenhados por cima. Um ponto fora do dedo denuncia erro de
geometria; um fundo trocado por cima da mão denuncia erro de máscara.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from research.config import load_config
from research.data.augment import EvalView, TrainAugment, load_backgrounds
from research.data.splits import loso_folds
from research.ingestion.lmdb_engine import LMDBStore, unpack_sample

REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_path(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_batch(store: LMDBStore, keys: list[str], size: int):
    import torch

    rgb, depth, points = [], [], []
    for key in keys:
        r, d, p = unpack_sample(store.read_sample(key.encode("utf-8")), size)
        rgb.append(r.copy())
        depth.append(d.astype(np.float32))
        points.append(p.copy())
    return (
        torch.from_numpy(np.stack(rgb)),
        torch.from_numpy(np.stack(depth)),
        torch.from_numpy(np.stack(points)),
    )


def _tile(rgb, points, label: str):
    import cv2

    image = np.ascontiguousarray((rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    for x, y in points.cpu().numpy():
        cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
    cv2.putText(image, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    return image


def preview(train_config: Path, out: Path, rows: int, views: int, fold: int, seed: int) -> Path:
    import cv2
    import torch

    config = load_config(train_config)
    data_config = load_config(_repo_path(config["paths"]["data_config"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with LMDBStore(_repo_path(data_config["paths"]["lmdb_path"]), 1 << 30, readonly=True) as store:
        meta = store.read_meta()
        size = int(meta["layout"]["size"])
        hand_side = size / float(meta["crop_rule"]["context_scale"])
        split = config["split"]
        keys = loso_folds(meta["samples"], split["actors"], split["ood"])[fold]["train"]
        rng = np.random.default_rng(seed)
        chosen = [keys[i] for i in rng.choice(len(keys), size=rows, replace=False)]
        rgb, depth, points = load_batch(store, chosen, size)
    view_cfg, aug_cfg = config["view"], config["augment"]
    photos = load_backgrounds(
        _repo_path(aug_cfg["background"]["photo_dir"]), view_cfg["input_size"]
    )
    augment = TrainAugment(
        aug_cfg, hand_side, view_cfg["view_context"], view_cfg["input_size"], device, seed, photos
    )
    evaluate = EvalView(hand_side, view_cfg["view_context"], view_cfg["input_size"], device)
    base = evaluate(rgb, depth, points)
    columns = [[_tile(base["rgb"][i], base["landmarks"][i], chosen[i]) for i in range(rows)]]
    for _ in range(views):
        aug = augment(rgb, depth, points)
        columns.append(
            [
                _tile(aug["rgb"][i], aug["landmarks"][i], "espelho" if aug["flip"][i] else "")
                for i in range(rows)
            ]
        )
    grid = np.vstack([np.hstack([col[i] for col in columns]) for i in range(rows)])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mosaico de vistas aumentadas do LMDB.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(preview(args.config, args.out, args.rows, args.views, args.fold, args.seed))


if __name__ == "__main__":
    main()
