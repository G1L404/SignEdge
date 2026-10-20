"""Atlas das 61 configurações: uma imagem real por classe e ator, rotulada CM1..CM61.

Serve para conferir a numeração contra a figura de referência antes de fechar um
vocabulário: cada célula é a vista de avaliação (a mesma que o modelo recebe) do
frame central da primeira pausa de um clipe, em alguns atores lado a lado. As
classes de `--vocab` ganham borda verde.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from research.config import load_config
from research.data.augment import EvalView
from research.data.preview import _repo_path, load_batch
from research.data.vocab import resolve_vocab
from research.ingestion.lmdb_engine import LMDBStore


def representative_keys(samples: dict[str, dict], actor: str) -> dict[int, str]:
    """{código: chave} com o frame central da primeira pausa do primeiro clipe do ator."""
    frames: dict[int, list[tuple]] = {}
    for key, meta in samples.items():
        if meta["actor"] == actor:
            frames.setdefault(int(meta["cls"]), []).append(
                (meta["clip"], meta["pause"], meta["frame"], key)
            )
    chosen = {}
    for code, rows in frames.items():
        rows.sort()
        first = [row for row in rows if row[:2] == rows[0][:2]]
        chosen[code] = first[len(first) // 2][3]
    return chosen


def atlas(
    train_config: Path,
    out: Path,
    actors: list[str],
    tile: int,
    columns: int,
    vocab: str | None,
    only_vocab: bool = False,
    cms: list[int] | None = None,
) -> Path:
    import cv2
    import torch

    config = load_config(train_config)
    data_config = load_config(_repo_path(config["paths"]["data_config"]))
    marked = set(resolve_vocab(config, vocab)["codes"]) if vocab else set()
    if cms:
        marked = {cm - 1 for cm in cms}
    with LMDBStore(_repo_path(data_config["paths"]["lmdb_path"]), 1 << 30, readonly=True) as store:
        meta = store.read_meta()
        size = int(meta["layout"]["size"])
        view = EvalView(size / float(meta["crop_rule"]["context_scale"]), 1.5, tile, "cpu")
        images = {}
        for actor in actors:
            keys = representative_keys(meta["samples"], actor)
            codes = sorted(keys)
            rgb, depth, points = load_batch(store, [keys[c] for c in codes], size)
            with torch.no_grad():
                out_rgb = view(rgb, depth, points)["rgb"]
            for code, image in zip(codes, out_rgb, strict=True):
                pixels = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                images[(code, actor)] = np.ascontiguousarray(pixels)
    blank = np.zeros((tile, tile, 3), np.uint8)
    cells = []
    for code in sorted(marked) if only_vocab else range(61):
        strip = np.hstack([images.get((code, actor), blank) for actor in actors])
        strip = cv2.copyMakeBorder(strip, 22, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        color = (60, 220, 60) if code in marked else (255, 255, 255)
        cv2.putText(strip, f"CM{code + 1}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if code in marked:
            cv2.rectangle(strip, (0, 0), (strip.shape[1] - 1, strip.shape[0] - 1), color, 3)
        cells.append(strip)
    while len(cells) % columns:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i : i + columns]) for i in range(0, len(cells), columns)]
    grid = np.vstack(rows)
    legend = np.zeros((28, grid.shape[1], 3), np.uint8)
    text = "atores por celula: " + ", ".join(actors)
    cv2.putText(legend, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack([legend, grid]), cv2.COLOR_RGB2BGR))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Atlas das configurações de mão do LMDB.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--actors", nargs="+", default=["C1", "C3", "C5"])
    parser.add_argument("--tile", type=int, default=112)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--vocab", help="destaca as classes desse vocabulário")
    parser.add_argument("--only-vocab", action="store_true", help="mostra só as do vocabulário")
    parser.add_argument("--cms", type=int, nargs="+", help="números CM a destacar (troca --vocab)")
    args = parser.parse_args(argv)
    print(
        atlas(
            args.config,
            args.out,
            args.actors,
            args.tile,
            args.columns,
            args.vocab,
            args.only_vocab,
            args.cms,
        )
    )


if __name__ == "__main__":
    main()
