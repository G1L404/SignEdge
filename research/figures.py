"""Figuras do capítulo: o atlas das 61 CMs e a matriz de confusão do student_kd.

    python -m research.figures atlas --out figuras/atlas.jpg
    python -m research.figures confusao --out figuras/confusao.pdf

O atlas é a vista de avaliação (a que o modelo recebe) do quadro central da primeira
pausa de cada CM, num único sinalizante (C4, sem rosto no quadro), em grade 11 × 6.
A matriz soma as pausas de validação dos 5 folds do aluno destilado.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

COLUMNS, TILE, GAP, LABEL = 11, 224, 10, 34
FONT = "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"


def atlas(config_path: Path, out: Path, actor: str) -> Path:
    import torch
    from PIL import Image, ImageDraw, ImageFont

    from research.config import load_config
    from research.data.augment import EvalView
    from research.data.class_atlas import representative_keys
    from research.data.preview import _repo_path, load_batch
    from research.ingestion.lmdb_engine import LMDBStore

    config = load_config(config_path)
    data_config = load_config(_repo_path(config["paths"]["data_config"]))
    with LMDBStore(_repo_path(data_config["paths"]["lmdb_path"]), 1 << 30, readonly=True) as store:
        meta = store.read_meta()
        size = int(meta["layout"]["size"])
        view = EvalView(size / float(meta["crop_rule"]["context_scale"]), 1.5, TILE, "cpu")
        keys = representative_keys(meta["samples"], actor)
        codes = sorted(keys)
        rgb, depth, points = load_batch(store, [keys[c] for c in codes], size)
        with torch.no_grad():
            views = view(rgb, depth, points)["rgb"]
    tiles = {
        code: (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        for code, image in zip(codes, views, strict=True)
    }

    rows = -(-61 // COLUMNS)
    width = COLUMNS * TILE + (COLUMNS - 1) * GAP
    height = rows * (TILE + LABEL) + (rows - 1) * GAP
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 26)
    for code in range(61):
        row, col = divmod(code, COLUMNS)
        x, y = col * (TILE + GAP), row * (TILE + LABEL + GAP)
        draw.text((x + 2, y + 2), f"CM {code + 1}", fill="black", font=font)
        canvas.paste(Image.fromarray(tiles[code]), (x, y + LABEL))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=92, dpi=(300, 300), optimize=True)
    return out


def confusion_figure(runs_dir: Path, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from research.training.aggregate import confusion, find_runs, load_pauses

    runs = find_runs(runs_dir, "student_kd")
    matrix = confusion([load_pauses(r) for r in runs.values()], 61)
    off = matrix.copy()
    np.fill_diagonal(off, 0)
    rates = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)

    plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.3, 6.0))
    image = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ticks = np.arange(61)
    ax.set_xticks(ticks, [str(t + 1) for t in ticks], fontsize=4.2, rotation=90)
    ax.set_yticks(ticks, [str(t + 1) for t in ticks], fontsize=4.2)
    ax.tick_params(length=1.5, width=0.4, pad=1)
    ax.set_xticks(np.arange(-0.5, 61, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 61, 1), minor=True)
    ax.grid(which="minor", color="0.92", lw=0.25)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.set_xlabel("CM prevista")
    ax.set_ylabel("CM verdadeira")
    for i in range(61):
        ax.text(
            i,
            i,
            int(matrix[i, i]),
            ha="center",
            va="center",
            fontsize=3.4,
            color="white" if rates[i, i] > 0.5 else "black",
        )
    for i, j in zip(*np.nonzero(off >= 2), strict=True):
        ax.text(
            j,
            i,
            int(off[i, j]),
            ha="center",
            va="center",
            fontsize=3.4,
            color="white" if rates[i, j] > 0.5 else "#b2182b",
        )
    cax = make_axes_locatable(ax).append_axes("right", size="2.5%", pad=0.08)
    bar = fig.colorbar(image, cax=cax)
    bar.set_label("fração das pausas da CM verdadeira")
    bar.outline.set_linewidth(0.4)
    bar.ax.tick_params(labelsize=6, width=0.4, length=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Figuras do capítulo.")
    sub = parser.add_subparsers(dest="figure", required=True)
    a = sub.add_parser("atlas")
    a.add_argument("--config", type=Path, default=Path("configs/distillation.yaml"))
    a.add_argument("--actor", default="C4")
    a.add_argument("--out", type=Path, default=Path("figuras/atlas.jpg"))
    c = sub.add_parser("confusao")
    c.add_argument("--runs-dir", type=Path, default=Path("research/experiments/runs"))
    c.add_argument("--out", type=Path, default=Path("figuras/confusao.pdf"))
    args = parser.parse_args(argv)
    if args.figure == "atlas":
        print(atlas(args.config, args.out, args.actor))
    else:
        print(confusion_figure(args.runs_dir, args.out))


if __name__ == "__main__":
    main()
