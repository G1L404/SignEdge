"""Validação dos TFLite contra o PyTorch FP32, no conjunto medido do run.

Mesma régua do treino: vista de avaliação, unidade = pausa, também espelhado.
Além das 61 classes, a concordância frame a frame do top-1 com o FP32 (o quanto a
quantização mudou as respostas).

`make export` só copia o arquivo para mobile/assets depois deste passo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.config import load_config
from research.optimization.quantizer import (
    SIGNATURE,
    app_views,
    default_out_dir,
    load_split,
    load_student,
    scale_head,
)
from research.training.metrics import summarize


def tflite_logits(model_path: Path, views: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Logits do TFLite, um frame por chamada como no app (CPU, XNNPACK)."""
    from ai_edge_litert.interpreter import Interpreter

    runner = Interpreter(str(model_path), num_threads=8).get_signature_runner(SIGNATURE)
    out = []
    for rgb, landmarks in views:
        for r, p in zip(rgb, landmarks, strict=True):
            result = runner(rgb=r[None], landmarks=p[None])
            out.append(next(iter(result.values()))[0])
    return np.stack(out).astype(np.float32)


@torch.no_grad()
def torch_logits(model, views: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Referência FP32 do PyTorch na CPU, a mesma entrada dos TFLite."""
    from research.optimization.quantizer import AppStudent

    app = AppStudent(model).eval()
    out = [app(torch.from_numpy(r), torch.from_numpy(p)).numpy() for r, p in views]
    return np.concatenate(out).astype(np.float32)


def report(logits: np.ndarray, split, reference=None) -> dict:
    labels, pauses = split.labels.numpy(), split.pause_ids.numpy()
    metrics = summarize(logits, labels, pauses)
    if reference is not None:
        metrics["top1_agree_fp32"] = round(
            float((logits.argmax(1) == reference.argmax(1)).mean()), 4
        )
        metrics["max_abs_logit_diff"] = round(float(np.abs(logits - reference).max()), 4)
    return metrics


def main(argv: list[str] | None = None) -> None:
    from research.train import _repo_path

    parser = argparse.ArgumentParser(description="Mede os TFLite contra o PyTorch FP32.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="diretório do run do aluno")
    parser.add_argument("--models", type=Path, nargs="*", help="padrão: todo .tflite do export")
    parser.add_argument("--split", default="val", choices=("val", "ood"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    run_dir = _repo_path(args.run)
    out_dir = default_out_dir(config, run_dir)
    models = (
        [_repo_path(m) for m in args.models] if args.models else sorted(out_dir.glob("*.tflite"))
    )

    model, record = load_student(run_dir)
    # A referência FP32 recebe o mesmo 1/T que o quantizador dobrou na cabeça.
    quant = out_dir / "quant.json"
    temperature = json.loads(quant.read_text())["temperature"] if quant.is_file() else 1.0
    scale_head(model, temperature)
    split, hand_side = load_split(record, args.split)
    results = {
        "run": str(run_dir),
        "split": args.split,
        "temperature": temperature,
        "frames": len(split),
        "pauses": split.n_pauses,
    }
    for flip in (False, True):
        views = list(app_views(split, hand_side, record["config"], flip))
        reference = torch_logits(model, views)
        suffix = "_espelhado" if flip else ""
        results[f"pytorch_fp32{suffix}"] = report(reference, split)
        for path in models:
            logits = tflite_logits(path, views)
            results[f"{path.stem}{suffix}"] = report(logits, split, reference)
    results["bytes"] = {p.name: p.stat().st_size for p in models}
    (out_dir / f"eval_{args.split}.json").write_text(json.dumps(results, indent=1), "utf-8")

    print(f"{'modelo':<28}{'MB':>7}{'pausa 61':>10}{'quadro':>9}{'top1=fp32':>11}")
    for name, m in results.items():
        if not isinstance(m, dict) or "pause_acc" not in m:
            continue
        size = results["bytes"].get(f"{name.removesuffix('_espelhado')}.tflite")
        mb = f"{size / 2**20:.2f}" if size else "-"
        agree = f"{m['top1_agree_fp32']:.1%}" if "top1_agree_fp32" in m else "-"
        print(f"{name:<28}{mb:>7}{m['pause_acc']:>10.1%}{m['frame_acc']:>9.1%}{agree:>11}")


if __name__ == "__main__":
    main()
