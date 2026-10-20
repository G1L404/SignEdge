"""Conversão do aluno (RGB + 21 pontos) para TFLite float e Int8 estático (PTQ).

Roda no venv de conversão (`.venv-tflite`, Python 3.12 com litert-torch); o venv
do treino (3.14) não tem a pilha TFLite.

Contrato de entrada do app, o mesmo da `EvalView`:
- `rgb`: (1, 224, 224, 3) float32 em [0, 1], NHWC como a câmera entrega; a
  normalização do ImageNet fica dentro do modelo.
- `landmarks`: (1, 21, 2) float32, os pontos do MediaPipe em pixels da vista 224.
Saída: (1, C) logits.

A calibração usa só os atores de treino do run, nunca o ator medido: um frame da
vista de avaliação por pausa sorteada. O ramo dos pontos pode ficar em float
(`--landmarks float`): ele é pequeno, e a normalização dele (centro e raio RMS em
pixels) é a parte do grafo que mais sofre com uma grade de 8 bits.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from research.config import load_config

SIGNATURE = "serving_default"
LANDMARK_SCOPE = ".*LandmarkEncoder.*"


class AppStudent(nn.Module):
    """O aluno com a entrada do app: RGB NHWC em [0, 1] e os pontos em pixels."""

    def __init__(self, student: nn.Module) -> None:
        super().__init__()
        self.student = student

    def forward(self, rgb: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        return self.student(rgb.permute(0, 3, 1, 2), landmarks)


def scale_head(model: nn.Module, temperature: float) -> nn.Module:
    """Dobra 1/T nos pesos e no bias da cabeça: logits / T sem op nenhuma a mais no grafo."""
    if temperature <= 0:
        raise ValueError(f"temperatura deve ser positiva: {temperature}")
    with torch.no_grad():
        model.head.weight /= temperature
        model.head.bias /= temperature
    return model


def load_student(run_dir: Path) -> tuple[nn.Module, dict]:
    """O `last.pt` de um run de aluno, em float32 e modo de avaliação."""
    from research.training.loop import build_model

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if record["stage"] not in ("student_ce", "student_kd"):
        raise ValueError(f"{run_dir} não é um aluno ({record['stage']})")
    model = build_model(record["stage"], record["config"], pretrained=False)
    model.load_state_dict(torch.load(run_dir / "last.pt", map_location="cpu"))
    return model.eval(), record


def load_split(record: dict, which: str, keep_pauses: int | None = None, seed: int = 0):
    """Conjunto `which` (train, val ou ood) do run e o lado da mão em pixels do crop.

    Com `keep_pauses`, fica só um frame de cada uma de `keep_pauses` pausas sorteadas,
    o bastante para calibrar sem carregar o fold inteiro.
    """
    from research.data.splits import all_actors_split, loso_folds
    from research.data.vocab import resolve_vocab
    from research.ingestion.lmdb_engine import LMDBStore
    from research.train import _repo_path
    from research.training.data import InMemorySplit

    config = record["config"]
    vocab = resolve_vocab(config, record["vocab"]["name"])
    data_config = load_config(_repo_path(config["paths"]["data_config"]))
    lmdb_path = _repo_path(data_config["paths"]["lmdb_path"])
    with LMDBStore(lmdb_path, int(data_config["lmdb"]["map_size"]), readonly=True) as store:
        meta = store.read_meta()
        actors, ood = config["split"]["actors"], config["split"]["ood"]
        if record["fold"] is None:
            chosen = all_actors_split(meta["samples"], actors, ood)
        else:
            chosen = loso_folds(meta["samples"], actors, ood)[record["fold"]]
        keys = chosen[which]
        if keep_pauses is not None:
            samples = meta["samples"]
            by_pause: dict[tuple, list[str]] = {}
            for key in keys:
                by_pause.setdefault((samples[key]["clip"], samples[key]["pause"]), []).append(key)
            rng = np.random.default_rng(seed)
            pauses = sorted(by_pause)
            picked = rng.choice(len(pauses), size=min(keep_pauses, len(pauses)), replace=False)
            keys = sorted(rng.choice(by_pause[pauses[i]]) for i in picked)
        split = InMemorySplit(store, keys, meta, vocab["label_of"])
    hand_side = int(meta["layout"]["size"]) / float(meta["crop_rule"]["context_scale"])
    return split, hand_side


def app_views(split, hand_side: float, config: dict, flip: bool = False, batch: int = 128):
    """Os frames de `split` na vista do app, em lotes de (rgb NHWC, pontos) numpy float32."""
    from research.data.augment import EvalView

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    view_cfg = config["view"]
    view = EvalView(hand_side, view_cfg["view_context"], view_cfg["input_size"], device)
    for start in range(0, len(split), batch):
        rgb, depth, landmarks, _ = split.batch(
            list(range(start, min(start + batch, len(split)))), device
        )
        out = view(rgb, depth, landmarks, flip)
        yield (
            out["rgb"].permute(0, 2, 3, 1).float().cpu().numpy(),
            out["landmarks"].float().cpu().numpy(),
        )


class GlobalMean(nn.Module):
    """`AdaptiveAvgPool2d(1)` como média: o conversor transforma o pooling adaptativo
    num GATHER_ND, que o XNNPACK não executa. O nó na CPU partia o grafo, e o LiteRT 1.4
    do Android falhava no invoke ("Input tensor N lacks data")."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3), keepdim=True)


def replace_global_pool(model: nn.Module) -> nn.Module:
    """Troca todo `AdaptiveAvgPool2d` de saída 1×1 por [GlobalMean] (mesma conta)."""
    for name, child in model.named_children():
        size = getattr(child, "output_size", None)
        if isinstance(child, nn.AdaptiveAvgPool2d) and size in (1, (1, 1)):
            setattr(model, name, GlobalMean())
        else:
            replace_global_pool(child)
    return model


def export_float(model: nn.Module, output: Path, input_size: int) -> Path:
    """TFLite float32 com a assinatura do app (entradas `rgb` e `landmarks`)."""
    import copy

    import litert_torch

    model = replace_global_pool(copy.deepcopy(model))
    sample = {
        "rgb": torch.rand(1, input_size, input_size, 3),
        "landmarks": torch.rand(1, 21, 2) * input_size,
    }
    edge = litert_torch.convert(AppStudent(model).eval(), sample_kwargs=sample)
    output.parent.mkdir(parents=True, exist_ok=True)
    edge.export(str(output))
    return output


def quantize_int8(
    float_path: Path, calibration: list[dict], output: Path, landmarks_float: bool
) -> Path:
    """PTQ estático: pesos Int8 por canal, ativações Int8 calibradas por min/max.

    Entrada e saída continuam float32: o app não precisa saber a escala de nada.
    """
    from ai_edge_quantizer import quantizer, recipe
    from ai_edge_quantizer.algorithm_manager import AlgorithmName
    from ai_edge_quantizer.qtyping import TFLOperationName

    quant = quantizer.Quantizer(str(float_path), recipe.static_wi8_ai8())
    # Sem isto as entradas viram Int8: os pontos em pixels cairiam numa grade de ~0,9 px.
    for edge in (TFLOperationName.INPUT, TFLOperationName.OUTPUT):
        quant.update_quantization_recipe(
            regex=".*", operation_name=edge, algorithm_key=AlgorithmName.NO_QUANTIZE
        )
    if landmarks_float:
        quant.update_quantization_recipe(
            regex=LANDMARK_SCOPE,
            operation_name=TFLOperationName.ALL_SUPPORTED,
            algorithm_key=AlgorithmName.NO_QUANTIZE,
        )
    qsv = quant.calibrate({SIGNATURE: calibration})
    output.parent.mkdir(parents=True, exist_ok=True)
    quant.quantize(qsv).export_model(str(output), overwrite=True)
    inline_buffers(output)
    return output


def inline_buffers(path: Path) -> int:
    """Põe dentro do flatbuffer os pesos que o ai-edge-quantizer grava depois dele.

    O quantizador guarda buffers grandes fora da tabela (`offset`/`size`). O LiteRT
    1.4 do tflite_flutter, no Android, não resolve esses offsets num modelo lido da
    memória: os pesos ficam vazios ("Input tensor N lacks data"). Os dados entram
    alinhados a 16 bytes, o `force_align` do schema. Devolve quantos buffers moveu.
    """
    import flatbuffers
    from ai_edge_litert import schema_py_generated as schema

    raw = path.read_bytes()
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(raw, 0))
    moved = 0
    for buffer in model.buffers:
        if buffer.offset > 1:
            buffer.data = np.frombuffer(raw[buffer.offset : buffer.offset + buffer.size], np.uint8)
            buffer.offset = buffer.size = 0
            moved += 1
    if not moved:
        return 0

    def pack(self, builder):
        data = None
        if self.data is not None and len(self.data):
            payload = np.asarray(self.data, dtype=np.uint8).tobytes()
            builder.StartVector(1, len(payload), 16)
            builder.head -= len(payload)
            builder.Bytes[builder.head : builder.head + len(payload)] = payload
            data = builder.EndVector()
        schema.BufferStart(builder)
        if data is not None:
            schema.BufferAddData(builder, data)
        return schema.BufferEnd(builder)

    original = schema.BufferT.Pack
    schema.BufferT.Pack = pack
    try:
        builder = flatbuffers.Builder(len(raw) + 1024)
        builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    finally:
        schema.BufferT.Pack = original
    path.write_bytes(bytes(builder.Output()))
    return moved


def calibration_set(record: dict, n_pauses: int, seed: int) -> list[dict]:
    """Um frame por pausa sorteada dos atores de treino, na vista do app."""
    split, hand_side = load_split(record, "train", keep_pauses=n_pauses, seed=seed)
    samples = []
    for rgb, landmarks in app_views(split, hand_side, record["config"]):
        samples += [
            {"rgb": r[None], "landmarks": p[None]} for r, p in zip(rgb, landmarks, strict=True)
        ]
    return samples


def default_out_dir(config: dict, run_dir: Path) -> Path:
    from research.train import _repo_path

    return _repo_path(config["paths"]["exports_dir"]) / run_dir.name


def main(argv: list[str] | None = None) -> None:
    from research.train import _repo_path

    parser = argparse.ArgumentParser(description="Exporta o aluno para TFLite float e Int8.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="diretório do run do aluno")
    parser.add_argument("--out", type=Path, help="padrão: paths.exports_dir/<run>")
    parser.add_argument("--landmarks", choices=("int8", "float"), default="int8")
    parser.add_argument("--calib-pauses", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="T da calibração (research.optimization.calibration)",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    run_dir = _repo_path(args.run)
    out_dir = _repo_path(args.out) if args.out else default_out_dir(config, run_dir)

    model, record = load_student(run_dir)
    scale_head(model, args.temperature)
    input_size = int(record["config"]["view"]["input_size"])
    float_path = export_float(model, out_dir / "model_float.tflite", input_size)
    calibration = calibration_set(record, args.calib_pauses, args.seed)
    suffix = "" if args.landmarks == "int8" else "_lmfloat"
    int8_path = quantize_int8(
        float_path, calibration, out_dir / f"model_int8{suffix}.tflite", args.landmarks == "float"
    )
    info = {
        "run": str(run_dir),
        "stage": record["stage"],
        "fold": record["fold"],
        "vocab": record["vocab"]["name"],
        "calibration": {
            "actors": record.get("train_actors"),
            "frames": len(calibration),
            "seed": args.seed,
        },
        "landmarks": args.landmarks,
        "temperature": args.temperature,
        "inputs": {"rgb": [1, input_size, input_size, 3], "landmarks": [1, 21, 2]},
        "bytes": {p.name: p.stat().st_size for p in (float_path, int8_path)},
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out_dir / f"quant{suffix}.json").write_text(json.dumps(info, indent=1), "utf-8")
    print(json.dumps(info["bytes"]), flush=True)


if __name__ == "__main__":
    main()
