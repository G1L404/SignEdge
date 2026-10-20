"""Kit do laboratório do minicurso: um zip que roda no Colab sem o LMDB.

Leva o código (`research/`, `configs/`), o student_kd de um fold do LOSO e as
vistas 224 já recortadas (EvalView) do ator que esse modelo nunca viu, mais
frames de calibração dos atores de treino e as predições fora do fold dos 5
student_kd (para ajustar T). O C6 não entra: ele é o teste final, medido uma vez.

As vistas vão em uint8 (o recorte float arredondado a 1/255): é o que a câmera
entrega, e o erro fica abaixo de meio nível de cinza.

Os recortes são derivados da LIBRAS-HC-RGBDS: confira a licença da base antes de
distribuir o kit.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from research.config import load_config
from research.optimization.quantizer import app_views, load_split
from research.train import REPO_ROOT, _repo_path


def _views(split, hand_side: float, config: dict, keep: np.ndarray):
    rgb, points = [], []
    for r, p in app_views(split, hand_side, config):
        rgb.append(np.clip(np.rint(r * 255), 0, 255).astype(np.uint8))
        points.append(p.astype(np.float32))
    return np.concatenate(rgb)[keep], np.concatenate(points)[keep]


def build(config_path: Path, run_dir: Path, out: Path, frames_per_pause: int, calib: int) -> Path:
    from research.training.aggregate import find_runs
    from research.training.metrics import pause_logits

    settings = load_config(config_path)
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if record["fold"] is None:
        raise ValueError("o kit usa um fold do LOSO: o modelo --all-actors já viu todos os atores")

    split, hand_side = load_split(record, "val")
    pauses = split.pause_ids.numpy()
    rank = np.zeros(len(pauses), dtype=int)
    seen: dict[int, int] = {}
    for i, p in enumerate(pauses):
        rank[i] = seen.get(int(p), 0)
        seen[int(p)] = rank[i] + 1
    keep = rank < frames_per_pause
    rgb, points = _views(split, hand_side, record["config"], keep)

    cal_split, cal_side = load_split(record, "train", keep_pauses=calib, seed=0)
    cal_rgb, cal_points = _views(
        cal_split, cal_side, record["config"], np.ones(len(cal_split), bool)
    )

    oof_logits, oof_labels, oof_fold = [], [], []
    for fold, d in find_runs(_repo_path(settings["paths"]["runs_dir"]), record["stage"]).items():
        z = np.load(d / "pred_val.npz")
        logits, labels = pause_logits(z["logits"].astype(np.float32), z["labels"], z["pause_ids"])
        oof_logits.append(logits)
        oof_labels.append(labels)
        oof_fold.append(np.full(len(labels), fold))

    out.parent.mkdir(parents=True, exist_ok=True)
    data = out.with_suffix(".npz")
    np.savez_compressed(
        data,
        rgb=rgb,
        landmarks=points,
        labels=split.labels.numpy()[keep],
        codes=split.codes.numpy()[keep],
        pause_ids=pauses[keep],
        calib_rgb=cal_rgb,
        calib_landmarks=cal_points,
        oof_logits=np.concatenate(oof_logits),
        oof_labels=np.concatenate(oof_labels),
        oof_fold=np.concatenate(oof_fold),
        demo_codes=np.array(settings["vocabularies"]["demo"]["codes"]),
    )
    info = {
        "run": run_dir.name,
        "stage": record["stage"],
        "fold": record["fold"],
        "val_actor": record["val_actor"],
        "frames": int(keep.sum()),
        "pauses": len(seen),
        "frames_per_pause": frames_per_pause,
        "calib_frames": len(cal_rgb),
        "oof_pauses": int(sum(len(x) for x in oof_labels)),
    }
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as kit:
        for path in sorted((REPO_ROOT / "research").rglob("*.py")):
            kit.write(path, path.relative_to(REPO_ROOT))
        for path in sorted((REPO_ROOT / "configs").glob("*.yaml")):
            kit.write(path, path.relative_to(REPO_ROOT))
        kit.write(REPO_ROOT / "pyproject.toml", "pyproject.toml")
        kit.write(run_dir / "last.pt", "kit/student/last.pt")
        kit.write(run_dir / "run.json", "kit/student/run.json")
        kit.write(data, "kit/data.npz")
        kit.writestr("kit/info.json", json.dumps(info, ensure_ascii=False, indent=1))
    data.unlink()
    print(json.dumps({**info, "zip_mb": round(out.stat().st_size / 2**20, 1)}))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Monta o kit do laboratório do minicurso.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="student_kd de um fold do LOSO")
    parser.add_argument("--out", type=Path, default=Path("research/experiments/kit/minicurso.zip"))
    parser.add_argument("--frames-per-pause", type=int, default=2)
    parser.add_argument("--calib", type=int, default=160, help="frames de calibração Int8")
    args = parser.parse_args(argv)
    build(
        args.config, _repo_path(args.run), _repo_path(args.out), args.frames_per_pause, args.calib
    )


if __name__ == "__main__":
    main()
