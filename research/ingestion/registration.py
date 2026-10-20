"""Alinhamento da profundidade ao RGB, por sessão.

O .oni da base não aplica o registro na reprodução: o arquivo diz que o modo
DEPTH_TO_COLOR não é suportado, e o pedido é ignorado sem erro. A profundidade
chega no espaço do sensor, deslocada da cor de forma diferente em cada sessão.

Na distância da mão (0,8 a 1,1 m) o deslocamento é quase constante. A calibração
procura, por sessão, o (dx, dy) que melhor encaixa a superfície mais próxima da
profundidade na silhueta da mão desenhada pelos 21 pontos do MediaPipe, somando
o encaixe de todos os frames das pausas de alguns clipes. `crop.shift_depth`
aplica o resultado: o conteúdo da profundidade anda (dx, dy) pixels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.ingestion.crop import context_box, hand_box
from research.ingestion.frame_clock import clip_pauses, scan_clip
from research.ingestion.oni_io import require_playable_oni

# Ligações dos 21 pontos do MediaPipe Hands; a palma é o polígono do pulso às bases.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
)  # fmt: skip
PALM = (0, 1, 5, 9, 13, 17)


def hand_silhouette(landmarks_xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Máscara da mão desenhada pelos pontos: palma cheia e dedos com a largura de um dedo."""
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    points = np.round(np.asarray(landmarks_xy, dtype=float)).astype(np.int32)
    palm = float(np.linalg.norm(points[0] - points[9]))
    thickness = max(3, int(round(0.3 * palm)))
    cv2.fillPoly(mask, [points[list(PALM)]], 1)
    for a, b in HAND_CONNECTIONS:
        cv2.line(mask, tuple(map(int, points[a])), tuple(map(int, points[b])), 1, thickness)
    return mask.astype(bool)


def alignment_map(
    depth: np.ndarray,
    landmarks_xy: np.ndarray,
    box: tuple[int, int, int, int],
    max_shift: int = 48,
    step: int = 2,
    band_mm: float = 100,
    max_mm: float = 2500,
) -> np.ndarray | None:
    """IoU entre a silhueta da mão e a superfície próxima, para cada (dy, dx) da grade.

    A grade vai de -max_shift a +max_shift, de `step` em `step`; o índice
    [i, j] é dy = -max_shift + i·step, dx = -max_shift + j·step. A janela é a
    caixa da mão com margem 2×. None se a janela não tiver profundidade útil.
    """
    height, width = depth.shape[:2]
    x0, y0, x1, y1 = context_box(box, 2.0)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if x0 >= x1 or y0 >= y1:
        return None
    win_h, win_w = y1 - y0, x1 - x0
    silhouette = hand_silhouette(np.asarray(landmarks_xy) - [x0, y0], (win_h, win_w))
    pad = max_shift
    big = np.zeros((win_h + 2 * pad, win_w + 2 * pad), dtype=depth.dtype)
    bx0, by0 = x0 - pad, y0 - pad
    sx0, sy0 = max(bx0, 0), max(by0, 0)
    sx1, sy1 = min(x1 + pad, width), min(y1 + pad, height)
    big[sy0 - by0 : sy1 - by0, sx0 - bx0 : sx1 - bx0] = depth[sy0:sy1, sx0:sx1]
    valid = (big > 0) & (big < max_mm)
    inner = valid[pad : pad + win_h, pad : pad + win_w]
    if inner.sum() < 100:
        return None
    z_near = float(np.percentile(big[pad : pad + win_h, pad : pad + win_w][inner], 2))
    near = valid & (big <= z_near + band_mm)
    shifts = range(-max_shift, max_shift + 1, step)
    scores = np.zeros((len(shifts), len(shifts)))
    sil_area = int(silhouette.sum())
    for i, dy in enumerate(shifts):
        for j, dx in enumerate(shifts):
            moved = near[pad - dy : pad - dy + win_h, pad - dx : pad - dx + win_w]
            inter = int(np.count_nonzero(moved & silhouette))
            union = sil_area + int(np.count_nonzero(moved)) - inter
            scores[i, j] = inter / union if union else 0.0
    return scores


def calibrate_clip(task: tuple[str, dict]) -> dict:
    """Soma dos mapas de encaixe nos frames das pausas de um clipe."""
    oni_path, params = task
    oni_path = Path(oni_path)
    try:
        return _calibrate_clip(oni_path, params)
    except Exception as exc:
        return {"file": oni_path.stem, "scores": None, "n_frames": 0, "per_frame": [],
                "error": f"{type(exc).__name__}: {exc}"}  # fmt: skip


def _calibrate_clip(oni_path: Path, params: dict) -> dict:
    require_playable_oni(oni_path)
    pairs, hands, landmarks = scan_clip(
        oni_path,
        params["mp_model"],
        params["mp_min_confidence"],
        close=params["openni_teardown"] == "close",
    )
    pauses = clip_pauses(
        landmarks,
        name=oni_path.name,
        min_track_coverage=params["min_track_coverage"],
        window=params["window"],
        motion_fraction=params["motion_fraction"],
        hold_fraction=params["hold_fraction"],
        frames_per_pause=params["frames_per_pause"],
        stride=params["frame_stride"],
        max_shape_change=params["max_shape_change"],
    )
    total, per_frame, used = None, [], 0
    for pause in pauses:
        for index in pause["frames"]:
            depth, color = pairs[index]
            box = hand_box(
                hands[index]["xy"], color.shape[0], params["hand_scale"], params["min_box_frac"]
            )
            scores = alignment_map(
                depth,
                hands[index]["xy"],
                box,
                params["max_shift"],
                params["shift_step"],
                params["band_mm"],
            )
            if scores is None:
                continue
            total = scores if total is None else total + scores
            per_frame.append(np.unravel_index(int(np.argmax(scores)), scores.shape))
            used += 1
    return {"file": oni_path.stem, "scores": total, "n_frames": used, "per_frame": per_frame}


def session_shift(results: list[dict], max_shift: int, step: int) -> dict:
    """(dx, dy) da sessão: o máximo da soma dos mapas de todos os frames."""
    maps = [result["scores"] for result in results if result.get("scores") is not None]
    if not maps:
        raise ValueError("nenhum frame com profundidade útil para calibrar")
    total = np.sum(maps, axis=0)
    n_frames = sum(result["n_frames"] for result in results)
    mean = total / n_frames
    i, j = np.unravel_index(int(np.argmax(mean)), mean.shape)
    origin = max_shift // step
    per_frame = np.array([pair for result in results for pair in result["per_frame"]])
    spread = np.median(np.abs(per_frame - [i, j]), axis=0) * step if len(per_frame) else [0, 0]
    return {
        "dx": int(-max_shift + j * step),
        "dy": int(-max_shift + i * step),
        "iou_before": round(float(mean[origin, origin]), 3),
        "iou_after": round(float(mean[i, j]), 3),
        "n_clips": len(maps),
        "n_frames": int(n_frames),
        "frame_spread_px": [float(spread[1]), float(spread[0])],
    }


def load_registration(path: Path) -> dict:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Calibração ausente em {path}. Rode `make calibrate`.")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def calibrate(config_path: Path, clips_per_session: int | None = None, workers: int | None = None):
    """Calibra todas as sessões e grava `paths.registration_path`."""
    from research.config import load_config
    from research.ingestion.build_dataset import _params, _repo_path, list_clips, run_isolated

    config_path = Path(config_path)
    config = load_config(config_path)
    params = _params(config)
    reg = config["registration"]
    params.update(
        max_shift=int(reg["max_shift"]),
        shift_step=int(reg["shift_step"]),
        band_mm=float(reg["band_mm"]),
    )
    per_session = clips_per_session or int(reg["clips_per_session"])
    raw_root = _repo_path(config["paths"]["raw_root"])
    sessions = sorted(path.name for path in raw_root.iterdir() if path.is_dir())
    tasks, owners = [], []
    for session in sessions:
        clips = [path for path in list_clips(raw_root, [session]) if path.stat().st_size > 0]
        picks = np.linspace(0, len(clips) - 1, min(per_session, len(clips))).round().astype(int)
        for index in sorted(set(picks.tolist())):
            tasks.append((str(clips[index]), params))
            owners.append(session)
    ingest = config["ingest"]
    results: dict[str, list[dict]] = {session: [] for session in sessions}
    stream = run_isolated(
        tasks,
        workers or int(ingest["num_workers"]),
        attempts=int(ingest["attempts"]),
        timeout_s=float(ingest["timeout_s"]),
        target="research.ingestion.registration:calibrate_clip",
    )
    for session, task, (outcome, _, failure) in zip(owners, tasks, stream):
        name = Path(task[0]).stem
        if outcome is None or outcome["scores"] is None:
            reason = failure if outcome is None else outcome["error"]
            print(f"{session} {name}: sem calibração ({reason})", flush=True)
            continue
        results[session].append(outcome)
        print(f"{session} {name}: {outcome['n_frames']} frames", flush=True)
    registration = {
        "method": "máximo do IoU médio entre a silhueta dos 21 pontos e a superfície próxima",
        "max_shift": params["max_shift"],
        "shift_step": params["shift_step"],
        "band_mm": params["band_mm"],
        "sessions": {
            session: session_shift(found, params["max_shift"], params["shift_step"])
            for session, found in results.items()
        },
    }
    out = _repo_path(config["paths"]["registration_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(registration, ensure_ascii=False, indent=1), encoding="utf-8")
    for session, shift in registration["sessions"].items():
        print(session, shift)
    return registration


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibra o deslocamento profundidade → cor.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--clips-per-session", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    calibrate(args.config, args.clips_per_session, args.workers)


if __name__ == "__main__":
    main()
