"""Mão no RGB pelo MediaPipe Hands. A profundidade só desempata entre duas mãos."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "data" / "models" / "hand_landmarker.task"

_LANDMARKER = None
_LANDMARKER_KEY = None


def _mediapipe():
    # O MediaPipe importa o sounddevice, que abre o PortAudio no import. Com vários
    # processos ao mesmo tempo, o PipeWire trava nessa abertura. Aqui não há áudio.
    sys.modules.setdefault("sounddevice", None)
    import mediapipe

    return mediapipe


def _landmarker(model_path: Path, min_confidence: float):
    global _LANDMARKER, _LANDMARKER_KEY
    key = (str(model_path), float(min_confidence))
    if _LANDMARKER is None or _LANDMARKER_KEY != key:
        _mediapipe()
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Modelo do MediaPipe ausente em {model_path}")
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
        )
        _LANDMARKER = HandLandmarker.create_from_options(options)
        _LANDMARKER_KEY = key
    return _LANDMARKER


def detect_hands(
    color_img: np.ndarray,
    depth: np.ndarray | None = None,
    model_path: Path = DEFAULT_MODEL,
    min_confidence: float = 0.3,
) -> list[dict]:
    """Mãos no RGB: 21 pontos em pixels, confiança, lado e profundidade mediana em mm."""
    mp = _mediapipe()
    height, width = color_img.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(color_img))
    result = _landmarker(model_path, min_confidence).detect(image)
    hands = []
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        xy = np.array([(point.x * width, point.y * height) for point in landmarks])
        best = max(handedness, key=lambda category: category.score)
        hands.append(
            {
                "xy": xy,
                "score": float(best.score),
                "label": best.category_name,
                "depth_mm": float("nan") if depth is None else median_depth(depth, xy),
            }
        )
    return hands


def hand_square(
    landmarks_xy: np.ndarray,
    scale: float = 1.4,
    min_side: int = 0,
) -> tuple[int, int, int, int]:
    """Quadrado centrado nos 21 pontos.

    O lado é a maior extensão vezes `scale`, nunca abaixo de `min_side`. O piso
    aumenta a caixa em volta do punho fechado em vez de descartar a amostra.
    """
    if scale <= 0 or min_side < 0:
        raise ValueError("scale precisa ser positivo e min_side, não negativo")
    x_min, y_min = landmarks_xy.min(axis=0)
    x_max, y_max = landmarks_xy.max(axis=0)
    side = max(int(round(max(x_max - x_min, y_max - y_min) * scale)), int(min_side), 1)
    x0 = int(round((x_min + x_max) / 2 - side / 2))
    y0 = int(round((y_min + y_max) / 2 - side / 2))
    return (x0, y0, x0 + side, y0 + side)


def median_depth(depth: np.ndarray, landmarks_xy: np.ndarray) -> float:
    """Mediana da profundidade nos pontos da mão, ignorando zeros. NaN se não houver leitura."""
    height, width = depth.shape[:2]
    xs = np.clip(np.round(landmarks_xy[:, 0]).astype(int), 0, width - 1)
    ys = np.clip(np.round(landmarks_xy[:, 1]).astype(int), 0, height - 1)
    values = depth[ys, xs]
    values = values[values > 0]
    return float(np.median(values)) if values.size else float("nan")


def pick_hand(hands: list[dict], seed_xy: np.ndarray | None) -> int | None:
    """Mão do sinal: a mais perto da câmera; sem profundidade, a mais perto da junta."""
    if not hands:
        return None
    depths = [hand["depth_mm"] for hand in hands]
    if any(np.isfinite(value) for value in depths):
        return int(np.nanargmin(depths))
    if seed_xy is not None:
        distances = [np.linalg.norm(hand["xy"].mean(axis=0) - seed_xy) for hand in hands]
        return int(np.argmin(distances))
    return int(np.argmax([hand["score"] for hand in hands]))
