"""Quais frames do clipe viram amostra: os das pausas da configuração.

O .oni é lido uma vez. O MediaPipe procura a mão em cada frame; entre duas mãos,
vale a mais perto da câmera. A média dos 21 pontos forma a trilha, e as pausas
saem dela. O .skl não é lido.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from research.ingestion.crop import shift_depth
from research.ingestion.errors import SampleRejected
from research.ingestion.hand_detector import DEFAULT_MODEL, detect_hands, pick_hand
from research.ingestion.hand_track import find_pauses, track_coverage
from research.ingestion.oni_io import read_all_frames


def scan_clip(
    oni_path,
    mp_model=None,
    mp_min_confidence: float = 0.3,
    close: bool = True,
    depth_shift: tuple[int, int] = (0, 0),
):
    """Pares (profundidade, cor), a mão do sinal em cada frame (ou None) e os pontos.

    Os pontos são (n, 21, 2) em pixels da foto, NaN onde a mão não foi vista.
    A profundidade anda `depth_shift` (dx, dy) antes de tudo, inclusive do
    desempate entre duas mãos: é o alinhamento da sessão (`registration`).
    """
    model = DEFAULT_MODEL if mp_model is None else Path(mp_model)
    pairs = [
        (shift_depth(depth, *depth_shift), color)
        for depth, color in read_all_frames(oni_path, close=close)
    ]
    hands: list[dict | None] = []
    landmarks = np.full((len(pairs), 21, 2), np.nan)
    for index, (depth, color) in enumerate(pairs):
        found = detect_hands(color, depth, model, mp_min_confidence)
        chosen = pick_hand(found, None)
        hand = None if chosen is None else {**found[chosen], "n_hands": len(found)}
        hands.append(hand)
        if hand is not None:
            landmarks[index] = hand["xy"]
    return pairs, hands, landmarks


def clip_pauses(
    landmarks: np.ndarray,
    name: str = "",
    min_track_coverage: float = 0.25,
    window: int = 5,
    motion_fraction: float = 0.4,
    hold_fraction: float = 0.15,
    frames_per_pause: int = 8,
    stride: int = 2,
    max_shape_change: float = 0.25,
) -> list[dict]:
    """As pausas do clipe. Recusa a trilha curta demais ou sem pausa com a mão levantada.

    A trilha é a média dos 21 pontos de cada frame.
    """
    landmarks = np.asarray(landmarks, dtype=float)
    track = np.full((len(landmarks), 2), np.nan)
    seen = ~np.isnan(landmarks[:, 0, 0])
    track[seen] = landmarks[seen].mean(axis=1)
    coverage = track_coverage(track)
    if coverage < min_track_coverage:
        raise SampleRejected(
            f"mão vista em {coverage:.0%} dos frames de {name}, abaixo de {min_track_coverage:.0%}"
        )
    pauses = find_pauses(
        track,
        window,
        motion_fraction,
        hold_fraction,
        frames_per_pause,
        stride,
        landmarks=landmarks,
        max_shape_change=max_shape_change,
    )
    if not pauses:
        raise SampleRejected(f"nenhuma pausa com a mão levantada em {name}")
    return pauses
