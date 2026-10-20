"""Trilha da mão e as pausas da configuração. Matemática pura, sem OpenNI.

Uma trilha é (n, 2) em pixels da foto, a média dos 21 pontos do MediaPipe, NaN
onde a mão não foi vista. Os índices são 0-based; o frame do .oni é o índice + 1.
"""

from __future__ import annotations

import numpy as np


def _hand_speed(hand_xy: np.ndarray, window: int) -> np.ndarray:
    """Velocidade por frame. O índice 0 fica infinito: não há frame anterior."""
    if len(hand_xy) < 2:
        raise ValueError("a trajetória precisa de pelo menos dois frames")
    if window > 1 and window % 2 == 0:
        raise ValueError("a janela de suavização precisa ser ímpar")
    speed = np.full(len(hand_xy), np.inf)
    speed[1:] = np.linalg.norm(np.diff(hand_xy, axis=0), axis=1)
    if window > 1:
        pad = window // 2
        kernel = np.ones(window) / window
        padded = np.pad(speed[1:], (pad, pad), mode="edge")
        speed = speed.copy()
        speed[1:] = np.convolve(padded, kernel, mode="valid")
    return speed


def fill_track(track: np.ndarray) -> np.ndarray | None:
    """Interpola centros que faltaram. Sem dois centros válidos, devolve None."""
    if track.ndim != 2 or track.shape[1] != 2 or len(track) < 2:
        return None
    good = ~np.isnan(track[:, 0])
    if int(good.sum()) < 2:
        return None
    index = np.arange(len(track))
    filled = np.empty_like(track, dtype=float)
    for axis in (0, 1):
        filled[:, axis] = np.interp(index, index[good], track[good, axis])
    return filled


def track_coverage(track: np.ndarray) -> float:
    """Fração dos frames em que a mão foi vista."""
    track = np.asarray(track, dtype=float)
    return float(np.mean(~np.isnan(track[:, 0]))) if len(track) else 0.0


def hand_shape(landmarks_xy: np.ndarray) -> np.ndarray:
    """Os 21 pontos sem posição e sem escala: centrados e divididos pelo raio médio."""
    centered = np.asarray(landmarks_xy, dtype=float) - np.mean(landmarks_xy, axis=0)
    radius = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    return centered / radius if radius > 0 else centered


def shape_change(a: np.ndarray, b: np.ndarray) -> float:
    """Distância média entre os pontos das duas formas, em raios da mão."""
    return float(np.mean(np.linalg.norm(hand_shape(a) - hand_shape(b), axis=1)))


def find_pauses(
    track: np.ndarray,
    window: int = 5,
    motion_fraction: float = 0.4,
    hold_fraction: float = 0.15,
    frames_per_pause: int = 8,
    stride: int = 2,
    landmarks: np.ndarray | None = None,
    max_shape_change: float = 0.25,
) -> list[dict]:
    """As pausas da configuração: a mão levantada antes e depois da descida ao repouso.

    Nos clipes da base a mão começa e termina segurando a configuração, e o
    movimento do meio é a descida até o repouso e a volta (C1 a C6, takes 1 e 2).
    A descida é a janela entre o primeiro e o último frame com velocidade acima de
    `motion_fraction` do pico; ela fica de fora. Antes dela está a pausa 0, depois
    a pausa 1. Em cada lado, o centro é o frame vista mais lento, e a pausa é o
    trecho contínuo em volta dele com a mão vista e velocidade até `hold_fraction`
    do pico. As bordas do clipe (`window` frames) ficam de fora: a suavização
    repete a borda e baixa a velocidade ali. Do trecho saem até `frames_per_pause`
    frames, a partir do centro, de `stride` em `stride`.

    A média dos pontos fica parada enquanto os dedos ainda fecham. Com `landmarks`
    (n, 21, 2), o trecho também para no primeiro frame cuja forma se afasta da
    forma do centro mais que `max_shape_change` raios da mão.

    Cada pausa é {"pause", "center", "run": (início, fim), "frames"}, 0-based.
    """
    if not 0 < hold_fraction < motion_fraction < 1:
        raise ValueError(
            f"precisa 0 < hold_fraction < motion_fraction < 1, recebido "
            f"{hold_fraction} e {motion_fraction}"
        )
    if frames_per_pause < 1 or stride < 1:
        raise ValueError("frames_per_pause e stride precisam ser positivos")
    track = np.asarray(track, dtype=float)
    filled = fill_track(track)
    if filled is None:
        return []
    seen = ~np.isnan(track[:, 0])
    speed = np.where(seen, _hand_speed(filled, window), np.inf)
    finite = np.isfinite(speed)
    peak = float(np.max(speed[finite], initial=0.0))
    n = len(track)
    edge = max(int(window), 1)
    inner = np.zeros(n, dtype=bool)
    inner[edge : n - edge] = True
    calm = finite & inner & (speed <= hold_fraction * peak)
    moving = np.flatnonzero(finite & (speed >= motion_fraction * peak)) if peak > 0 else []
    if len(moving) == 0:
        sides = [(0, n)]
    else:
        sides = [(0, int(moving[0])), (int(moving[-1]) + 1, n)]

    pauses = []
    for pause_id, (start, stop) in enumerate(sides):
        candidates = start + np.flatnonzero(calm[start:stop])
        if candidates.size == 0:
            continue
        center = int(candidates[np.argmin(speed[candidates])])
        steady = calm.copy()
        if landmarks is not None:
            for frame in start + np.flatnonzero(calm[start:stop]):
                change = shape_change(landmarks[frame], landmarks[center])
                steady[frame] = change <= max_shape_change
        low = center
        while low - 1 >= start and steady[low - 1]:
            low -= 1
        high = center
        while high + 1 < stop and steady[high + 1]:
            high += 1
        frames = [center]
        step = 1
        while len(frames) < frames_per_pause and (
            center - step * stride >= low or center + step * stride <= high
        ):
            for frame in (center - step * stride, center + step * stride):
                if low <= frame <= high and len(frames) < frames_per_pause:
                    frames.append(frame)
            step += 1
        pauses.append(
            {"pause": pause_id, "center": center, "run": (low, high), "frames": sorted(frames)}
        )
    return pauses
