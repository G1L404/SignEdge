"""Caixa da mão, a margem em volta dela e o recorte conjunto de RGB, profundidade e pontos.

Caixas são (x_min, y_min, x_max, y_max) em pixels, canto final exclusivo. A regra
da caixa é a mesma do app: lado dos 21 pontos × `hand_scale`, com piso de
`min_box_frac` da altura da foto, e a margem multiplica esse lado.
"""

from __future__ import annotations

import numpy as np

from research.ingestion.hand_detector import hand_square


def min_box_px(frame_height: int, min_box_frac: float) -> int:
    """Piso do lado da caixa em pixels. Na foto de 480 px, 112/480 dá 112."""
    if frame_height <= 0 or not 0 < min_box_frac <= 1:
        raise ValueError("frame_height precisa ser positivo e min_box_frac estar em (0, 1]")
    return int(round(frame_height * min_box_frac))


def hand_box(
    landmarks_xy: np.ndarray, frame_height: int, hand_scale: float, min_box_frac: float
) -> tuple[int, int, int, int]:
    """Caixa justa da mão, a que o app desenha."""
    return hand_square(landmarks_xy, hand_scale, min_box_px(frame_height, min_box_frac))


def context_box(box: tuple[int, int, int, int], context_scale: float) -> tuple[int, int, int, int]:
    """O mesmo centro, com o lado multiplicado por `context_scale`."""
    if context_scale < 1:
        raise ValueError(f"context_scale precisa ser pelo menos 1, recebido {context_scale}")
    x_min, y_min, x_max, y_max = box
    side = int(round((x_max - x_min) * context_scale))
    x0 = int(round((x_min + x_max) / 2 - side / 2))
    y0 = int(round((y_min + y_max) / 2 - side / 2))
    return (x0, y0, x0 + side, y0 + side)


def shift_depth(depth: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Move o conteúdo da profundidade (dx, dy) pixels. O que entra pela borda é 0."""
    dx, dy = int(dx), int(dy)
    if dx == 0 and dy == 0:
        return depth
    height, width = depth.shape[:2]
    out = np.zeros_like(depth)
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return out
    out[src_y0 + dy : src_y1 + dy, src_x0 + dx : src_x1 + dx] = depth[src_y0:src_y1, src_x0:src_x1]
    return out


def shift_square_inside(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Empurra o quadrado para dentro da foto. O lado não muda."""
    x_min, y_min, x_max, y_max = (int(value) for value in bbox)
    side = x_max - x_min
    if side <= 0 or y_max - y_min != side:
        raise ValueError(f"a caixa precisa ser um quadrado positivo, recebida {bbox}")
    if side <= width:
        if x_min < 0:
            x_max -= x_min
            x_min = 0
        elif x_max > width:
            x_min -= x_max - width
            x_max = width
    if side <= height:
        if y_min < 0:
            y_max -= y_min
            y_min = 0
        elif y_max > height:
            y_min -= y_max - height
            y_max = height
    return (x_min, y_min, x_max, y_max)


def crop_square(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Recorta o quadrado. Se ele cabe na foto, entra inteiro, sem tarja de borda."""
    height, width = image.shape[:2]
    x_min, y_min, x_max, y_max = shift_square_inside(bbox, width, height)
    src_x0, src_y0 = max(x_min, 0), max(y_min, 0)
    src_x1, src_y1 = min(x_max, width), min(y_max, height)
    if src_x0 >= src_x1 or src_y0 >= src_y1:
        raise ValueError(f"a caixa {bbox} cai fora da imagem {width}×{height}")
    patch = image[src_y0:src_y1, src_x0:src_x1]
    top, left = src_y0 - y_min, src_x0 - x_min
    bottom, right = y_max - src_y1, x_max - src_x1
    if top == left == bottom == right == 0:
        return patch
    pad = ((top, bottom), (left, right)) + ((0, 0),) * (image.ndim - 2)
    return np.pad(patch, pad, mode="edge")


def crop_sample(
    color: np.ndarray,
    depth: np.ndarray,
    landmarks_xy: np.ndarray,
    box: tuple[int, int, int, int],
    context_scale: float,
    size: int,
) -> dict:
    """Recorte com margem: RGB uint8 (3, S, S), profundidade uint16 (S, S) em mm e pontos.

    Os pontos e a caixa justa saem nas coordenadas do recorte, em pixels de S.
    A profundidade é redimensionada pelo vizinho mais próximo: o valor em mm
    nunca mistura a mão com o fundo.
    """
    import cv2

    if size <= 0:
        raise ValueError(f"size precisa ser positivo, recebido {size}")
    height, width = color.shape[:2]
    if depth.shape[:2] != (height, width):
        raise ValueError(
            f"profundidade {depth.shape[:2]} e cor {(height, width)} em grades diferentes"
        )
    outer = shift_square_inside(context_box(box, context_scale), width, height)
    side = outer[2] - outer[0]
    rgb_interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    rgb = cv2.resize(crop_square(color, outer), (size, size), interpolation=rgb_interp)
    deep = cv2.resize(crop_square(depth, outer), (size, size), interpolation=cv2.INTER_NEAREST)
    scale = size / side
    origin = np.array([outer[0], outer[1]], dtype=float)
    points = ((np.asarray(landmarks_xy, dtype=float) - origin) * scale).astype(np.float32)
    inner = (np.array(box, dtype=float).reshape(2, 2) - origin) * scale
    return {
        "rgb": np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))),
        "depth": np.ascontiguousarray(deep.astype(np.uint16)),
        "landmarks": points,
        "hand_box": [round(float(value), 2) for value in inner.reshape(-1)],
        "context_box": [int(value) for value in outer],
    }
