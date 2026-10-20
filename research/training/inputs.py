"""Entradas dos modelos a partir de uma vista (aumentada ou de avaliação)."""

from __future__ import annotations

import torch


def relative_depth(depth: torch.Tensor, landmarks: torch.Tensor, range_mm: float) -> torch.Tensor:
    """(z − mediana da profundidade nos 21 pontos) / range_mm, em [−1, 1]; sem leitura → +1.

    Sem nenhum ponto com leitura, a referência é a mediana da vista. O resultado
    descreve a forma 3D da mão, não a distância dela à câmera.
    """
    size = depth.shape[-1]
    index = landmarks.floor().long().clamp(0, size - 1)
    flat = depth.flatten(1)
    at_points = flat.gather(1, index[..., 1] * size + index[..., 0])
    nan = torch.full_like(at_points, float("nan"))
    z_hand = torch.where(at_points > 0, at_points, nan).nanmedian(dim=1).values
    z_view = torch.where(flat > 0, flat, torch.full_like(flat, float("nan"))).nanmedian(1).values
    z_ref = torch.where(torch.isnan(z_hand), z_view, z_hand).nan_to_num(1000.0)
    relative = ((depth - z_ref[:, None, None, None]) / range_mm).clamp(-1, 1)
    return torch.where(depth > 0, relative, torch.ones_like(relative))


def model_inputs(view: dict[str, torch.Tensor], range_mm: float) -> dict[str, torch.Tensor]:
    """rgb, landmarks e, se a vista tiver profundidade, a profundidade relativa."""
    inputs = {"landmarks": view["landmarks"]}
    if "rgb" in view:
        inputs["rgb"] = view["rgb"].contiguous(memory_format=torch.channels_last)
    if "depth" in view:
        rel = relative_depth(view["depth"], view["landmarks"], range_mm)
        inputs["depth"] = rel.contiguous(memory_format=torch.channels_last)
    return inputs
