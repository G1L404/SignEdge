"""Perdas: KL contra o professor e entropia cruzada com rótulos misturados."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float):
    """KL(professor ‖ aluno) nas distribuições com temperatura, × T² para manter a escala."""
    student = F.log_softmax(student_logits.float() / temperature, dim=1)
    teacher = F.log_softmax(teacher_logits.float() / temperature, dim=1)
    kl = F.kl_div(student, teacher, log_target=True, reduction="batchmean")
    return kl * temperature * temperature


def mixed_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lam: float = 1.0,
    perm: torch.Tensor | None = None,
    smoothing: float = 0.0,
) -> torch.Tensor:
    """Entropia cruzada; com mixup, a média dos dois rótulos ponderada por lambda."""
    loss = F.cross_entropy(logits.float(), labels, label_smoothing=smoothing)
    if perm is None or lam >= 1.0:
        return loss
    other = F.cross_entropy(logits.float(), labels[perm], label_smoothing=smoothing)
    return lam * loss + (1 - lam) * other
