"""Métricas por frame e por pausa. A pausa (média dos logits dos seus frames) é a unidade
independente; o frame mede o que o app vê num instante."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def macro_f1(pred: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    scores = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (labels == c))
        fp = np.sum((pred == c) & (labels != c))
        fn = np.sum((pred != c) & (labels == c))
        if tp + fp + fn == 0:
            continue
        scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else float("nan")


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            total += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return float(total)


def pause_logits(
    logits: np.ndarray, labels: np.ndarray, pause_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Média dos logits por pausa e o rótulo de cada pausa."""
    order = np.unique(pause_ids)
    pooled = np.stack([logits[pause_ids == p].mean(axis=0) for p in order])
    pooled_labels = np.array([labels[pause_ids == p][0] for p in order])
    return pooled, pooled_labels


def summarize(logits: np.ndarray, labels: np.ndarray, pause_ids: np.ndarray) -> dict:
    n_classes = logits.shape[1]
    probs = softmax(logits)
    top5 = np.argsort(-logits, axis=1)[:, :5]
    pooled, pooled_labels = pause_logits(logits, labels, pause_ids)
    pause_pred = pooled.argmax(axis=1)
    nll = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean()
    return {
        "frame_acc": round(float((logits.argmax(axis=1) == labels).mean()), 4),
        "frame_top5": round(float((top5 == labels[:, None]).any(axis=1).mean()), 4),
        "pause_acc": round(float((pause_pred == pooled_labels).mean()), 4),
        "pause_macro_f1": round(macro_f1(pause_pred, pooled_labels, n_classes), 4),
        "frame_ece": round(expected_calibration_error(probs, labels), 4),
        "frame_nll": round(float(nll), 4),
        "n_frames": int(len(labels)),
        "n_pauses": int(len(pooled_labels)),
    }
