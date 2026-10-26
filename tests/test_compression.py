"""Sanidade da poda: peso mascarado vira zero e permanece zero."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from research.config import load_config
from research.optimization.pruner import sparsity, zero_pruned

ROOT = Path(__file__).resolve().parents[1]


def test_pruned_weights_are_exactly_zero() -> None:
    weights = np.array([1.0, 2.5, -3.0, 4.0])
    mask = np.array([True, False, True, False])
    pruned = zero_pruned(weights, mask)

    assert pruned[1] == 0.0
    assert pruned[3] == 0.0
    assert pruned[0] == 1.0
    assert pruned[2] == -3.0
    assert sparsity(pruned) == 0.5
    assert weights[1] == 2.5


def test_compression_amount_is_configured() -> None:
    config = load_config(ROOT / "configs" / "distillation.yaml")
    amount = config["compression"]["prune_amount"]
    assert 0 < amount < 1
