"""Leitura dos YAML de configs/. Único lugar que abre arquivo de configuração."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Carrega um YAML de configuração e exige um mapeamento na raiz."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config inválida (esperado um mapeamento): {config_path}")
    return data
