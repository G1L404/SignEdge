"""Vocabulário de um treino: as 61 configurações, com rótulo = código da base (CM − 1).

O dicionário vai para o `run.json` de cada execução e é o que a agregação filtra
(`vocab.name == "full"`).
"""

from __future__ import annotations

N_CODES = 61


def resolve_vocab(config: dict, name: str | None = None) -> dict:
    """{"name", "codes", "other", "names", "num_classes", "label_of"} das 61 classes."""
    if name not in (None, "full"):
        raise ValueError(f"vocabulário {name!r}: só existe o completo, com as 61 classes")
    codes = list(range(N_CODES))
    return {
        "name": "full",
        "codes": codes,
        "other": False,
        "names": [f"HC{code:02d}" for code in codes],
        "num_classes": N_CODES,
        "label_of": codes,
    }
