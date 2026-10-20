"""Vocabulário de um treino: as 61 configurações ou um subconjunto, com "outra" opcional.

Os códigos são os da base (0..60, CM = código + 1). No subconjunto, as classes
ganham rótulos 0..k-1 na ordem da lista; com `other`, todas as demais viram o
rótulo k ("outra"), para o app não forçar uma resposta quando a mão faz uma
configuração fora do vocabulário. Sem `other`, as demais saem do treino.
"""

from __future__ import annotations

N_CODES = 61
OTHER_NAME = "outra"


def resolve_vocab(config: dict, name: str | None) -> dict:
    """{"name", "codes", "other", "other_cap_factor", "names", "num_classes", "label_of"}.

    `label_of[código]` é o rótulo do treino, ou None para códigos que saem.
    """
    if name is None or name == "full":
        codes = list(range(N_CODES))
        other, cap = False, None
    else:
        entry = config["vocabularies"][name]
        codes = [int(c) for c in entry["codes"]]
        other = bool(entry.get("other", False))
        cap = float(entry["other_cap_factor"]) if other else None
    if len(set(codes)) != len(codes) or not all(0 <= c < N_CODES for c in codes):
        raise ValueError(f"vocabulário {name} com código repetido ou fora de 0..{N_CODES - 1}")
    label_of: list[int | None] = [None] * N_CODES
    for label, code in enumerate(codes):
        label_of[code] = label
    names = [f"HC{code:02d}" for code in codes]
    if other:
        for code in range(N_CODES):
            if label_of[code] is None:
                label_of[code] = len(codes)
        names.append(OTHER_NAME)
    return {
        "name": name or "full",
        "codes": codes,
        "other": other,
        "other_cap_factor": cap,
        "names": names,
        "num_classes": len(names),
        "label_of": label_of,
    }
