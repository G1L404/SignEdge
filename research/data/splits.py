"""Divisão leave-one-signer-out e amostragem de um frame por pausa.

Os frames de uma pausa são quase o mesmo exemplo; a unidade independente é a
pausa, e a divisão nunca separa frames de um mesmo ator. Cada fold treina em
todos os atores menos um e valida nesse um; nada é escolhido pelo ator de
validação dentro do fold (o checkpoint é o último, com passos fixos). Os
hiperparâmetros saem da média dos folds.

Os atores fora da distribuição (C6, gravado em 2012 por outra pessoa e em outro
lugar) não entram em fold nenhum: `all_actors_split` treina com todos os atores
dos folds e mede o C6 uma única vez, no fim.
"""

from __future__ import annotations

import random
from collections import defaultdict


def _by_actor(samples: dict[str, dict], actors: list[str], ood: list[str]) -> dict:
    overlap = set(actors) & set(ood)
    if overlap:
        raise ValueError(f"ator nos folds e no OOD ao mesmo tempo: {sorted(overlap)}")
    by_actor: dict[str, list[str]] = defaultdict(list)
    for key, meta in samples.items():
        by_actor[meta["actor"]].append(key)
    unknown = set(by_actor) - set(actors) - set(ood)
    if unknown:
        raise ValueError(f"atores sem papel na divisão: {sorted(unknown)}")
    return by_actor


def loso_folds(samples: dict[str, dict], actors: list[str], ood: list[str]) -> list[dict]:
    """Um fold por ator de validação, que treina em todos os outros atores dos folds.

    `samples` é `__meta__["samples"]`. Cada fold traz as chaves de train e val,
    ordenadas. Os atores de `ood` não aparecem em fold nenhum.
    """
    if len(actors) < 2:
        raise ValueError("LOSO precisa de pelo menos 2 atores")
    by_actor = _by_actor(samples, actors, ood)
    folds = []
    for val in actors:
        train = [actor for actor in actors if actor != val]
        folds.append(
            {
                "val_actor": val,
                "train_actors": train,
                "train": sorted(key for actor in train for key in by_actor[actor]),
                "val": sorted(by_actor[val]),
            }
        )
    return folds


def all_actors_split(samples: dict[str, dict], actors: list[str], ood: list[str]) -> dict:
    """Treino com todos os atores dos folds; o OOD é o teste final, medido uma vez."""
    by_actor = _by_actor(samples, actors, ood)
    return {
        "train_actors": list(actors),
        "ood_actors": list(ood),
        "train": sorted(key for actor in actors for key in by_actor[actor]),
        "ood": sorted(key for actor in ood for key in by_actor[actor]),
    }


def group_by_pause(keys: list[str], samples: dict[str, dict]) -> list[list[int]]:
    """Índices de `keys` agrupados por (clipe, pausa), na ordem das chaves."""
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        meta = samples[key]
        groups[(meta["clip"], meta["pause"])].append(index)
    return list(groups.values())


class PauseSampler:
    """Um frame sorteado por pausa a cada época, em ordem embaralhada.

    Pausas com 16 frames pesam o mesmo que pausas com 5. Os outros frames da
    pausa aparecem em outras épocas, como variação natural do mesmo exemplo.
    Com `labels` (um rótulo por chave) e `caps` ({rótulo: máximo de pausas por
    época}), um rótulo com pausas demais (a classe "outra" do vocabulário da
    demo) entra com um sorteio novo de `cap` pausas a cada época.
    """

    def __init__(
        self,
        keys: list[str],
        samples: dict[str, dict],
        seed: int = 0,
        labels: list[int] | None = None,
        caps: dict[int, int] | None = None,
    ) -> None:
        self._groups = group_by_pause(keys, samples)
        self._seed = seed
        self._epoch = 0
        self._caps = dict(caps or {})
        self._labels = [None if labels is None else labels[g[0]] for g in self._groups]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        total = 0
        for label in set(self._labels):
            count = self._labels.count(label)
            total += min(count, self._caps.get(label, count))
        return total

    def __iter__(self):
        rng = random.Random(self._seed * 1_000_003 + self._epoch)
        by_label: dict = defaultdict(list)
        for group, label in zip(self._groups, self._labels, strict=True):
            by_label[label].append(group)
        groups = []
        for label in sorted(by_label, key=str):
            members = by_label[label]
            cap = self._caps.get(label)
            groups.extend(rng.sample(members, cap) if cap and cap < len(members) else members)
        picks = [rng.choice(group) for group in groups]
        rng.shuffle(picks)
        return iter(picks)
