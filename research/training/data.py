"""As amostras de um conjunto em memória, como uint8/int16, prontas para virar lote na GPU.

Um fold de treino tem ~4 mil frames (~1,6 GB): cabe na RAM e dispensa workers
do DataLoader, que com LMDB e fork só trazem problema. A augmentation roda na GPU.
"""

from __future__ import annotations

import numpy as np
import torch

from research.ingestion.lmdb_engine import LMDBStore, unpack_sample


class InMemorySplit:
    """RGB (N, 3, S, S) uint8, profundidade (N, S, S) int16 em mm, pontos e rótulos."""

    def __init__(
        self,
        store: LMDBStore,
        keys: list[str],
        meta: dict,
        label_of: list[int | None] | None = None,
    ) -> None:
        """`label_of[código]` é o rótulo do treino para a classe da base; None tira a amostra."""
        samples = meta["samples"]
        if label_of is None:
            label_of = list(range(61))
        keys = [key for key in keys if label_of[int(samples[key]["cls"])] is not None]
        if not keys:
            raise ValueError("conjunto vazio")
        size = int(meta["layout"]["size"])
        pin = torch.cuda.is_available()
        n = len(keys)
        self.keys = list(keys)
        self.size = size
        self.rgb = torch.empty((n, 3, size, size), dtype=torch.uint8, pin_memory=pin)
        self.depth = torch.empty((n, size, size), dtype=torch.int16, pin_memory=pin)
        self.landmarks = torch.empty((n, 21, 2), dtype=torch.float32)
        for index, key in enumerate(self.keys):
            rgb, depth, points = unpack_sample(store.read_sample(key.encode("utf-8")), size)
            self.rgb[index] = torch.from_numpy(rgb.copy())
            self.depth[index] = torch.from_numpy(depth.astype(np.int16))
            self.landmarks[index] = torch.from_numpy(points.copy())
        self.codes = torch.tensor([int(samples[key]["cls"]) for key in self.keys])
        self.labels = torch.tensor([label_of[int(code)] for code in self.codes])
        pauses: dict[tuple[str, int], int] = {}
        self.pause_ids = torch.tensor(
            [
                pauses.setdefault((samples[key]["clip"], samples[key]["pause"]), len(pauses))
                for key in self.keys
            ]
        )
        self.n_pauses = len(pauses)

    def __len__(self) -> int:
        return len(self.keys)

    def batch(self, index, device) -> tuple[torch.Tensor, ...]:
        index = torch.as_tensor(index, dtype=torch.long)
        return (
            self.rgb[index].to(device, non_blocking=True),
            self.depth[index].to(device, non_blocking=True).float(),
            self.landmarks[index].to(device, non_blocking=True),
            self.labels[index].to(device, non_blocking=True),
        )
