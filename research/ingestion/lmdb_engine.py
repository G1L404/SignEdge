"""Banco em LMDB. A abertura e o fechamento do ambiente ficam no mesmo objeto.

O caminho e o map_size vêm de configs/data_pipeline.yaml; o código não escolhe
o diretório.

Cada amostra é um registro binário, nesta ordem e little-endian:
RGB uint8 (3, S, S), profundidade uint16 (S, S) em mm e 21 pontos float32 (21, 2)
em pixels do recorte. S vem de `__meta__["layout"]["size"]`. A chave `__meta__`
guarda, em JSON, o layout, a regra do recorte e, por amostra, o clipe de origem,
ator, take, classe, pausa, frame e o lado da mão. A passagem para float acontece
no Dataset, quando o DataLoader monta o lote.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

META_KEY = b"__meta__"
N_LANDMARKS = 21


def record_bytes(size: int) -> int:
    return 3 * size * size + 2 * size * size + N_LANDMARKS * 2 * 4


def pack_sample(rgb: np.ndarray, depth: np.ndarray, landmarks: np.ndarray) -> bytes:
    """Junta RGB, profundidade e pontos num registro. Recusa forma ou tipo fora do layout."""
    size = rgb.shape[-1]
    if rgb.shape != (3, size, size) or rgb.dtype != np.uint8:
        raise ValueError(f"RGB precisa ser uint8 (3, S, S), recebido {rgb.dtype} {rgb.shape}")
    if depth.shape != (size, size) or depth.dtype != np.uint16:
        raise ValueError(
            f"profundidade precisa ser uint16 ({size}, {size}), "
            f"recebida {depth.dtype} {depth.shape}"
        )
    if landmarks.shape != (N_LANDMARKS, 2):
        raise ValueError(f"pontos precisam ser ({N_LANDMARKS}, 2), recebidos {landmarks.shape}")
    return b"".join(
        (
            np.ascontiguousarray(rgb).tobytes(),
            np.ascontiguousarray(depth, dtype="<u2").tobytes(),
            np.ascontiguousarray(landmarks, dtype="<f4").tobytes(),
        )
    )


def unpack_sample(raw: bytes, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separa o registro em RGB, profundidade e pontos, sem copiar."""
    if len(raw) != record_bytes(size):
        raise ValueError(
            f"registro com {len(raw)} bytes; o layout de S={size} tem {record_bytes(size)}"
        )
    rgb_end = 3 * size * size
    depth_end = rgb_end + 2 * size * size
    rgb = np.frombuffer(raw, dtype=np.uint8, count=rgb_end).reshape(3, size, size)
    depth = np.frombuffer(raw, dtype="<u2", count=size * size, offset=rgb_end).reshape(size, size)
    points = np.frombuffer(raw, dtype="<f4", offset=depth_end).reshape(N_LANDMARKS, 2)
    return rgb, depth, points


class LMDBStore:
    """Gravação e leitura do LMDB. `__exit__` fecha o ambiente."""

    def __init__(self, path: str | Path, map_size: int, readonly: bool = False) -> None:
        if map_size <= 0:
            raise ValueError(f"map_size precisa ser positivo, recebido {map_size}")
        self.path = Path(path)
        self.map_size = map_size
        self.readonly = readonly
        self._env = None

    def __enter__(self) -> LMDBStore:
        import lmdb

        if not self.readonly:
            self.path.mkdir(parents=True, exist_ok=True)
        elif not self.path.is_dir():
            raise FileNotFoundError(f"LMDB ausente em {self.path}. Rode `make ingest`.")
        self._env = lmdb.open(
            str(self.path),
            map_size=self.map_size,
            readonly=self.readonly,
            lock=not self.readonly,
            readahead=False,
            meminit=False,
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        env = self._env
        self._env = None
        if env is not None:
            env.close()

    def _require_env(self):
        if self._env is None:
            raise RuntimeError("abra o LMDBStore com `with` antes de ler ou gravar")
        return self._env

    def write_samples(self, samples: Iterable[tuple[bytes, bytes]], batch: int = 64) -> int:
        """Grava pares (chave, valor) e devolve a quantidade escrita."""
        env = self._require_env()
        count = 0
        txn = env.begin(write=True)
        try:
            for key, value in samples:
                txn.put(key, value)
                count += 1
                if count % batch == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            txn.commit()
        except BaseException:
            txn.abort()
            raise
        return count

    def read_sample(self, key: bytes) -> bytes:
        """Lê um valor pela chave."""
        with self._require_env().begin() as txn:
            value = txn.get(key)
        if value is None:
            raise KeyError(key)
        return bytes(value)

    def write_meta(self, meta: dict) -> None:
        self.write_samples([(META_KEY, json.dumps(meta, ensure_ascii=False).encode("utf-8"))])

    def read_meta(self) -> dict:
        return json.loads(self.read_sample(META_KEY))


class SignEdgeDataset:
    """Lê o registro uint8/uint16 e só então entrega float32.

    Cada item é um dicionário: "rgb" (3, S, S) em [0, 1], "depth" (S, S) em mm
    (0 = sem leitura), "landmarks" (21, 2) em pixels do recorte e "label", a classe.
    """

    def __init__(self, store: LMDBStore, keys: list[str], meta: dict | None = None) -> None:
        meta = store.read_meta() if meta is None else meta
        self._store = store
        self._keys = list(keys)
        self._size = int(meta["layout"]["size"])
        self._labels = [int(meta["samples"][key]["cls"]) for key in self._keys]

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, index: int) -> dict:
        import torch

        raw = self._store.read_sample(self._keys[index].encode("utf-8"))
        rgb, depth, points = unpack_sample(raw, self._size)
        return {
            "rgb": torch.tensor(rgb, dtype=torch.float32).div_(255.0),
            "depth": torch.tensor(depth.astype(np.float32)),
            "landmarks": torch.tensor(points),
            "label": torch.tensor(self._labels[index], dtype=torch.int64),
        }
