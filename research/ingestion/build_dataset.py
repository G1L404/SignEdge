"""Ingestão da base: cada .oni vira até 16 amostras no LMDB, ou um motivo.

De cada clipe saem os frames das duas pausas da configuração (até 8 por pausa).
Cada amostra guarda o recorte com margem em RGB, a profundidade crua no mesmo
recorte e os 21 pontos da mão (layout em `lmdb_engine`). A chave é
`ator_take_classe_pPausa_fFrame`, com o frame 1-based do .oni.

Cada clipe sai como uma linha do manifesto, inclusive os vazios e os recusados.
O manifesto guarda a regra do recorte (a mesma do app) e os hashes do YAML e do
modelo do MediaPipe: com eles, um LMDB antigo diz de qual política saiu.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np

from research.config import load_config
from research.ingestion.crop import crop_sample, hand_box
from research.ingestion.errors import SampleRejected
from research.ingestion.frame_clock import clip_pauses, scan_clip
from research.ingestion.lmdb_engine import N_LANDMARKS, LMDBStore, pack_sample, record_bytes
from research.ingestion.oni_io import require_playable_oni
from research.ingestion.registration import load_registration

REPO_ROOT = Path(__file__).resolve().parents[2]
_LMDB_FILES = {"data.mdb", "lock.mdb", "manifest.json"}
_TEARDOWNS = {"close", "exit"}


def clip_identity(oni_path) -> dict:
    """Ator, take e classe pelo nome do arquivo, e a chave canônica do clipe.

    O C6 grava o take com dois dígitos (C6_01_10); a chave fica C6_1_10.
    """
    name = Path(oni_path).stem
    parts = name.split("_")
    if len(parts) != 3 or not parts[0].startswith("C"):
        raise ValueError(f"nome fora do padrão ator_take_classe: {name}")
    actor, take, cls = parts[0], int(parts[1]), int(parts[2])
    return {"key": f"{actor}_{take}_{cls}", "actor": actor, "take": take, "cls": cls, "file": name}


def sample_key(clip_key: str, pause: int, frame: int) -> str:
    """Chave da amostra. `frame` é 1-based, como no .oni."""
    return f"{clip_key}_p{pause}_f{frame:03d}"


def list_clips(raw_root: Path, sessions: list[str] | None = None) -> list[Path]:
    """Todos os .oni das sessões, vazios inclusive: eles também entram no manifesto."""
    if not raw_root.is_dir():
        raise FileNotFoundError(
            f"Base não encontrada em {raw_root}. Crie o symlink data/raw/libras-hc-rgbds."
        )
    dirs = sorted(path for path in raw_root.iterdir() if path.is_dir())
    if sessions is not None:
        wanted = set(sessions)
        missing = wanted - {path.name for path in dirs}
        if missing:
            raise FileNotFoundError(f"sessões ausentes em {raw_root}: {sorted(missing)}")
        dirs = [path for path in dirs if path.name in wanted]
    clips = []
    for session in dirs:
        found = sorted(session.glob("*.oni"), key=lambda path: clip_identity(path)["cls"])
        clips.extend(found)
    return clips


def _finite_or_none(value: float) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), 1)


def process_clip(task: tuple[str, dict]) -> tuple[dict, list[tuple[str, bytes, dict]]]:
    """Um clipe vira (linha do manifesto, amostras). Recusado ou com erro, sem amostras.

    `task` é (caminho do .oni, parâmetros de `_params`). Cada amostra é
    (chave, registro do LMDB, metadados do `__meta__`).
    """
    oni_path, params = task
    oni_path = Path(oni_path)
    identity = clip_identity(oni_path)
    record = {**identity, "status": "recusado", "reason": None}
    samples: list[tuple[str, bytes, dict]] = []
    try:
        require_playable_oni(oni_path)
        session = oni_path.parent.name
        if session not in params["depth_shifts"]:
            raise ValueError(f"sessão {session} sem calibração da profundidade")
        depth_shift = tuple(params["depth_shifts"][session])
        record["depth_shift"] = list(depth_shift)
        pairs, hands, landmarks = scan_clip(
            oni_path,
            params["mp_model"],
            params["mp_min_confidence"],
            close=params["openni_teardown"] == "close",
            depth_shift=depth_shift,
        )
        record["n_frames"] = len(pairs)
        record["coverage"] = round(float(np.mean([hand is not None for hand in hands])), 3)
        pauses = clip_pauses(
            landmarks,
            name=oni_path.name,
            min_track_coverage=params["min_track_coverage"],
            window=params["window"],
            motion_fraction=params["motion_fraction"],
            hold_fraction=params["hold_fraction"],
            frames_per_pause=params["frames_per_pause"],
            stride=params["frame_stride"],
            max_shape_change=params["max_shape_change"],
        )
        for pause in pauses:
            for index in pause["frames"]:
                depth, color = pairs[index]
                hand = hands[index]
                box = hand_box(
                    hand["xy"], color.shape[0], params["hand_scale"], params["min_box_frac"]
                )
                crop = crop_sample(
                    color, depth, hand["xy"], box, params["context_scale"], params["crop_size"]
                )
                key = sample_key(identity["key"], pause["pause"], index + 1)
                blob = pack_sample(crop["rgb"], crop["depth"], crop["landmarks"])
                meta = {
                    "clip": identity["key"],
                    "actor": identity["actor"],
                    "take": identity["take"],
                    "cls": identity["cls"],
                    "pause": pause["pause"],
                    "frame": index + 1,
                    "handedness": hand["label"],
                    "hand_score": round(hand["score"], 3),
                    "hand_depth_mm": _finite_or_none(hand["depth_mm"]),
                    "n_hands": hand["n_hands"],
                    "hand_box": crop["hand_box"],
                    "context_box": crop["context_box"],
                    "sha256": hashlib.sha256(blob).hexdigest(),
                }
                samples.append((key, blob, meta))
    except SampleRejected as exc:
        record["reason"] = str(exc)
        return record, []
    except Exception as exc:
        record.update(status="erro", reason=f"{type(exc).__name__}: {exc}")
        return record, []
    record.update(
        status="aceito",
        pauses=[
            {
                "pause": pause["pause"],
                "center": pause["center"] + 1,
                "run": [pause["run"][0] + 1, pause["run"][1] + 1],
                "frames": [index + 1 for index in pause["frames"]],
            }
            for pause in pauses
        ],
        samples={key: meta["sha256"] for key, _, meta in samples},
    )
    return record, samples


_CHILD = (
    "import importlib, os, pickle, sys\n"
    "module, name = sys.argv[3].split(':')\n"
    "target = getattr(importlib.import_module(module), name)\n"
    "with open(sys.argv[1], 'rb') as handle:\n"
    "    task = pickle.load(handle)\n"
    "with open(sys.argv[2], 'wb') as handle:\n"
    "    pickle.dump(target(task), handle)\n"
    "if task[1]['openni_teardown'] == 'exit':\n"
    "    os._exit(0)\n"
)


def run_isolated(
    tasks,
    workers: int,
    attempts: int = 3,
    timeout_s: float = 180,
    target: str = "research.ingestion.build_dataset:process_clip",
):
    """`target(task)` num subprocesso por clipe. Gera (resultado, tentativas, falha) na ordem.

    A thread de reprodução do libOniFile às vezes cai com 8 ou mais processos
    lendo ao mesmo tempo, e num pool o segfault derruba o lote. Aqui ele custa
    uma nova tentativa: o .oni devolve os mesmos frames a cada leitura. Depois de
    `attempts` falhas, o resultado é None e a falha diz por quê.
    """
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}

    def run(task):
        failure = None
        with tempfile.TemporaryDirectory(prefix="signedge-") as tmp:
            task_path, out_path = Path(tmp) / "task.pkl", Path(tmp) / "out.pkl"
            task_path.write_bytes(pickle.dumps(task))
            for attempt in range(1, attempts + 1):
                out_path.unlink(missing_ok=True)
                try:
                    code = subprocess.run(
                        [sys.executable, "-c", _CHILD, str(task_path), str(out_path), target],
                        cwd=REPO_ROOT,
                        env=env,
                        capture_output=True,
                        timeout=timeout_s,
                    ).returncode
                except subprocess.TimeoutExpired:
                    failure = f"travou por mais de {timeout_s:.0f} s"
                    continue
                if code == 0 and out_path.is_file():
                    return pickle.loads(out_path.read_bytes()), attempt, None
                failure = f"processo terminou com código {code}"
        return None, attempts, f"{failure} nas {attempts} tentativas"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(run, tasks)


def _clip_results(tasks, stream):
    """(linha, amostras) de cada clipe; a falha do subprocesso vira linha de erro."""
    for task, (result, attempts, failure) in zip(tasks, stream):
        if result is None:
            record = {**clip_identity(task[0]), "status": "erro", "reason": failure}
            samples = []
        else:
            record, samples = result
        record["attempts"] = attempts
        yield record, samples


def _repo_path(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _params(config: dict) -> dict:
    crop, clock, ingest = config["crop"], config["clock"], config["ingest"]
    teardown = str(ingest["openni_teardown"])
    if teardown not in _TEARDOWNS:
        raise ValueError(f"ingest.openni_teardown precisa ser close ou exit, recebido {teardown}")
    return {
        "mp_model": str(_repo_path(crop["mp_model"])),
        "mp_min_confidence": float(crop["mp_min_confidence"]),
        "hand_scale": float(crop["hand_scale"]),
        "min_box_frac": float(crop["min_box_frac"]),
        "context_scale": float(crop["context_scale"]),
        "crop_size": int(crop["crop_size"]),
        "min_track_coverage": float(clock["min_track_coverage"]),
        "window": int(clock["smooth_window"]),
        "motion_fraction": float(clock["motion_fraction"]),
        "hold_fraction": float(clock["hold_fraction"]),
        "frames_per_pause": int(clock["frames_per_pause"]),
        "frame_stride": int(clock["frame_stride"]),
        "max_shape_change": float(clock["max_shape_change"]),
        "openni_teardown": teardown,
    }


def crop_rule(params: dict) -> dict:
    """A regra do recorte que o app precisa repetir, sem nada do .oni."""
    return {
        "hand_center": "meio da extensão dos 21 pontos do MediaPipe em x e em y",
        "hand_side": "maior extensão dos 21 pontos × hand_scale, piso min_box_frac × altura",
        "hand_scale": params["hand_scale"],
        "min_box_frac": params["min_box_frac"],
        "context_scale": params["context_scale"],
        "crop_size": params["crop_size"],
        "two_hands": "a mais perto da câmera (mediana da profundidade nos 21 pontos)",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _versions() -> dict:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "opencv-python", "mediapipe", "openni", "lmdb"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _prepare_lmdb_dir(path: Path, overwrite: bool) -> None:
    """Nunca mistura chaves de duas políticas: o diretório precisa estar vazio."""
    if not path.exists() or not any(path.iterdir()):
        return
    if not overwrite:
        raise FileExistsError(f"LMDB já existe em {path}. Use --overwrite para refazer.")
    stray = {child.name for child in path.iterdir()} - _LMDB_FILES
    if stray:
        raise FileExistsError(f"{path} tem arquivos que não são do LMDB: {sorted(stray)}")
    shutil.rmtree(path)


def _stream_samples(results, records: list[dict], sample_meta: dict):
    for record, samples in results:
        records.append(record)
        mark = f"{len(samples)} amostras" if record["status"] == "aceito" else record["status"]
        tries = f" ({record['attempts']} tentativas)" if record.get("attempts", 1) > 1 else ""
        print(
            f"[{len(records)}] {record['key']}: {mark}{tries} {record['reason'] or ''}", flush=True
        )
        for key, blob, meta in samples:
            sample_meta[key] = meta
            yield key.encode("utf-8"), blob


def build(
    config_path: Path,
    lmdb_path: Path | None = None,
    manifest_path: Path | None = None,
    sessions: list[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
    workers: int | None = None,
    clip_keys: list[str] | None = None,
) -> dict:
    """Grava o LMDB e o manifesto. Devolve o manifesto."""
    config_path = Path(config_path)
    config = load_config(config_path)
    params = _params(config)
    registration_path = _repo_path(config["paths"]["registration_path"])
    registration = load_registration(registration_path)
    params["depth_shifts"] = {
        session: (shift["dx"], shift["dy"]) for session, shift in registration["sessions"].items()
    }
    lmdb_path = _repo_path(lmdb_path or config["paths"]["lmdb_path"])
    manifest_path = _repo_path(manifest_path or config["paths"]["manifest_path"])
    clips = list_clips(_repo_path(config["paths"]["raw_root"]), sessions)
    if clip_keys is not None:
        by_key = {clip_identity(path)["key"]: path for path in clips}
        missing = sorted(set(clip_keys) - set(by_key))
        if missing:
            raise FileNotFoundError(f"clipes ausentes: {missing}")
        clips = [by_key[key] for key in clip_keys]
    if limit is not None:
        clips = clips[:limit]
    keys = [clip_identity(path)["key"] for path in clips]
    if len(set(keys)) != len(keys):
        raise ValueError("duas gravações caem na mesma chave canônica")
    _prepare_lmdb_dir(lmdb_path, overwrite)

    ingest = config["ingest"]
    records: list[dict] = []
    sample_meta: dict[str, dict] = {}
    tasks = [(str(path), params) for path in clips]
    stream = run_isolated(
        tasks,
        workers or int(ingest["num_workers"]),
        attempts=int(ingest["attempts"]),
        timeout_s=float(ingest["timeout_s"]),
    )
    results = _clip_results(tasks, stream)
    depth_registration = {
        "file": str(registration_path),
        "sha256": _sha256(registration_path),
        "shifts_px": {session: list(shift) for session, shift in params["depth_shifts"].items()},
    }
    size = params["crop_size"]
    layout = {
        "size": size,
        "rgb": ["uint8", [3, size, size]],
        "depth": ["uint16_le_mm", [size, size]],
        "landmarks": ["float32_le_px", [N_LANDMARKS, 2]],
        "record_bytes": record_bytes(size),
    }
    with LMDBStore(lmdb_path, int(config["lmdb"]["map_size"])) as store:
        store.write_samples(_stream_samples(results, records, sample_meta))
        store.write_meta(
            {
                "layout": layout,
                "crop_rule": crop_rule(params),
                "config_sha256": _sha256(config_path),
                "depth_registration": depth_registration,
                "samples": sample_meta,
            }
        )

    counts = {status: 0 for status in ("aceito", "recusado", "erro")}
    for record in records:
        counts[record["status"]] += 1
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "mp_model_sha256": _sha256(params["mp_model"]),
        "versions": _versions(),
        "lmdb_path": str(lmdb_path),
        "layout": layout,
        "crop_rule": crop_rule(params),
        "depth_registration": depth_registration,
        "counts": {**counts, "amostras": len(sample_meta)},
        "payload_bytes": len(sample_meta) * record_bytes(size),
        "clips": records,
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=1)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(text, encoding="utf-8")
    (lmdb_path / "manifest.json").write_text(text, encoding="utf-8")
    print(f"Ingestão concluída: {manifest['counts']} em {lmdb_path}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Grava o LMDB da base UFPR e o manifesto.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lmdb-path", type=Path, help="sobrepõe paths.lmdb_path")
    parser.add_argument("--manifest-path", type=Path, help="sobrepõe paths.manifest_path")
    parser.add_argument("--sessions", help="sessões separadas por vírgula, ex.: C1_1,C6_2")
    parser.add_argument("--clips", help="chaves de clipes separadas por vírgula, ex.: C6_1_10")
    parser.add_argument("--limit", type=int, help="só os N primeiros clipes, para smoke test")
    parser.add_argument("--workers", type=int, help="sobrepõe ingest.num_workers")
    parser.add_argument("--overwrite", action="store_true", help="apaga o LMDB anterior")
    args = parser.parse_args(argv)
    build(
        args.config,
        lmdb_path=args.lmdb_path,
        manifest_path=args.manifest_path,
        sessions=args.sessions.split(",") if args.sessions else None,
        limit=args.limit,
        overwrite=args.overwrite,
        workers=args.workers,
        clip_keys=args.clips.split(",") if args.clips else None,
    )


if __name__ == "__main__":
    main()
