"""Perdas, profundidade relativa, métricas, modelos e um treino de ponta a ponta minúsculo."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from research.config import load_config  # noqa: E402
from research.ingestion.lmdb_engine import LMDBStore, pack_sample  # noqa: E402
from research.networks.student import build_student, normalize_landmarks  # noqa: E402
from research.networks.teacher import build_teacher  # noqa: E402
from research.training.inputs import relative_depth  # noqa: E402
from research.training.losses import kd_kl, mixed_cross_entropy  # noqa: E402
from research.training.metrics import expected_calibration_error, macro_f1, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = ROOT / "configs" / "distillation.yaml"


def test_kd_is_zero_when_student_matches_teacher_and_keeps_scale() -> None:
    logits = torch.randn(8, 61)
    assert kd_kl(logits, logits, 1.0).abs() < 1e-6
    other = torch.randn(8, 61)
    assert kd_kl(other, logits, 4.0) > 0


def test_mixed_cross_entropy_weights_both_labels() -> None:
    logits = torch.randn(4, 5)
    labels = torch.tensor([0, 1, 2, 3])
    perm = torch.tensor([1, 0, 3, 2])
    a = mixed_cross_entropy(logits, labels)
    b = mixed_cross_entropy(logits, labels[perm])
    mixed = mixed_cross_entropy(logits, labels, lam=0.3, perm=perm)
    assert torch.allclose(mixed, 0.3 * a + 0.7 * b)


def test_relative_depth_is_zero_on_the_hand_and_far_without_reading() -> None:
    depth = torch.full((2, 1, 32, 32), 2000.0)
    depth[:, :, 8:24, 8:24] = 800
    depth[1, :, :4, :4] = 0
    points = torch.full((2, 21, 2), 16.0) + torch.linspace(-6, 6, 21)[None, :, None]
    rel = relative_depth(depth, points, range_mm=300)
    assert torch.allclose(rel[:, :, 8:24, 8:24], torch.zeros(1))
    assert (rel[:, :, 0, 31] == 1).all() and (rel[1, :, :4, :4] == 1).all()


def test_landmark_normalization_ignores_position_and_size() -> None:
    points = torch.rand(3, 21, 2) * 50
    moved = points * 2.5 + torch.tensor([30.0, -12.0])
    assert torch.allclose(normalize_landmarks(points), normalize_landmarks(moved), atol=1e-5)


def test_metrics_pool_frames_by_pause() -> None:
    logits = np.zeros((4, 3))
    logits[0, 0], logits[1, 1], logits[2, 2], logits[3, 2] = 3.0, 1.0, 2.0, 2.0
    labels = np.array([0, 0, 2, 2])
    pauses = np.array([0, 0, 1, 1])
    out = summarize(logits, labels, pauses)
    assert out["frame_acc"] == 0.75 and out["pause_acc"] == 1.0 and out["n_pauses"] == 2
    assert macro_f1(np.array([0, 1]), np.array([0, 1]), 3) == 1.0
    sure = np.eye(3)[[0, 1, 2]]
    assert expected_calibration_error(sure, np.array([0, 1, 2])) == 0.0


def test_models_take_the_inputs_the_app_and_the_lmdb_provide() -> None:
    cfg = load_config(TRAIN_CONFIG)["model"]
    rgb, depth = torch.rand(2, 3, 64, 64), torch.rand(2, 1, 64, 64)
    points = torch.rand(2, 21, 2) * 64
    for arch in ("convnext_tiny", "resnet34"):
        teacher = build_teacher(arch, 61, depth=True, pretrained=False).eval()
        assert teacher(rgb=rgb, depth=depth, landmarks=points).shape == (2, 61)
    student = build_student(cfg["student"], 61, pretrained=False).eval()
    assert student(rgb=rgb, landmarks=points, depth=depth).shape == (2, 61)


def _tiny_lmdb(root: Path, size: int = 96) -> Path:
    """Dois clipes por classe e ator, 2 pausas de 2 frames; classes 0..2 com cores próprias."""
    path = root / "lmdb"
    samples, records = {}, []
    rng = np.random.default_rng(0)
    for actor in ("C1", "C2", "C3", "C4", "C5", "C6"):
        for cls in range(3):
            clip = f"{actor}_1_{cls}"
            for pause in (0, 1):
                for frame in (1, 3):
                    rgb = np.full((3, size, size), 40 + 80 * cls, np.uint8)
                    depth = np.full((size, size), 2000, np.uint16)
                    depth[30:66, 30:66] = 800
                    points = (48 + rng.uniform(-14, 14, (21, 2))).astype(np.float32)
                    key = f"{clip}_p{pause}_f{frame:03d}"
                    records.append((key.encode(), pack_sample(rgb, depth, points)))
                    samples[key] = {"clip": clip, "actor": actor, "take": 1, "cls": cls,
                                    "pause": pause, "frame": frame}  # fmt: skip
    meta = {
        "layout": {"size": size},
        "crop_rule": {"context_scale": 1.75},
        "samples": samples,
    }
    with LMDBStore(path, 1 << 26) as store:
        store.write_samples(records)
        store.write_meta(meta)
    return path


def _tiny_configs(tmp: Path) -> Path:
    lmdb = _tiny_lmdb(tmp)
    data = load_config(ROOT / "configs" / "data_pipeline.yaml")
    data["paths"]["lmdb_path"] = str(lmdb)
    data["lmdb"]["map_size"] = 1 << 26
    (tmp / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    config = copy.deepcopy(load_config(TRAIN_CONFIG))
    config["model"]["pretrained"] = False
    config["model"]["num_classes"] = 3
    config["model"]["teacher"]["arch"] = "resnet34"
    config["view"]["input_size"] = 64
    config["paths"]["data_config"] = str(tmp / "data.yaml")
    config["paths"]["runs_dir"] = str(tmp / "runs")
    config["augment"]["background"]["photo_dir"] = str(tmp / "sem_fotos")
    config["run"].update(eval_every=2, eval_batch_size=8)
    config["vocabularies"] = {"duas": {"codes": [2, 0], "other": True, "other_cap_factor": 1}}
    for stage in config["stages"].values():
        stage.update(batch_size=4, warmup_steps=1)
    path = tmp / "train.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_every_stage_trains_and_distillation_refuses_a_foreign_teacher(tmp_path: Path) -> None:
    from research.train import run

    config = _tiny_configs(tmp_path)
    for stage in ("landmarks", "student_ce"):
        out = run(config, stage, fold=0, steps=3, name=stage, device="cpu")
        result = json.loads((out / "result.json").read_text())
        assert set(result["metrics"]) == {"val", "val_espelhado"}
        assert (out / "last.pt").is_file() and not (out / "best.pt").exists()
        assert (out / "pred_val.npz").is_file()
    teacher = run(config, "teacher", fold=0, steps=3, name="teacher", device="cpu")
    student = run(config, "student_kd", 0, teacher_run=teacher, steps=3, name="kd", device="cpu")
    record = json.loads((student / "run.json").read_text())
    assert record["provenance"]["teacher_run"] == str(teacher)
    assert record["val_actor"] == "C1" and record["train_actors"] == ["C2", "C3", "C4", "C5"]
    with pytest.raises(ValueError, match="não serve"):
        run(config, "student_kd", 1, teacher_run=teacher, steps=1, name="x", device="cpu")
    with pytest.raises(ValueError, match="não serve"):
        run(config, "student_kd", 0, teacher_run=teacher, steps=1, name="y", device="cpu",
            vocab="duas")  # fmt: skip

    final = run(config, "landmarks", fold=None, steps=2, name="todos", device="cpu")
    result = json.loads((final / "result.json").read_text())
    assert set(result["metrics"]) == {"ood", "ood_espelhado"}
    curve = [json.loads(line) for line in (final / "curve.jsonl").read_text().splitlines()]
    assert all(row["eval"] == {} for row in curve)
    record = json.loads((final / "run.json").read_text())
    assert record["train_actors"] == ["C1", "C2", "C3", "C4", "C5"]


def test_vocabulary_measures_closed_and_open() -> None:
    from research.training.aggregate import closed_accuracy, open_report

    full = {"name": "full", "codes": list(range(61)), "other": False}
    logits = np.zeros((4, 61), np.float32)
    logits[0, 1], logits[1, 9], logits[1, 2], logits[2, 1], logits[3, 7] = 1, 3, 2, 1, 1
    pauses = {"logits": logits, "codes": np.array([1, 2, 5, 7]), "vocab": full}
    assert closed_accuracy(pauses, [1, 2]) == 1.0
    report = open_report(pauses, [1, 2])
    assert report["vocab_recall"] == 0.5 and report["false_accept"] == 0.5
    assert report["other_rejected"] == 0.5 and report["precision"] == 0.5
    demo = {"name": "d", "codes": [1, 2], "other": True}
    pauses = {"logits": np.eye(3, dtype=np.float32)[[0, 2, 2]], "codes": np.array([1, 2, 30]),
              "vocab": demo}  # fmt: skip
    report = open_report(pauses, [1, 2])
    assert report["vocab_recall"] == 0.5 and report["false_accept"] == 0.0


def test_demo_vocabulary_relabels_and_caps_the_other_class(tmp_path: Path) -> None:
    from research.train import run

    config = _tiny_configs(tmp_path)
    out = run(config, "landmarks", fold=2, steps=2, name="demo", device="cpu", vocab="duas")
    record = json.loads((out / "run.json").read_text())
    assert record["vocab"]["names"] == ["HC02", "HC00", "outra"]
    assert record["vocab"]["num_classes"] == 3
    assert record["other_cap"] == {"2": 8}
    assert (out / "labels.txt").read_text().split() == ["HC02", "HC00", "outra"]
    pred = np.load(out / "pred_val.npz")
    assert pred["logits"].shape[1] == 3
    expected = {2: 0, 0: 1, 1: 2}
    assert all(expected[int(c)] == int(lab) for c, lab in zip(pred["codes"], pred["labels"]))
