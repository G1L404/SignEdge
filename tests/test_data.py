"""Divisão por ator, um frame por pausa e augmentation que mantém os pontos sobre a mão."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from research.config import load_config
from research.data.splits import PauseSampler, all_actors_split, group_by_pause, loso_folds

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = ROOT / "configs" / "distillation.yaml"


def _samples() -> dict[str, dict]:
    samples = {}
    for actor in ("C1", "C2", "C3", "C4", "C5", "C6"):
        for cls in range(3):
            for pause in (0, 1):
                for frame in range(1, 4 + cls):
                    key = f"{actor}_1_{cls}_p{pause}_f{frame:03d}"
                    samples[key] = {"actor": actor, "clip": f"{actor}_1_{cls}", "pause": pause}
    return samples


def test_loso_folds_train_on_four_validate_on_one_and_never_touch_c6() -> None:
    samples = _samples()
    split = load_config(TRAIN_CONFIG)["split"]
    folds = loso_folds(samples, split["actors"], split["ood"])
    assert [fold["val_actor"] for fold in folds] == ["C1", "C2", "C3", "C4", "C5"]
    for fold in folds:
        assert set(fold) == {"val_actor", "train_actors", "train", "val"}
        roles = {name: {samples[key]["actor"] for key in fold[name]} for name in ("train", "val")}
        assert roles["val"] == {fold["val_actor"]}
        assert roles["train"] == set(fold["train_actors"]) and len(roles["train"]) == 4
        assert fold["val_actor"] not in roles["train"] and "C6" not in roles["train"]


def test_all_actors_split_trains_on_the_folds_and_measures_only_c6() -> None:
    samples = _samples()
    split = load_config(TRAIN_CONFIG)["split"]
    final = all_actors_split(samples, split["actors"], split["ood"])
    assert {samples[key]["actor"] for key in final["train"]} == set(split["actors"])
    assert {samples[key]["actor"] for key in final["ood"]} == {"C6"}


def test_loso_folds_refuse_an_actor_without_a_role() -> None:
    with pytest.raises(ValueError, match="sem papel"):
        loso_folds(_samples(), ["C1", "C2", "C3", "C4"], ["C6"])


def test_pause_sampler_caps_a_label_with_too_many_pauses() -> None:
    samples = _samples()
    keys = sorted(samples)
    labels = [0 if samples[key]["clip"].endswith("_0") else 1 for key in keys]
    sampler = PauseSampler(keys, samples, seed=0, labels=labels, caps={1: 5})
    picks = list(sampler)
    assert len(picks) == len(sampler) == 12 + 5
    assert sum(labels[i] == 1 for i in picks) == 5
    sampler.set_epoch(1)
    assert {i for i in sampler if labels[i] == 1} != {i for i in picks if labels[i] == 1}


def test_vocab_maps_the_listed_codes_in_order_and_the_rest_to_other() -> None:
    from research.data.vocab import resolve_vocab

    config = {"vocabularies": {"v": {"codes": [5, 0], "other": True, "other_cap_factor": 2}}}
    vocab = resolve_vocab(config, "v")
    assert vocab["names"] == ["HC05", "HC00", "outra"]
    assert vocab["label_of"][5] == 0 and vocab["label_of"][0] == 1 and vocab["label_of"][7] == 2
    closed = resolve_vocab({"vocabularies": {"v": {"codes": [5]}}}, "v")
    assert closed["label_of"][5] == 0 and closed["label_of"][0] is None
    assert resolve_vocab(config, None)["num_classes"] == 61


def test_pause_sampler_draws_one_frame_per_pause_and_changes_by_epoch() -> None:
    samples = _samples()
    keys = sorted(samples)
    sampler = PauseSampler(keys, samples, seed=3)
    groups = group_by_pause(keys, samples)
    first = list(sampler)
    assert len(first) == len(sampler) == len(groups) == 6 * 3 * 2
    owner = {index: n for n, group in enumerate(groups) for index in group}
    assert sorted(owner[index] for index in first) == list(range(len(groups)))
    assert list(sampler) == first
    sampler.set_epoch(1)
    assert list(sampler) != first


torch = pytest.importorskip("torch")
from research.data.augment import (  # noqa: E402
    EvalView,
    TrainAugment,
    landmarks_on_mask,
    near_surface,
    soft_hand_mask,
)

SIZE = 288
HAND_SIDE = SIZE / 1.75


def _cfg(**overrides) -> dict:
    cfg = copy.deepcopy(load_config(TRAIN_CONFIG)["augment"])
    cfg["background"]["p"] = 0.0
    cfg["occlusion"]["p"] = 0.0
    photo = cfg["photometric"]
    for key in ("white_balance", "brightness", "contrast", "saturation", "hue"):
        photo[key] = 0.0
    photo.update(gamma=[1.0, 1.0], blur_p=0, motion_p=0, noise_p=0, downscale_p=0)
    for section, values in overrides.items():
        cfg[section].update(values)
    return cfg


def _dot_batch(batch: int = 6, seed: int = 0):
    """Fundo escuro com um quadrado claro de 5 px sob cada ponto; mão perto na profundidade."""
    rng = np.random.default_rng(seed)
    rgb = np.zeros((batch, 3, SIZE, SIZE), np.uint8)
    depth = np.full((batch, SIZE, SIZE), 2000, np.uint16)
    points = np.empty((batch, 21, 2), np.float32)
    for b in range(batch):
        pts = SIZE / 2 + rng.uniform(-55, 55, size=(21, 2))
        points[b] = pts
        for x, y in pts.astype(int):
            rgb[b, :, y - 2 : y + 3, x - 2 : x + 3] = 255
        x0, y0 = pts.min(0).astype(int) - 6
        x1, y1 = pts.max(0).astype(int) + 6
        depth[b, y0:y1, x0:x1] = 800
    return (
        torch.from_numpy(rgb),
        torch.from_numpy(depth.astype(np.float32)),
        torch.from_numpy(points),
    )


def _bright_under_points(rgb: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    index = points.floor().long().clamp(0, rgb.shape[-1] - 1)
    rows = torch.arange(rgb.shape[0])[:, None]
    return rgb[rows, 0, index[..., 1], index[..., 0]]


def test_eval_view_at_the_recorded_context_is_the_recorded_crop() -> None:
    rgb, depth, points = _dot_batch()
    view = EvalView(HAND_SIDE, 1.75, SIZE)
    centered = points - (points.amin(1, keepdim=True) + points.amax(1, keepdim=True)) / 2 + SIZE / 2
    out = view(rgb, depth, centered)
    assert torch.allclose(out["landmarks"], centered, atol=1e-3)
    flipped = view(rgb, depth, centered, flip=True)
    assert torch.allclose(flipped["landmarks"][..., 0], SIZE - centered[..., 0], atol=1e-3)


def test_rotated_flipped_views_keep_every_landmark_on_its_dot() -> None:
    rgb, depth, points = _dot_batch(batch=16)
    cfg = _cfg(geometry={"flip_p": 0.5, "max_rotation_deg": 15})
    augment = TrainAugment(cfg, HAND_SIDE, 1.5, 224, seed=1)
    out = augment(rgb, depth, points)
    assert out["flip"].any() and not out["flip"].all()
    assert (out["landmarks"] >= 0).all() and (out["landmarks"] <= 224).all()
    assert (_bright_under_points(out["rgb"], out["landmarks"]) > 0.5).all()
    near = out["depth"][:, 0] < 1000
    index = out["landmarks"].floor().long().clamp(0, 223)
    rows = torch.arange(16)[:, None]
    assert near[rows, index[..., 1], index[..., 0]].all()


def test_background_swap_changes_the_backdrop_and_spares_the_hand() -> None:
    rgb, depth, points = _dot_batch(batch=8)
    rgb[:] = 255
    hand = depth < 1000
    rgb[:, 1:][hand[:, None].expand(-1, 2, -1, -1)] = 0
    cfg = _cfg(background={"p": 1.0})
    out = TrainAugment(cfg, HAND_SIDE, 1.5, 224, seed=2)(rgb, depth, points)
    inner = out["mask"][:, 0] > 0.999
    outer = out["mask"][:, 0] < 1e-3
    red = (out["rgb"][:, 0] > 0.99) & (out["rgb"][:, 1] < 0.01)
    assert red[inner].float().mean() > 0.99
    white = (out["rgb"] > 0.99).all(1)
    assert white[outer].float().mean() < 0.5


def test_hand_mask_covers_the_landmarks_and_not_the_wall() -> None:
    _, depth, points = _dot_batch(batch=4)
    near = near_surface(depth.unsqueeze(1), band_mm=120)
    mask = soft_hand_mask(near, close_px=3, edge_px=2)
    assert (landmarks_on_mask(mask, points) == 1).all()
    assert (mask[depth.unsqueeze(1) >= 2000] < 0.5).all()


def test_occlusion_never_covers_more_than_the_allowed_landmarks() -> None:
    rgb, depth, points = _dot_batch(batch=32)
    cfg = _cfg(occlusion={"p": 1.0, "max_boxes": 1, "area": [0.05, 0.15], "max_landmarks": 3})
    augment = TrainAugment(cfg, HAND_SIDE, 1.5, 224, seed=4)
    canvas = torch.zeros(32, 3, 224, 224)
    view = EvalView(HAND_SIDE, 1.5, 224)(rgb, depth, points)
    occluded = augment._occlude(canvas.clone(), view["landmarks"])
    assert (occluded != canvas).any()
    index = view["landmarks"].floor().long().clamp(0, 223)
    rows = torch.arange(32)[:, None]
    hit = (occluded != canvas).any(1)[rows, index[..., 1], index[..., 0]].sum(1)
    assert (hit <= 3).all()


def test_photometric_stays_in_range_and_leaves_depth_alone() -> None:
    rgb, depth, points = _dot_batch(batch=8)
    cfg = copy.deepcopy(load_config(TRAIN_CONFIG)["augment"])
    out = TrainAugment(cfg, HAND_SIDE, 1.5, 224, seed=5)(rgb, depth, points)
    assert out["rgb"].min() >= 0 and out["rgb"].max() <= 1
    assert set(torch.unique(out["depth"]).tolist()) <= {0.0, 800.0, 2000.0}
