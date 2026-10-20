"""Sanidade das pausas, do recorte com margem e do layout do LMDB."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from research.config import load_config
from research.ingestion.build_dataset import (
    _prepare_lmdb_dir,
    clip_identity,
    list_clips,
    process_clip,
    sample_key,
)
from research.ingestion.crop import (
    context_box,
    crop_sample,
    crop_square,
    hand_box,
    min_box_px,
    shift_depth,
    shift_square_inside,
)
from research.ingestion.errors import SampleRejected
from research.ingestion.frame_clock import clip_pauses
from research.ingestion.hand_detector import hand_square, pick_hand
from research.ingestion.hand_track import find_pauses
from research.ingestion.lmdb_engine import (
    LMDBStore,
    SignEdgeDataset,
    pack_sample,
    record_bytes,
    unpack_sample,
)
from research.ingestion.registration import alignment_map, hand_silhouette, session_shift

ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = ROOT / "configs" / "data_pipeline.yaml"


def test_crop_rule_comes_from_config() -> None:
    config = load_config(DATA_CONFIG)
    assert config["crop"]["hand_scale"] == 1.4
    assert min_box_px(480, config["crop"]["min_box_frac"]) == 112
    assert config["crop"]["context_scale"] == 1.75
    assert config["crop"]["crop_size"] == 288
    clock = config["clock"]
    assert 0 < clock["hold_fraction"] < clock["motion_fraction"] < 1
    assert clock["frames_per_pause"] == 8
    assert clock["frame_stride"] == 2
    assert config["ingest"]["openni_teardown"] in {"close", "exit"}


def _hand(x0: float, y0: float, size: float, depth_mm: float) -> dict:
    corners = np.array([[x0, y0], [x0 + size, y0], [x0, y0 + size], [x0 + size, y0 + size]])
    return {"xy": corners, "score": 0.9, "label": "Left", "depth_mm": depth_mm}


def test_hand_square_grows_a_small_fist_to_the_floor() -> None:
    small = _hand(100, 100, 40, 800)["xy"]
    assert hand_square(small, scale=1.4, min_side=112) == (64, 64, 176, 176)
    large = _hand(100, 100, 100, 800)["xy"]
    x_min, y_min, x_max, y_max = hand_square(large, scale=1.4, min_side=112)
    assert x_max - x_min == y_max - y_min == 140
    assert (x_min + x_max) / 2 == pytest.approx(150, abs=1)


def test_pick_hand_keeps_the_hand_closer_to_the_camera() -> None:
    hands = [_hand(0, 0, 50, 950), _hand(200, 200, 50, 760)]
    assert pick_hand(hands, None) == 1
    assert pick_hand([], None) is None


def test_box_floor_scales_with_the_frame_height() -> None:
    assert min_box_px(480, 112 / 480) == 112
    assert min_box_px(1080, 112 / 480) == 252
    fist = np.array([[0.5, 0.5], [0.55, 0.55]])
    small = hand_box(fist * [640, 480], 480, 1.4, 112 / 480)
    large = hand_box(fist * [1280, 960], 960, 1.4, 112 / 480)
    assert small[2] - small[0] == 112
    assert large[2] - large[0] == 224


def test_context_box_keeps_the_center() -> None:
    assert context_box((100, 100, 200, 200), 1.75) == (62, 62, 237, 237)
    with pytest.raises(ValueError, match="context_scale"):
        context_box((0, 0, 10, 10), 0.5)


def _scene(height: int = 480, width: int = 640):
    rng = np.random.default_rng(0)
    color = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), 1400, dtype=np.uint16)
    depth[200:300, 300:400] = 700
    depth[:, :20] = 0
    return color, depth


def test_crop_sample_keeps_rgb_depth_and_points_aligned() -> None:
    color, depth = _scene()
    points = np.array([[320.0, 220.0], [380.0, 280.0], [350.0, 250.0]])
    box = hand_box(points, 480, 1.4, 112 / 480)
    sample = crop_sample(color, depth, points, box, context_scale=1.75, size=288)
    assert sample["rgb"].shape == (3, 288, 288) and sample["rgb"].dtype == np.uint8
    assert sample["depth"].shape == (288, 288) and sample["depth"].dtype == np.uint16
    assert sample["landmarks"].shape == (3, 2) and sample["landmarks"].dtype == np.float32
    assert set(np.unique(sample["depth"])) <= {0, 700, 1400}
    assert sample["landmarks"][2] == pytest.approx([144, 144], abs=1)
    x0, y0, x1, y1 = sample["hand_box"]
    assert x1 - x0 == pytest.approx(288 / 1.75, abs=1)
    cx, cy = sample["landmarks"][2].round().astype(int)
    assert sample["depth"][cy, cx] == 700


def test_crop_sample_points_follow_the_shift_at_the_border() -> None:
    color, depth = _scene()
    points = np.array([[5.0, 10.0], [60.0, 70.0]])
    box = hand_box(points, 480, 1.4, 112 / 480)
    sample = crop_sample(color, depth, points, box, context_scale=1.75, size=196)
    assert sample["context_box"][0] == 0 and sample["context_box"][1] == 0
    side = sample["context_box"][2]
    px, py = (points[1] * 196 / side).round().astype(int)
    assert sample["landmarks"][1] == pytest.approx(points[1] * 196 / side, abs=1e-3)
    assert side == 196
    assert sample["rgb"][:, py, px].tolist() == color[int(points[1][1]), int(points[1][0])].tolist()


def test_shift_depth_moves_the_content_and_fills_with_zero() -> None:
    depth = np.zeros((10, 10), dtype=np.uint16)
    depth[2, 3] = 800
    moved = shift_depth(depth, dx=4, dy=-2)
    assert moved[0, 7] == 800 and moved.sum() == 800
    assert shift_depth(depth, 0, 0) is depth


def test_alignment_map_finds_the_shift_that_puts_depth_on_the_hand() -> None:
    angles = np.linspace(0, 2 * np.pi, 21, endpoint=False)
    points = np.stack([320 + 30 * np.cos(angles), 240 + 30 * np.sin(angles)], axis=1)
    depth = np.full((480, 640), 2000, dtype=np.uint16)
    sil = hand_silhouette(points, (480, 640))
    # A profundidade da mão está 10 px à esquerda e 6 px abaixo da cor.
    depth[np.roll(np.roll(sil, -10, axis=1), 6, axis=0)] = 800
    box = hand_box(points, 480, 1.4, 112 / 480)
    scores = alignment_map(depth, points, box, max_shift=20, step=2)
    i, j = np.unravel_index(int(np.argmax(scores)), scores.shape)
    assert (-20 + 2 * j, -20 + 2 * i) == (10, -6)
    shift = session_shift(
        [{"scores": scores, "n_frames": 1, "per_frame": [(i, j)]}], max_shift=20, step=2
    )
    assert (shift["dx"], shift["dy"]) == (10, -6)
    assert shift["iou_after"] > shift["iou_before"]


def test_shift_square_lands_inside_without_changing_the_side() -> None:
    shifted = shift_square_inside((-10, 20, 30, 60), width=100, height=50)
    assert shifted == (0, 10, 40, 50)
    assert shift_square_inside((10, 10, 40, 40), width=100, height=50) == (10, 10, 40, 40)


def test_crop_square_moves_into_the_frame_instead_of_padding_black() -> None:
    image = np.zeros((6, 4, 3), dtype=np.uint8)
    image[:4] = (255, 255, 255)
    cropped = crop_square(image, (0, 4, 4, 8))
    assert cropped.shape == (4, 4, 3)
    assert cropped[0, 0, 0] == 255


def test_crop_square_pads_color_and_depth_with_the_border() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:] = (10, 20, 30)
    cropped = crop_square(image, (-2, -1, 6, 7))
    assert cropped.shape == (8, 8, 3)
    assert cropped[0, 0].tolist() == [10, 20, 30]
    depth = np.full((4, 4), 900, dtype=np.uint16)
    assert crop_square(depth, (-2, -1, 6, 7)).tolist() == np.full((8, 8), 900).tolist()


def _dip_track() -> np.ndarray:
    """Mão levantada (y≈150), descida até y=450 e volta, mão levantada de novo."""
    rng = np.random.default_rng(1)
    up = 150 + rng.normal(0, 0.3, 20)
    down = np.linspace(150, 450, 15)
    back = np.linspace(450, 150, 15)
    up_again = 150 + rng.normal(0, 0.3, 20)
    ys = np.concatenate([up, down, back, up_again])
    return np.stack([np.full(len(ys), 320.0), ys], axis=1)


def test_find_pauses_are_the_raised_hand_on_both_sides_of_the_dip() -> None:
    track = _dip_track()
    pauses = find_pauses(track, window=5, frames_per_pause=8, stride=2)
    assert [pause["pause"] for pause in pauses] == [0, 1]
    first, last = pauses
    assert max(first["frames"]) < 20 <= 50 <= min(last["frames"])
    for pause in pauses:
        assert 1 <= len(pause["frames"]) <= 8
        assert all(track[frame, 1] < 160 for frame in pause["frames"])
        assert all((frame - pause["center"]) % 2 == 0 for frame in pause["frames"])
        assert all(5 <= frame < len(track) - 5 for frame in pause["frames"])


def test_find_pauses_skips_frames_without_a_hand() -> None:
    track = _dip_track()
    track[8:12] = np.nan
    for pause in find_pauses(track, window=5):
        assert not any(8 <= frame < 12 for frame in pause["frames"])


def test_find_pauses_without_motion_is_a_single_pause() -> None:
    track = np.stack([np.full(30, 300.0), np.full(30, 150.0)], axis=1)
    pauses = find_pauses(track, window=5)
    assert len(pauses) == 1 and pauses[0]["pause"] == 0


def _landmarks_on(track: np.ndarray) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, 21, endpoint=False)
    shape = 40 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return track[:, None, :] + shape[None]


def test_pause_stops_where_the_fingers_are_still_closing() -> None:
    track = _dip_track()
    landmarks = _landmarks_on(track)
    # Média parada, mas metade dos pontos (os "dedos") ainda encolhendo até o frame 12.
    for frame in range(13):
        squeeze = 1 - 0.06 * (13 - frame)
        landmarks[frame, ::2] = track[frame] + (landmarks[frame, ::2] - track[frame]) * squeeze
        landmarks[frame, 1::2] = track[frame] + (landmarks[frame, 1::2] - track[frame]) / squeeze
    loose = find_pauses(track, window=5)[0]
    strict = find_pauses(track, window=5, landmarks=landmarks, max_shape_change=0.25)[0]
    assert min(loose["frames"]) < 9
    assert min(strict["frames"]) >= 9


def test_clip_pauses_rejects_a_short_track() -> None:
    landmarks = np.full((20, 21, 2), np.nan)
    landmarks[[3, 9]] = 10.0
    with pytest.raises(SampleRejected, match="mão vista"):
        clip_pauses(landmarks, name="C1_1_0.oni", window=1)


def test_clip_identity_normalizes_the_two_digit_take_of_c6() -> None:
    assert clip_identity("x/C6_1/C6_01_10.oni")["key"] == "C6_1_10"
    assert clip_identity("C1_2_0.oni") == {
        "key": "C1_2_0",
        "actor": "C1",
        "take": 2,
        "cls": 0,
        "file": "C1_2_0",
    }
    assert sample_key("C6_1_10", 1, 7) == "C6_1_10_p1_f007"
    with pytest.raises(ValueError, match="padrão"):
        clip_identity("leia-me.oni")


def test_list_clips_keeps_empty_oni_for_the_manifest(tmp_path: Path) -> None:
    session = tmp_path / "C1_1"
    session.mkdir()
    for name, payload in (("C1_1_10", b"oni"), ("C1_1_6", b""), ("C1_1_0", b"oni")):
        (session / f"{name}.oni").write_bytes(payload)
    (tmp_path / "C2_1").mkdir()
    (tmp_path / "C2_1" / "C2_1_0.oni").write_bytes(b"oni")
    assert [path.stem for path in list_clips(tmp_path, ["C1_1"])] == ["C1_1_0", "C1_1_6", "C1_1_10"]
    with pytest.raises(FileNotFoundError, match="C9_9"):
        list_clips(tmp_path, ["C9_9"])


def test_process_clip_records_the_reason_instead_of_raising(tmp_path: Path) -> None:
    oni_path = tmp_path / "C3_1_3.oni"
    oni_path.write_bytes(b"")
    record, samples = process_clip((str(oni_path), {}))
    assert samples == []
    assert record["key"] == "C3_1_3"
    assert record["status"] == "recusado"
    assert "vazio" in record["reason"]


def test_lmdb_dir_is_never_mixed_with_an_old_build(tmp_path: Path) -> None:
    target = tmp_path / "db"
    target.mkdir()
    (target / "data.mdb").write_bytes(b"x")
    with pytest.raises(FileExistsError, match="--overwrite"):
        _prepare_lmdb_dir(target, overwrite=False)
    (target / "notas.txt").write_text("", encoding="utf-8")
    with pytest.raises(FileExistsError, match="notas.txt"):
        _prepare_lmdb_dir(target, overwrite=True)
    (target / "notas.txt").unlink()
    _prepare_lmdb_dir(target, overwrite=True)
    assert not target.exists()


def _record(size: int = 8):
    rgb = np.arange(3 * size * size, dtype=np.uint64).astype(np.uint8).reshape(3, size, size)
    depth = np.arange(size * size, dtype=np.uint16).reshape(size, size) * 50
    points = np.linspace(0, size, 42, dtype=np.float32).reshape(21, 2)
    return rgb, depth, points


def test_pack_and_unpack_are_inverse() -> None:
    rgb, depth, points = _record()
    raw = pack_sample(rgb, depth, points)
    assert len(raw) == record_bytes(8)
    back = unpack_sample(raw, 8)
    for original, restored in zip((rgb, depth, points), back):
        assert np.array_equal(original, restored)
    with pytest.raises(ValueError, match="uint16"):
        pack_sample(rgb, depth.astype(np.int32), points)
    with pytest.raises(ValueError, match="layout"):
        unpack_sample(raw[:-1], 8)


def test_dataset_returns_float_rgb_depth_in_mm_and_the_label() -> None:
    torch = pytest.importorskip("torch")
    rgb, depth, points = _record()
    raw = pack_sample(np.full_like(rgb, 255), depth, points)

    class _Store:
        def read_sample(self, key: bytes) -> bytes:
            return raw

    meta = {"layout": {"size": 8}, "samples": {"C1_1_5_p0_f010": {"cls": 5}}}
    item = SignEdgeDataset(_Store(), ["C1_1_5_p0_f010"], meta)[0]
    assert item["rgb"].shape == (3, 8, 8) and torch.all(item["rgb"] == 1.0)
    assert item["depth"].dtype == torch.float32 and item["depth"][0, 1] == 50
    assert item["landmarks"].shape == (21, 2)
    assert int(item["label"]) == 5


def test_store_closes_cleanly_before_open() -> None:
    store = LMDBStore(path=ROOT / "data" / "lmdb" / "sanity", map_size=1024)
    store.__exit__(None, None, None)
    assert store._env is None


def test_lmdb_roundtrip_closes_environment(tmp_path: Path) -> None:
    """O ambiente aberto em write_samples precisa estar fechado ao sair do contexto."""
    pytest.importorskip("lmdb")
    raw = pack_sample(*_record())
    with LMDBStore(tmp_path / "db", map_size=64 * len(raw)) as store:
        assert store.write_samples([(b"C1_1_0_p0_f010", raw)]) == 1
        store.write_meta({"layout": {"size": 8}})
    assert store._env is None
    with LMDBStore(tmp_path / "db", map_size=64 * len(raw), readonly=True) as store:
        assert store.read_sample(b"C1_1_0_p0_f010") == raw
        assert store.read_meta() == {"layout": {"size": 8}}
        with pytest.raises(KeyError):
            store.read_sample(b"C9_9_9_p0_f001")
