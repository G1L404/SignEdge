"""Augmentation em lote, na GPU, para RGB, profundidade, máscara e os 21 pontos juntos.

Cada transformação modela uma diferença entre a gravação (Kinect de 2011, parede
clara, recorte exato) e o app (câmera de celular, fundo qualquer, caixa do
MediaPipe com ruído, as duas mãos):

- geometria: a caixa do app varia em centro, escala e rotação; o espelhamento
  cobre a outra mão e câmeras que entregam o frame espelhado ou não;
- fundo: a superfície próxima da profundidade separa mão e antebraço do resto,
  que é trocado (o fundo da base acompanha a ordem de gravação das classes);
- fotometria e sensor: balanço de branco, exposição, gama, desfoque, ruído e
  perda de resolução;
- oclusão: retângulos que nunca cobrem mais que alguns dos 21 pontos.

A profundidade só recebe a geometria. Coordenadas são contínuas: o pixel i
cobre [i, i + 1), como os pontos do MediaPipe, e o espelho é x' = largura − x.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

N_LANDMARKS = 21


def as_float_batch(rgb: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """RGB em [0, 1] (B, 3, S, S) e profundidade em mm (B, 1, S, S), float32."""
    if rgb.dtype == torch.uint8:
        rgb = rgb.float().div_(255.0)
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    return rgb.float(), depth.float()


def hand_center(landmarks: torch.Tensor) -> torch.Tensor:
    """Meio da extensão dos 21 pontos, a mesma regra do recorte da ingestão."""
    return (landmarks.amin(dim=1) + landmarks.amax(dim=1)) / 2


def _rotation(theta: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)


def view_matrix(
    center: torch.Tensor,
    view_side: torch.Tensor,
    theta: torch.Tensor,
    flip: torch.Tensor,
    crop_size: int,
) -> torch.Tensor:
    """Matriz (B, 2, 3) do `affine_grid`: coordenada normalizada da saída → da entrada.

    A saída é o quadrado de lado `view_side` (pixels da entrada) centrado em
    `center`, girado de `theta` e espelhado em x onde `flip`.
    """
    sign = torch.where(flip, -1.0, 1.0).to(center)
    linear = _rotation(theta) * (view_side / crop_size)[:, None, None]
    linear = linear * torch.stack([sign, torch.ones_like(sign)], -1)[:, None, :]
    offset = 2 * center / crop_size - 1
    return torch.cat([linear, offset[:, :, None]], dim=2)


def map_landmarks(
    landmarks: torch.Tensor,
    center: torch.Tensor,
    view_side: torch.Tensor,
    theta: torch.Tensor,
    flip: torch.Tensor,
    out_size: int,
) -> torch.Tensor:
    """Pontos da entrada para pixels da saída, inverso exato de `view_matrix`."""
    local = (landmarks - center[:, None, :]) @ _rotation(theta)
    local[..., 0] = torch.where(flip[:, None], -local[..., 0], local[..., 0])
    return local * (out_size / view_side)[:, None, None] + out_size / 2


def warp(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    landmarks: torch.Tensor,
    center: torch.Tensor,
    view_side: torch.Tensor,
    theta: torch.Tensor,
    flip: torch.Tensor,
    out_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recorta a vista. RGB bilinear com reflexo na borda; profundidade sem interpolar."""
    batch, crop_size = rgb.shape[0], rgb.shape[-1]
    matrix = view_matrix(center, view_side, theta, flip, crop_size)
    grid = F.affine_grid(matrix, [batch, 1, out_size, out_size], align_corners=False)
    rgb_out = F.grid_sample(
        rgb, grid, mode="bilinear", padding_mode="reflection", align_corners=False
    )
    depth_out = F.grid_sample(
        depth, grid, mode="nearest", padding_mode="zeros", align_corners=False
    )
    points = map_landmarks(landmarks, center, view_side, theta, flip, out_size)
    return rgb_out.clamp_(0, 1), depth_out, points


def near_surface(depth: torch.Tensor, band_mm: float, max_mm: float = 2500) -> torch.Tensor:
    """Máscara (B, 1, H, W) da superfície mais próxima: do 2º percentil até + `band_mm`."""
    flat = depth.flatten(1)
    valid = (flat > 0) & (flat < max_mm)
    ordered = torch.where(valid, flat, torch.full_like(flat, float("inf"))).sort(dim=1).values
    count = valid.sum(dim=1)
    index = ((count - 1).clamp(min=0).float() * 0.02).long()[:, None]
    z_near = ordered.gather(1, index)
    near = valid & (flat <= z_near + band_mm) & (count[:, None] > 0)
    return near.view_as(depth).float()


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return F.max_pool2d(mask, 2 * radius + 1, stride=1, padding=radius) if radius else mask


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return 1 - _dilate(1 - mask, radius)


def soft_hand_mask(near: torch.Tensor, close_px: int, edge_px: int) -> torch.Tensor:
    """Fecha os buracos da leitura do Kinect, recua a borda e suaviza a transição."""
    closed = _erode(_dilate(near, close_px), close_px)
    inner = _erode(closed, edge_px)
    return F.avg_pool2d(inner, 2 * edge_px + 1, stride=1, padding=edge_px, count_include_pad=False)


def landmarks_on_mask(mask: torch.Tensor, points: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """Fração dos 21 pontos que caem na máscara (com folga de `radius` px)."""
    grown = _dilate((mask > 0.5).float(), radius)
    size = mask.shape[-1]
    index = points.floor().long().clamp_(0, size - 1)
    flat = grown.flatten(1)
    hits = flat.gather(1, index[..., 1] * size + index[..., 0])
    return hits.mean(dim=1)


def load_backgrounds(directory: str | Path, size: int) -> torch.Tensor | None:
    """Imagens de fundo (N, 3, size, size) uint8, cortadas no centro. None sem imagens."""
    import cv2

    directory = Path(directory)
    if not directory.is_dir():
        return None
    images = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        side = min(height, width)
        y0, x0 = (height - side) // 2, (width - side) // 2
        square = cv2.resize(image[y0 : y0 + side, x0 : x0 + side], (size, size), cv2.INTER_AREA)
        images.append(torch.from_numpy(cv2.cvtColor(square, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
    return torch.stack(images) if images else None


class TrainAugment:
    """Vista aumentada de um lote. `__call__` devolve rgb, depth, mask, landmarks e flip.

    `cfg` é a seção `augment` de configs/distillation.yaml. A vista sem
    augmentation (`EvalView`) é a caixa da mão × `view_context`, sem giro; o
    treino sorteia em torno dela, e a escala média fica na da avaliação.
    """

    def __init__(
        self,
        cfg: dict,
        hand_side_px: float,
        view_context: float,
        out_size: int,
        device: str | torch.device = "cpu",
        seed: int = 0,
        backgrounds: torch.Tensor | None = None,
    ) -> None:
        self.cfg = cfg
        self.hand_side_px = float(hand_side_px)
        self.view_context = float(view_context)
        self.out_size = int(out_size)
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.backgrounds = None if backgrounds is None else backgrounds.to(self.device)

    def _uniform(self, shape, low: float, high: float) -> torch.Tensor:
        return torch.rand(shape, generator=self.gen, device=self.device) * (high - low) + low

    def _normal(self, shape) -> torch.Tensor:
        return torch.randn(shape, generator=self.gen, device=self.device)

    def _chance(self, batch: int, p: float) -> torch.Tensor:
        return torch.rand(batch, generator=self.gen, device=self.device) < p

    def __call__(self, rgb, depth, landmarks) -> dict[str, torch.Tensor]:
        rgb, depth = as_float_batch(rgb.to(self.device), depth.to(self.device))
        landmarks = landmarks.to(self.device).float()
        geo = self._sample_geometry(landmarks)
        rgb, depth, points = warp(rgb, depth, landmarks, *geo, self.out_size)
        bg = self.cfg["background"]
        near = near_surface(depth, float(bg["band_mm"]))
        mask = soft_hand_mask(near, int(bg["close_px"]), int(bg["edge_px"]))
        rgb = self._swap_background(rgb, mask, points)
        rgb = self._photometric(rgb)
        rgb = self._occlude(rgb, points)
        return {"rgb": rgb, "depth": depth, "mask": mask, "landmarks": points, "flip": geo[3]}

    def landmarks_only(self, landmarks: torch.Tensor) -> dict[str, torch.Tensor]:
        """Só a geometria, aplicada aos pontos: a linha de base dos pontos não precisa da imagem."""
        landmarks = landmarks.to(self.device).float()
        center, side, theta, flip = self._sample_geometry(landmarks)
        points = map_landmarks(landmarks, center, side, theta, flip, self.out_size)
        return {"landmarks": points, "flip": flip}

    def _sample_geometry(self, landmarks: torch.Tensor):
        """Sorteia até `tries` vistas por amostra e fica com a primeira que contém os 21 pontos."""
        cfg = self.cfg["geometry"]
        batch, tries = landmarks.shape[0], int(cfg["tries"])
        base_side = self.hand_side_px * self.view_context
        shape = (batch, tries)
        low, high = (math.log(v) for v in cfg["scale_range"])
        scale = (self._normal(shape) * float(cfg["scale_sigma"])).clamp_(low, high).exp_()
        theta = self._uniform(shape, -1, 1) * math.radians(float(cfg["max_rotation_deg"]))
        jitter = (self._normal((*shape, 2)) * float(cfg["center_jitter"])).clamp_(-2.5, 2.5)
        center = hand_center(landmarks)[:, None, :] + jitter * self.hand_side_px
        flip = self._chance(batch, float(cfg["flip_p"]))
        side = base_side * scale
        margin = float(cfg["landmark_margin"]) * self.out_size
        points = map_landmarks(
            landmarks.repeat_interleave(tries, 0),
            center.flatten(0, 1),
            side.flatten(),
            theta.flatten(),
            flip.repeat_interleave(tries),
            self.out_size,
        ).view(batch, tries, N_LANDMARKS, 2)
        inside = ((points >= margin) & (points <= self.out_size - margin)).all(-1).all(-1)
        first = torch.where(
            inside.any(1), inside.float().argmax(1), torch.full_like(flip, -1, dtype=torch.long)
        )
        pick = first.clamp(min=0)
        rows = torch.arange(batch, device=self.device)
        fallback = first < 0
        center = torch.where(fallback[:, None], hand_center(landmarks), center[rows, pick])
        side = torch.where(fallback, torch.full_like(side[:, 0], base_side), side[rows, pick])
        theta = torch.where(fallback, torch.zeros_like(theta[:, 0]), theta[rows, pick])
        return center, side, theta, flip

    def _swap_background(self, rgb, mask, points) -> torch.Tensor:
        cfg = self.cfg["background"]
        batch = rgb.shape[0]
        usable = landmarks_on_mask(mask, points) >= float(cfg["min_landmarks_on_mask"])
        swap = self._chance(batch, float(cfg["p"])) & usable
        if not swap.any():
            return rgb
        backdrop = self._backdrops(rgb, mask)
        blend = mask * rgb + (1 - mask) * backdrop
        return torch.where(swap[:, None, None, None], blend, rgb)

    def _backdrops(self, rgb, mask) -> torch.Tensor:
        """Fundo por amostra: foto do banco (se houver) ou fundo sintético com tons de pele."""
        batch, size = rgb.shape[0], self.out_size
        synthetic = self._synthetic(rgb, mask)
        if self.backgrounds is None:
            return synthetic
        pick = torch.randint(
            0, len(self.backgrounds), (batch,), generator=self.gen, device=self.device
        )
        photos = self.backgrounds[pick].float().div_(255.0)
        if photos.shape[-1] != size:
            photos = F.interpolate(photos, size=(size, size), mode="bilinear", align_corners=False)
        mirror = self._chance(batch, 0.5)
        photos = torch.where(mirror[:, None, None, None], photos.flip(-1), photos)
        use_photo = self._chance(batch, float(self.cfg["background"]["photo_p"]))
        return torch.where(use_photo[:, None, None, None], photos, synthetic)

    def _synthetic(self, rgb, mask) -> torch.Tensor:
        """Gradiente suave, textura, retângulos e elipses com a cor da pele de outra amostra."""
        batch, size = rgb.shape[0], self.out_size
        coarse = self._uniform((batch, 3, 3, 3), 0, 1)
        field = F.interpolate(coarse, size=(size, size), mode="bicubic", align_corners=False)
        texture = F.interpolate(
            self._normal((batch, 1, 24, 24)),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )
        field = field + texture * self._uniform((batch, 1, 1, 1), 0, 0.15)
        ys, xs = torch.meshgrid(
            torch.arange(size, device=self.device, dtype=torch.float32) + 0.5,
            torch.arange(size, device=self.device, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        for _ in range(int(self.cfg["background"]["clutter_shapes"])):
            x0, y0 = self._uniform((2, batch, 1, 1), -0.2, 1.0) * size
            w, h = self._uniform((2, batch, 1, 1), 0.1, 0.6) * size
            box = ((xs >= x0) & (xs < x0 + w) & (ys >= y0) & (ys < y0 + h)).unsqueeze(1)
            color = self._uniform((batch, 3, 1, 1), 0, 1)
            draw = self._chance(batch, 0.6)[:, None, None, None] & box
            field = torch.where(draw, color, field)
        gray = field.mean(dim=1, keepdim=True)
        chroma = self._uniform((batch, 1, 1, 1), *self.cfg["background"]["saturation"])
        field = gray + (field - gray) * chroma
        weight = mask.sum(dim=(2, 3)).clamp(min=1)
        skin = (rgb * mask).sum(dim=(2, 3)) / weight
        donor = torch.roll(torch.arange(batch, device=self.device), 1)
        skin = skin[donor][:, :, None, None]
        for _ in range(int(self.cfg["background"]["skin_blobs"])):
            cx, cy = self._uniform((2, batch, 1, 1), 0, 1) * size
            ax, ay = self._uniform((2, batch, 1, 1), 0.08, 0.3) * size
            blob = (((xs - cx) / ax) ** 2 + ((ys - cy) / ay) ** 2 < 1).unsqueeze(1)
            tone = skin * self._uniform((batch, 1, 1, 1), 0.8, 1.15)
            draw = self._chance(batch, 0.5)[:, None, None, None] & blob
            field = torch.where(draw, tone, field)
        return field.clamp_(0, 1)

    def _photometric(self, rgb) -> torch.Tensor:
        cfg = self.cfg["photometric"]
        batch = rgb.shape[0]
        ones = (batch, 1, 1, 1)
        gains = torch.exp(self._uniform((batch, 3, 1, 1), -1, 1) * float(cfg["white_balance"]))
        gains[:, 1] = 1
        rgb = rgb * gains
        rgb = rgb * (1 + self._uniform(ones, -1, 1) * float(cfg["brightness"]))
        gray = (rgb * rgb.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, True)
        mean = gray.mean(dim=(2, 3), keepdim=True)
        rgb = (rgb - mean) * (1 + self._uniform(ones, -1, 1) * float(cfg["contrast"])) + mean
        gray = (rgb * rgb.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, True)
        rgb = gray + (rgb - gray) * (1 + self._uniform(ones, -1, 1) * float(cfg["saturation"]))
        rgb = self._hue(rgb, self._uniform((batch,), -1, 1) * float(cfg["hue"]) * 2 * math.pi)
        low, high = (math.log(v) for v in cfg["gamma"])
        gamma = self._uniform(ones, low, high).exp_()
        rgb = rgb.clamp(1e-4, 1) ** gamma
        rgb = self._blur(rgb)
        rgb = self._downscale(rgb)
        noisy = self._chance(batch, float(cfg["noise_p"]))[:, None, None, None]
        std = self._uniform(ones, *cfg["noise_std"])
        rgb = torch.where(noisy, rgb + self._normal(rgb.shape) * std, rgb)
        return rgb.clamp_(0, 1)

    @staticmethod
    def _hue(rgb, angle) -> torch.Tensor:
        """Giro do matiz no plano de crominância do YIQ, sem mexer na luminância."""
        to_yiq = rgb.new_tensor(
            [[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]]
        )
        cos, sin = torch.cos(angle), torch.sin(angle)
        spin = torch.zeros(len(angle), 3, 3, device=rgb.device)
        spin[:, 0, 0] = 1
        spin[:, 1, 1], spin[:, 1, 2], spin[:, 2, 1], spin[:, 2, 2] = cos, -sin, sin, cos
        matrix = torch.linalg.inv(to_yiq) @ spin @ to_yiq
        return torch.einsum("bij,bjhw->bihw", matrix, rgb)

    def _blur(self, rgb) -> torch.Tensor:
        """Desfoque gaussiano ou de movimento, com núcleo próprio por amostra."""
        cfg = self.cfg["photometric"]
        batch, k = rgb.shape[0], 9
        coords = torch.arange(k, device=self.device, dtype=torch.float32) - k // 2
        ys, xs = torch.meshgrid(coords, coords, indexing="ij")
        sigma = self._uniform((batch, 1, 1), *cfg["blur_sigma"])
        gauss = torch.exp(-(xs**2 + ys**2) / (2 * sigma**2))
        length = self._uniform((batch, 1, 1), *cfg["motion_length"])
        angle = self._uniform((batch, 1, 1), 0, math.pi)
        along = xs * torch.cos(angle) + ys * torch.sin(angle)
        across = -xs * torch.sin(angle) + ys * torch.cos(angle)
        motion = (1 - across.abs()).clamp(min=0) * (along.abs() <= length / 2)
        identity = ((xs == 0) & (ys == 0)).float().expand(batch, k, k)
        pick = torch.rand(batch, generator=self.gen, device=self.device)
        p_gauss, p_motion = float(cfg["blur_p"]), float(cfg["motion_p"])
        kernel = torch.where((pick < p_gauss)[:, None, None], gauss, identity)
        in_motion = (pick >= p_gauss) & (pick < p_gauss + p_motion)
        kernel = torch.where(in_motion[:, None, None], motion, kernel)
        kernel = kernel / kernel.sum(dim=(1, 2), keepdim=True)
        weight = kernel.repeat_interleave(3, 0)[:, None]
        padded = F.pad(rgb.reshape(1, batch * 3, *rgb.shape[-2:]), [k // 2] * 4, mode="reflect")
        return F.conv2d(padded, weight, groups=batch * 3).view_as(rgb)

    def _downscale(self, rgb) -> torch.Tensor:
        """Perde resolução e volta ao tamanho: sensor pior, distância maior, zoom digital."""
        cfg = self.cfg["photometric"]
        chosen = self._chance(rgb.shape[0], float(cfg["downscale_p"])).nonzero().flatten()
        if chosen.numel() == 0:
            return rgb
        out = rgb.clone()
        size = rgb.shape[-1]
        for index in chosen.tolist():
            factor = float(self._uniform((1,), *cfg["downscale_factor"]))
            small = max(8, int(round(size * factor)))
            low = F.interpolate(rgb[index : index + 1], size=(small, small), mode="area")
            out[index] = F.interpolate(
                low, size=(size, size), mode="bilinear", align_corners=False
            )[0]
        return out

    def _occlude(self, rgb, points) -> torch.Tensor:
        """Retângulos de ruído colorido que cobrem no máximo `max_landmarks` dos 21 pontos.

        Só o RGB é ocluído: a profundidade fica como informação privilegiada do professor.
        """
        cfg = self.cfg["occlusion"]
        batch, size = rgb.shape[0], self.out_size
        tries = int(cfg["tries"])
        ys, xs = torch.meshgrid(
            torch.arange(size, device=self.device, dtype=torch.float32) + 0.5,
            torch.arange(size, device=self.device, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        for _ in range(int(cfg["max_boxes"])):
            active = self._chance(batch, float(cfg["p"]))
            area = self._uniform((batch, tries), *cfg["area"]) * size * size
            aspect = self._uniform((batch, tries), -1, 1).mul_(math.log(2)).exp_()
            w, h = (area * aspect).sqrt(), (area / aspect).sqrt()
            x0 = self._uniform((batch, tries), 0, 1) * (size - w).clamp(min=0)
            y0 = self._uniform((batch, tries), 0, 1) * (size - h).clamp(min=0)
            px, py = points[:, None, :, 0], points[:, None, :, 1]
            covered = (
                (px >= x0[..., None]) & (px < (x0 + w)[..., None])
                & (py >= y0[..., None]) & (py < (y0 + h)[..., None])
            ).sum(-1)  # fmt: skip
            ok = covered <= int(cfg["max_landmarks"])
            pick = ok.float().argmax(1)
            rows = torch.arange(batch, device=self.device)
            active = active & ok.any(1)
            bx0, by0, bw, bh = (t[rows, pick][:, None, None, None] for t in (x0, y0, w, h))
            box = (xs >= bx0) & (xs < bx0 + bw) & (ys >= by0) & (ys < by0 + bh)
            fill = self._uniform((batch, 3, 1, 1), 0, 1) + self._normal(rgb.shape) * 0.08
            rgb = torch.where(active[:, None, None, None] & box, fill.clamp(0, 1), rgb)
        return rgb


class EvalView:
    """Vista de avaliação e do app: caixa da mão × `view_context`, sem giro nem espelho."""

    def __init__(self, hand_side_px: float, view_context: float, out_size: int, device="cpu"):
        self.hand_side_px = float(hand_side_px)
        self.view_context = float(view_context)
        self.out_size = int(out_size)
        self.device = torch.device(device)

    def __call__(self, rgb, depth, landmarks, flip: bool = False) -> dict[str, torch.Tensor]:
        rgb, depth = as_float_batch(rgb.to(self.device), depth.to(self.device))
        landmarks = landmarks.to(self.device).float()
        batch = rgb.shape[0]
        center = hand_center(landmarks)
        side = torch.full((batch,), self.hand_side_px * self.view_context, device=self.device)
        theta = torch.zeros(batch, device=self.device)
        flips = torch.full((batch,), flip, dtype=torch.bool, device=self.device)
        rgb, depth, points = warp(rgb, depth, landmarks, center, side, theta, flips, self.out_size)
        return {"rgb": rgb, "depth": depth, "landmarks": points, "flip": flips}

    def landmarks_only(self, landmarks, flip: bool = False) -> dict[str, torch.Tensor]:
        landmarks = landmarks.to(self.device).float()
        batch = landmarks.shape[0]
        side = torch.full((batch,), self.hand_side_px * self.view_context, device=self.device)
        theta = torch.zeros(batch, device=self.device)
        flips = torch.full((batch,), flip, dtype=torch.bool, device=self.device)
        points = map_landmarks(landmarks, hand_center(landmarks), side, theta, flips, self.out_size)
        return {"landmarks": points, "flip": flips}


def mixup(
    tensors: dict[str, torch.Tensor],
    alpha: float,
    generator: torch.Generator,
    dominant: tuple[str, ...] = (),
) -> tuple[dict[str, torch.Tensor], float, torch.Tensor]:
    """Mistura o lote com uma permutação dele mesmo. Só na destilação: o professor rotula a mistura.

    Os tensores em `dominant` (os pontos, que não têm média com sentido) ficam com
    a amostra de maior peso. Devolve (mistura, lambda, permutação), para a perda
    dura ponderar os dois rótulos.
    """
    first = next(iter(tensors.values()))
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(first.shape[0], generator=generator, device=first.device)
    mixed = {}
    for name, value in tensors.items():
        if name in dominant:
            mixed[name] = value if lam >= 0.5 else value[perm]
        else:
            mixed[name] = lam * value + (1 - lam) * value[perm]
    return mixed, lam, perm
