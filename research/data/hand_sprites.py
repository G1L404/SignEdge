"""Sprite sheet das 61 mãos para o app: render 3D sintético (SDF + sphere tracing).

O app não embarca imagens da base: cada mão sai dos 21 pontos canônicos de
`canonical_hands.compute` (x, y normalizados) e de um z por ponto.

1. Juntas 3D. O z é o dos `hand_world_landmarks` do MediaPipe (`hand_landmarker.task`)
   no recorte RGB do quadro escolhido, levado à escala do x, y (raio RMS). Vale quando a
   nova detecção cai sobre os pontos guardados (erro médio abaixo de `MATCH_PX`). Senão,
   o z é estimado: cada falange tem um comprimento 3D fixo, em palmas (percentil 95 do
   comprimento 2D nas 61 CMs), e o encurtamento 2D vira profundidade numa cadeia plana
   em que o ângulo de flexão só cresce; de perfil, a base dos dedos avança do mínimo ao
   polegar. Nos dois casos, nenhuma junta de dedo fica dentro da palma (a que cair nela
   sai para a câmera), e no punho fechado a falange que volta para o punho encolhe no
   plano, compensando as pontas arredondadas das cápsulas. Os ajustes por CM
   (cruzamento, polegar, inclinação dos dedos) são as constantes do topo.
2. Superfície. A palma (elipsoide achatado de frente para a câmera, raios metacarpais
   0–5, 0–9, 0–13, 0–17, a fileira 5–9–13–17), as almofadas tenar e hipotenar, o
   metacarpo do polegar e o pulso, cortado reto com borda arredondada, se fundem por
   `smin` polinomial. Cada dedo são cápsulas cônicas unidas à palma por um `smin` curto:
   o dedo dobrado sobre a palma fica separado dela por um vinco, e entre dois dedos fica
   um sulco. As pregas (tenar, transversal, do pulso) e os vincos das juntas são
   depressões gaussianas somadas ao SDF, e por isso as normais e a oclusão os enxergam.
3. Render ortográfico a 320 px, reduzido a 160 px (anti-aliasing 2×): normais por
   diferenças centrais com um relevo fino de pele, oclusão ambiente pelo SDF ao longo
   da normal, sombra de fresta onde dois dedos (ou um dedo dobrado e a palma) se
   encostam, sombra suave da luz principal (alto à esquerda), preenchimento quente,
   espalhamento subsuperficial no terminador e especular leve. A cor de cada dedo tinge
   o albedo, em rampa desde a base, só na superfície do próprio dedo.

    python -m research.data.hand_sprites --out mobile/assets/hands_61.webp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

COLUMNS, ROWS, TILE = 11, 6, 160
RENDER = 2 * TILE
CHAINS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
PALM_POINTS = (0, 5, 9, 13, 17)
COLORS = ["#F97316", "#3B82F6", "#22C55E", "#A855F7", "#EC4899"]
SKIN, SKIN_DEEP = "#E8B496", "#C68B6B"
TINT = 0.78  # peso da cor do dedo no albedo, com a cor puxada para o brilho da pele
MARGIN = 0.05
MAX_BEND = np.radians(115)  # flexão máxima de uma junta
STEPS = 96
MATCH_PX = 8.0  # erro médio máximo entre a nova detecção e os pontos guardados
THUMB = (1.22, 1.05, 0.90, 0.78)  # raio do polegar nas juntas 1, 2, 3, 4, em r_finger
# CMs com indicador e médio cruzados: um passa à frente do outro, e as pontas se afastam
CROSSED = {34, 35}
CROSS_GAP = 0.18  # diferença mínima de z entre as distais cruzadas, em palmas
TIP_GAP = 0.30  # distância mínima, no plano, da ponta de trás ao dedo da frente, em palmas
# CMs com os dedos dobrados a 90° na direção da câmera: giro dos dedos em torno da
# fileira dos nós, que mostra o comprimento deles na tela
FINGER_PITCH = {30: np.radians(20)}
FOLD_RETRACT = 0.88  # encurtamento 2D da falange que volta para o punho, no punho fechado
FIST_REACH = 1.15  # dedo em punho: a ponta fica mais perto do punho que isto × a base
# CMs com o polegar encostado no indicador: fração do caminho de 2–4 até a lateral dele
THUMB_TUCK = {1: 0.35, 9: 0.35}
FIST_COMPACT = {1: 0.78, 9: 0.78}  # punho alto: as juntas PIP descem até isto do nó
THUMB_BETWEEN = {11}  # a ponta do polegar entre o indicador e o médio (letra T)
EVEN_FINGERS = {26}  # os 4 dedos com o mesmo z, descendo juntos sobre o polegar
WRIST_CUT = 0.55  # onde o pulso termina, ao longo do eixo dele
RADIAL_TILT = 0.06  # de perfil, a base avança (2 − dedo) × isto × perfil, em palmas


def profile(xy: np.ndarray, palm: float) -> float:
    """0 com a palma de frente, 1 de perfil: o quanto 5–17 encolheu no plano."""
    return float(np.clip(1.0 - np.linalg.norm(xy[17] - xy[5]) / (0.75 * palm), 0.0, 1.0))


def _linear(hex_color: str) -> np.ndarray:
    rgb = np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return rgb**2.2


# --- 1. juntas 3D ----------------------------------------------------------------------


def bone_lengths(hands: np.ndarray) -> dict[tuple[int, int], float]:
    """Comprimento 3D de cada falange, em palmas: percentil 95 nas 61 CMs."""
    from research.data.canonical_hands import palm_size

    palm = palm_size(hands)
    pairs = [(a, b) for chain in CHAINS for a, b in zip(chain, chain[1:], strict=False)]
    return {
        (a, b): float(np.percentile(np.linalg.norm(hands[:, b] - hands[:, a], axis=1) / palm, 95))
        for a, b in pairs
    }


def _inside(point: np.ndarray, polygon: np.ndarray) -> bool:
    signs = [
        np.sign((b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]))
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True)
    ]
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


def lift(xyz: np.ndarray, lengths: dict[tuple[int, int], float]) -> np.ndarray:
    """(21, 3) com o z das juntas dos dedos refeito pelos comprimentos 3D."""
    from research.data.canonical_hands import palm_size

    xy = xyz[:, :2]
    palm = float(palm_size(xy[None])[0])
    # plano da palma pelo sensor, com a inclinação contida (o sensor é ruidoso)
    a = np.c_[xy[list(PALM_POINTS)], np.ones(5)]
    coef = np.linalg.lstsq(a, xyz[list(PALM_POINTS), 2], rcond=None)[0]
    coef[:2] = np.clip(coef[:2] * 0.6, -0.35, 0.35)
    plane = lambda p: float(coef[0] * p[0] + coef[1] * p[1] + coef[2])  # noqa: E731
    out = np.c_[xy, [plane(p) for p in xy]]
    side = profile(xy, palm)
    for f, chain in enumerate(CHAINS):
        # de perfil, a borda radial (polegar, indicador) vem para a câmera
        out[chain[0], 2] += (2 - f) * RADIAL_TILT * side * palm
        axis = xy[chain[1]] - (xy[0] if chain[0] != 1 else xy[1])
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        theta = 0.0
        for a_, b_ in zip(chain, chain[1:], strict=False):
            full = lengths[(a_, b_)] * palm
            bone = xy[b_] - xy[a_]
            seen = min(np.linalg.norm(bone), full)
            along = float(np.clip(np.dot(bone, axis) / full, -1, 1))
            first = np.arccos(along)
            # duas soluções: θ ou 2π − θ; a flexão só cresce, até MAX_BEND por junta
            options = [
                t for t in (first, 2 * np.pi - first) if theta - 0.15 <= t <= theta + MAX_BEND
            ]
            new = min(options, key=lambda t: abs(t - theta)) if options else theta + MAX_BEND
            if not options and first < theta:
                new = max(first, theta)
            theta = new
            dz = np.sqrt(max(full**2 - seen**2, 0.0)) * np.sign(np.sin(theta) or 1.0)
            out[b_, 2] = out[a_, 2] + dz
    if side > 0.35:
        # de perfil, médio, anelar e mínimo ficam atrás do indicador e do polegar
        front = min(out[list(CHAINS[0]), 2].max(), out[list(CHAINS[1]), 2].max())
        for chain in CHAINS[2:]:
            out[list(chain), 2] = np.minimum(out[list(chain), 2], front)
    return _off_palm(out, palm)


def _fold_retract(p: np.ndarray) -> np.ndarray:
    """Punho fechado: a falange que volta para o punho encolhe um pouco no plano, para
    compensar as pontas arredondadas das cápsulas, e arrasta o resto do dedo."""
    p = p.copy()
    xy = p[:, :2].copy()
    for chain in CHAINS[1:]:
        if np.linalg.norm(xy[chain[3]] - xy[0]) >= FIST_REACH * np.linalg.norm(
            xy[chain[0]] - xy[0]
        ):
            continue
        axis = xy[chain[1]] - xy[0]
        axis /= np.linalg.norm(axis) + 1e-9
        for a, b in zip(chain[1:], chain[2:], strict=False):
            bone = xy[b] - xy[a]
            if bone @ axis < 0:
                p[list(chain[chain.index(b) :]), :2] += (FOLD_RETRACT - 1) * bone
    return p


def _tuck_thumb(p: np.ndarray, pull: float) -> np.ndarray:
    """Polegar encostado no indicador: 2, 3 e 4 vão juntos para a lateral do indicador
    (5–6), sem vão entre eles e sem entortar o polegar."""
    p = p.copy()
    p[[2, 3, 4]] += pull * ((p[5] + p[6]) / 2 - (p[3] + p[4]) / 2)
    return p


def _compact_fist(p: np.ndarray, keep: float) -> np.ndarray:
    """Punho alto: cada PIP desce em direção ao nó, e o resto do dedo vai junto."""
    p = p.copy()
    for chain in CHAINS[1:]:
        shift = (keep - 1) * (p[chain[1], :2] - p[chain[0], :2])
        p[list(chain[1:]), :2] += shift
    return p


def _thumb_between(p: np.ndarray, palm: float) -> np.ndarray:
    """Letra T: a ponta do polegar entre o indicador e o médio, à frente deles."""
    p = p.copy()
    p[4, :2] = (p[6, :2] + p[10, :2]) / 2
    p[4, 2] = max(p[6, 2], p[10, 2]) + 0.1 * palm
    p[3] = (p[2] + p[4]) / 2
    p[3, 2] = max(p[3, 2], p[4, 2] - 0.05 * palm)
    return p


def _even_fingers(p: np.ndarray) -> np.ndarray:
    """Os 4 dedos com o mesmo z em cada junta: descem juntos, sem se entortar."""
    p = p.copy()
    for j in (1, 2, 3):
        joints = [chain[j] for chain in CHAINS[1:]]
        p[joints, 2] = p[joints, 2].mean()
    return p


def _off_palm(p: np.ndarray, palm: float) -> np.ndarray:
    """Nenhuma junta de dedo dentro da palma: a que cair nela sai para a câmera.

    A palma é o elipsoide do desenho; a junta empurrada leva o resto do dedo junto,
    só em z, para o x, y dos pontos ficar como veio.
    """
    e1, e2, e3 = _frame(p)
    top = p[[5, 9, 13, 17]].mean(0)
    center = (p[0] + top) / 2
    ru = 0.5 * np.linalg.norm(p[17] - p[5]) + 0.1 * palm
    rv = 0.5 * np.linalg.norm(top - p[0])
    clear = 0.24 * palm + 0.14 * palm  # meia espessura da palma mais o raio do dedo
    for f, chain in enumerate(CHAINS):
        for j in chain[2:] if f == 0 else chain[1:]:
            q = p[j] - center
            u, v, w = q @ e1, q @ e2, q @ e3
            if (u / ru) ** 2 + (v / rv) ** 2 < 1 and w < clear:
                rest = list(chain[chain.index(j) :])
                p[rest, 2] += clear - w
    return p


# --- 2. superfície -----------------------------------------------------------------------


def _frame(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Referencial da palma no plano da imagem, com e3 para a câmera."""
    e1 = np.r_[p[17, :2] - p[5, :2], 0.0]
    e1 /= np.linalg.norm(e1)
    e2 = np.r_[p[[5, 9, 13, 17], :2].mean(0) - p[0, :2], 0.0]
    e2 -= e1 * np.dot(e2, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2, np.array([0.0, 0.0, 1.0])


def build_scene(p: np.ndarray) -> dict:
    """Primitivas da mão em (21, 3): elipsoides e cápsulas cônicas por grupo."""
    palm = float(np.mean([np.linalg.norm(p[a] - p[b]) for a, b in ((0, 5), (0, 17), (5, 17))]))
    e1, e2, e3 = _frame(p)
    knuckles = p[[5, 9, 13, 17]]
    seen_width = np.linalg.norm(p[17, :2] - p[5, :2])
    width = max(seen_width, 0.58 * palm)  # piso anatômico do pulso: de perfil, 5–17 encolhe
    r_finger = 0.145 * palm
    top = knuckles.mean(0)
    rot = np.stack([e1, e2, e3])
    ellipsoids = [
        # palma: do pulso à fileira dos nós, achatada; de perfil a largura gira para
        # a profundidade, então o elipsoide não alarga além do que se vê
        (
            (p[0] + top) / 2 + e2 * 0.02 * palm,
            rot,
            (
                0.55 * max(seen_width, 0.38 * palm),
                0.52 * np.linalg.norm(top - p[0]),
                0.24 * palm,
            ),
        ),
    ]
    # tenar: entre o pulso, a base do polegar e o indicador
    thenar_c = 0.30 * p[1] + 0.30 * p[2] + 0.15 * p[0] + 0.25 * p[5]
    t1 = p[2] - p[0]
    t1 -= e3 * np.dot(t1, e3)
    t1 /= np.linalg.norm(t1) + 1e-9
    t2 = np.cross(e3, t1)
    ellipsoids.append(
        (
            thenar_c + e3 * 0.03 * palm,
            np.stack([t1, t2, e3]),
            (0.36 * palm, 0.22 * palm, 0.2 * palm),
        )
    )
    # hipotenar: a almofada da borda da palma, entre o pulso e o mínimo
    u1 = p[17] - p[0]
    u1 -= e3 * np.dot(u1, e3)
    u1 /= np.linalg.norm(u1) + 1e-9
    ellipsoids.append(
        (
            0.45 * p[0] + 0.45 * p[17] + 0.10 * p[9] + e3 * 0.02 * palm,
            np.stack([u1, np.cross(e3, u1), e3]),
            (0.34 * palm, 0.17 * palm, 0.19 * palm),
        )
    )
    capsules = []  # (a, b, ra, rb, grupo); grupo 0 = palma
    for k in (5, 9, 13, 17):
        capsules.append((p[0], p[k], 0.16 * palm, 0.9 * r_finger, 0))
    for a, b in ((5, 9), (9, 13), (13, 17)):
        capsules.append((p[a], p[b], 0.95 * r_finger, 0.92 * r_finger, 0))
    # membrana entre o polegar e o indicador (adutor do polegar)
    capsules.append((p[1], p[5], 0.15 * palm, 0.11 * palm, 0))
    capsules.append((p[2], p[5], 0.11 * palm, 0.10 * palm, 0))
    wrist_dir = p[0] - top
    wrist_dir[2] = 0.0  # o antebraço segue no plano da imagem: o corte fica de perfil
    wrist_dir /= np.linalg.norm(wrist_dir)
    wrist = (
        p[0] - wrist_dir * 0.15 * palm,
        p[0] + wrist_dir * 0.95 * palm,
        0.32 * width,
        0.29 * width,
    )
    center = p[list(PALM_POINTS)].mean(0)
    creases = [  # (curva, profundidade em palmas)
        # tenar: da borda radial entre o indicador e o polegar, contornando a tenar
        (_curve(0.72 * (p[5] + p[9]) / 2 + 0.28 * p[0], center, (p[0] + p[1]) / 2), 0.011),
        # transversal: abaixo do indicador até a borda abaixo do mínimo
        (
            _curve(
                0.78 * p[5] + 0.22 * p[0],
                0.6 * (p[9] + p[13]) / 2 + 0.4 * p[0],
                0.72 * p[17] + 0.28 * p[0],
            ),
            0.011,
        ),
        # pulso, logo abaixo de p0, rasa
        (
            _curve(
                p[0] + wrist_dir * 0.14 * palm - e1 * 0.42 * width,
                p[0] + wrist_dir * 0.17 * palm,
                p[0] + wrist_dir * 0.14 * palm + e1 * 0.42 * width,
            ),
            0.004,
        ),
    ]
    scale = [0.95, 0.82, 0.73, 0.65]
    rings = []  # (junta, eixo, raio): vincos interfalangeanos
    for f, chain in enumerate(CHAINS):
        # o polegar tem a base carnuda (o metacarpo 1→2 fica dentro da tenar)
        radii = THUMB if f == 0 else [k * (0.9 if f == 4 else 1.0) for k in scale]
        for j, (a, b) in enumerate(zip(chain, chain[1:], strict=False)):
            capsules.append((p[a], p[b], radii[j] * r_finger, radii[j + 1] * r_finger, f + 1))
        for j in (2,) if f == 0 else (1, 2):  # PIP e DIP; no polegar, só a IP
            axis = p[chain[j + 1]] - p[chain[j - 1]]
            rings.append((p[chain[j]], axis / (np.linalg.norm(axis) + 1e-9), radii[j] * r_finger))
    return {
        "palm": palm,
        "ellipsoids": ellipsoids,
        "capsules": capsules,
        "wrist": wrist,
        "wrist_frame": rot,
        "frame": (center, rot),
        "thumb_on_palm": _inside(p[4, :2], p[list(PALM_POINTS), :2]),
        "creases": creases,
        "rings": rings,
    }


def _curve(a: np.ndarray, control: np.ndarray, b: np.ndarray, n: int = 10) -> np.ndarray:
    """Bézier quadrática a → b, em n pontos."""
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * control + t**2 * b


class Scene:
    """O SDF de uma mão em torch, avaliado em lotes de pontos (N, 3)."""

    def __init__(self, scene: dict, device: str):
        import torch

        self.t = torch
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731
        self.palm = scene["palm"]
        self.ell = [(f(c), f(r), f(s)) for c, r, s in scene["ellipsoids"]]
        caps = scene["capsules"]
        self.cap_a = f([c[0] for c in caps])
        self.cap_b = f([c[1] for c in caps])
        self.cap_r = f([[c[2], c[3]] for c in caps])
        self.cap_g = [c[4] for c in caps]
        wa, wb, wr1, wr2 = scene["wrist"]
        self.wrist = (f(wa), f(wb), wr1, wr2)
        self.wrist_rot = f(scene["wrist_frame"])
        self.k_palm = 0.09 * self.palm
        self.k_join = 0.012 * self.palm  # curto: o dedo dobrado fica separado da palma
        self.k_thumb_base = 0.11 * self.palm  # o metacarpo do polegar se funde à tenar
        self.k_thumb = 0.045 * self.palm  # transição das falanges do polegar para a base
        self.k_finger = 0.006 * self.palm
        self.k_wrist = 0.22 * self.palm
        self.thumb_on_palm = scene["thumb_on_palm"]
        center, rot = scene["frame"]
        self.center, self.rot = f(center), f(rot)
        curves = scene["creases"]
        self.crease_a = f(np.concatenate([c[:-1] for c, _ in curves]) @ rot[:2].T)
        self.crease_b = f(np.concatenate([c[1:] for c, _ in curves]) @ rot[:2].T)
        self.crease_amp = f(np.concatenate([[amp] * (len(c) - 1) for c, amp in curves]))
        self.ring_p = f([r[0] for r in scene["rings"]])
        self.ring_u = f([r[1] for r in scene["rings"]])
        self.ring_r = f([r[2] for r in scene["rings"]])
        axis = wb - wa
        self.cut_len = float(np.linalg.norm(axis)) * WRIST_CUT
        self.wrist_u = f(axis / np.linalg.norm(axis))

    def smin(self, a, b, k):
        h = (0.5 + 0.5 * (b - a) / k).clamp(0, 1)
        return b * (1 - h) + a * h - k * h * (1 - h)

    def smax(self, a, b, k):
        return -self.smin(-a, -b, k)

    def creases(self, p):
        """Profundidade das pregas e vincos em p: somada ao SDF, afunda a superfície."""
        t = self.t
        # pregas da palma: distância no plano da palma, só do lado palmar
        q = p - self.center
        uv = q @ self.rot[:2].T + self.center @ self.rot[:2].T
        ab = self.crease_b - self.crease_a
        pa = uv[:, None] - self.crease_a[None]
        h = ((pa * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-9)).clamp(0, 1)
        dist = (pa - ab * h[..., None]).norm(dim=-1)
        palmar = t.sigmoid((q @ self.rot[2]) / (0.03 * self.palm))
        depth = self.crease_amp * t.exp(-((dist / (0.028 * self.palm)) ** 2))
        lines = self.palm * depth.max(1).values * palmar
        # vincos das juntas: um anel estreito em volta de cada PIP e DIP
        v = p[:, None] - self.ring_p[None]
        ax = (v * self.ring_u[None]).sum(-1)
        rad = (v - self.ring_u[None] * ax[..., None]).norm(dim=-1)
        near = t.exp(-(((rad - self.ring_r) / (0.6 * self.ring_r)).clamp(min=0) ** 2) * 4)
        band = t.exp(-((ax / (0.03 * self.palm)) ** 2)) * near * (rad < 1.6 * self.ring_r)
        rings = 0.010 * self.palm * band.max(1).values
        return t.maximum(lines, rings)

    def capsules(self, p):
        """(N, C) distância a cada cápsula cônica e a posição h ao longo dela."""
        pa = p[:, None] - self.cap_a[None]
        ba = (self.cap_b - self.cap_a)[None]
        h = ((pa * ba).sum(-1) / (ba * ba).sum(-1)).clamp(0, 1)
        r = self.cap_r[:, 0] * (1 - h) + self.cap_r[:, 1] * h
        return (pa - ba * h[..., None]).norm(dim=-1) - r, h

    def ellipsoid(self, p, center, rot, radii):
        q = (p - center) @ rot.T
        k0 = (q / radii).norm(dim=-1)
        k1 = (q / radii**2).norm(dim=-1)
        return k0 * (k0 - 1) / (k1 + 1e-9)

    def wrist_sdf(self, p):
        a, b, r1, r2 = self.wrist
        q = (p - a) @ self.wrist_rot.T
        flat = self.t.tensor([1.0, 1.0, 1.55], device=p.device)
        ba = (b - a) @ self.wrist_rot.T
        q = q * flat
        ba = ba * flat
        h = ((q * ba).sum(-1) / (ba * ba).sum()).clamp(0, 1)
        tube = ((q - ba * h[:, None]).norm(dim=-1) - (r1 * (1 - h) + r2 * h)) / 1.55
        # corte reto do pulso, com a borda arredondada
        cut = (p - a) @ self.wrist_u - self.cut_len
        return self.smax(tube, cut, 0.05 * self.palm), h

    def groups(self, p):
        """Distância da palma e de cada dedo, e o h de cada cápsula (para o tingimento)."""
        caps, h = self.capsules(p)
        palm = None
        for c, r, s in self.ell:
            e = self.ellipsoid(p, c, r, s)
            palm = e if palm is None else self.smin(palm, e, self.k_palm)
        palm = self.smin(palm, self.wrist_sdf(p)[0], self.k_wrist)  # sem degrau no pulso
        fingers = []
        for g in range(6):
            idx = [i for i, cg in enumerate(self.cap_g) if cg == g]
            if g == 0:
                for i in idx:
                    palm = self.smin(palm, caps[:, i], self.k_palm)
            else:
                fingers.append([caps[:, i] for i in idx])
        # o metacarpo do polegar (1→2) faz parte da palma
        palm = self.smin(palm, fingers[0][0], self.k_thumb_base)
        return palm, fingers, h

    def __call__(self, p):
        palm, fingers, _ = self.groups(p)
        out = None
        for f, (prox, mid, dist) in enumerate(fingers):
            if f == 0:  # o polegar livre são só 2→3 e 3→4
                joined = self.smin(palm, self.smin(mid, dist, self.k_finger), self.k_thumb)
            else:
                finger = self.smin(self.smin(prox, mid, self.k_finger), dist, self.k_finger)
                joined = self.smin(palm, finger, self.k_join)
            out = joined if out is None else self.t.minimum(out, joined)
        return out + self.creases(p)

    def contact(self, p):
        """Distância à segunda superfície mais próxima: pequena onde duas se encostam."""
        palm, fingers, _ = self.groups(p)
        whole = self.t.stack(
            [self.t.stack(f[1:] if g == 0 else f, 0).min(0).values for g, f in enumerate(fingers)],
            -1,
        )
        finger_pair = whole.topk(2, dim=-1, largest=False).values[:, 1]
        # dedo dobrado sobre a palma; do polegar, só a distal, quando a ponta cai na palma
        outer = [self.t.minimum(f[1], f[2]) for f in fingers[1:]]
        if self.thumb_on_palm:
            outer.append(fingers[0][2])
        folded = self.t.maximum(palm, self.t.stack(outer, -1).min(-1).values)
        return self.t.minimum(finger_pair, folded)


# --- 3. render ---------------------------------------------------------------------------


def _view(scene: dict) -> tuple[np.ndarray, float]:
    """Centro e meio lado da janela que cobre a mão e o pulso."""
    extent = [c[0][:2] for c in scene["capsules"]] + [c[1][:2] for c in scene["capsules"]]
    wa, wb, _, _ = scene["wrist"]
    extent.append(wa[:2] + (wb[:2] - wa[:2]) * WRIST_CUT)  # o pulso termina cortado
    lo = np.min(extent, 0) - scene["palm"] * 0.22
    hi = np.max(extent, 0) + scene["palm"] * 0.22
    center = (lo + hi) / 2
    half = float((hi - lo).max() / 2 / (1 - 2 * MARGIN))
    return center, half


def render(points: np.ndarray, device: str = "cuda") -> np.ndarray:
    """RGBA float (TILE, TILE, 4) de uma mão (21, 3), alfa não pré-multiplicado."""
    import torch
    import torch.nn.functional as F

    spec = build_scene(points)
    sdf = Scene(spec, device)
    center, half = _view(spec)
    wa, wb, _, _ = spec["wrist"]
    n = RENDER
    lin = torch.linspace(-1, 1, n, device=device) * half * (1 - 1 / n)
    yy, xx = torch.meshgrid(lin + float(center[1]), lin + float(center[0]), indexing="ij")
    z_top = float(max(points[:, 2].max(), wa[2], wb[2]) + spec["palm"])
    z_bottom = float(points[:, 2].min() - spec["palm"])
    origin = torch.stack([xx, yy, torch.full_like(xx, z_top)], -1).reshape(-1, 3)
    direction = torch.tensor([0.0, 0.0, -1.0], device=device)
    t = torch.zeros(origin.shape[0], device=device)
    hit = torch.zeros_like(t, dtype=torch.bool)
    eps = 1e-3 * spec["palm"]
    with torch.no_grad():
        active = torch.ones_like(hit)
        for _ in range(STEPS):
            idx = active.nonzero().squeeze(1)
            if idx.numel() == 0:
                break
            d = sdf(origin[idx] + direction * t[idx, None])
            t[idx] += 0.85 * d
            done = d < eps
            hit[idx[done]] = True
            gone = t[idx] > z_top - z_bottom
            active[idx[done | gone]] = False
        p = origin + direction * t[:, None]
        rgb = torch.zeros(origin.shape[0], 3, device=device)
        alpha = hit.float()
        idx = hit.nonzero().squeeze(1)
        q = p[idx]
        # normais
        e = 2e-3 * spec["palm"]
        offsets = torch.eye(3, device=device) * e
        normal = torch.stack([sdf(q + offsets[i]) - sdf(q - offsets[i]) for i in range(3)], -1)
        normal = F.normalize(normal, dim=-1)
        # poros: um relevo fino de alta frequência na normal
        w = 70.0 / spec["palm"]
        pores = torch.stack(
            [
                torch.sin(w * (q @ torch.tensor(d, device=device)) + ph)
                for d, ph in (
                    ([1.0, 0.6, 0.3], 0.0),
                    ([-0.4, 1.0, 0.7], 1.7),
                    ([0.5, -0.8, 1.0], 3.1),
                )
            ],
            -1,
        )
        normal = F.normalize(normal + 0.012 * pores, dim=-1)
        # oclusão ambiente
        ao = torch.zeros(q.shape[0], device=device)
        for i in range(1, 6):
            h = 0.06 * spec["palm"] * i
            ao += (h - sdf(q + normal * h)).clamp(min=0) / (2.0 ** (i - 1))
        ao = (1 - 4.0 * ao / spec["palm"]).clamp(0.1, 1) ** 1.3
        # luz principal (alto à esquerda) e sombra suave
        key = F.normalize(torch.tensor([-0.55, -0.7, 0.75], device=device), dim=0)
        shadow = torch.ones(q.shape[0], device=device)
        start = q + normal * 4 * eps
        travel = torch.full_like(shadow, 0.02 * spec["palm"])
        for _ in range(40):
            d = sdf(start + key * travel[:, None])
            shadow = torch.minimum(shadow, (10 * d / travel).clamp(0, 1))
            travel += d.clamp(0.01 * spec["palm"], 0.3 * spec["palm"])
        shadow = shadow.clamp(0, 1) ** 1.5
        # fresta: onde dois dedos, ou um dedo dobrado e a palma, se encostam
        gap = sdf.contact(q)
        s = (gap / (0.09 * spec["palm"])).clamp(0, 1)
        crevice = 0.32 + 0.68 * s * s * (3 - 2 * s)
        # albedo: pele com os dedos tingidos desde a base
        palm_d, fingers, h = sdf.groups(q)
        skin = torch.tensor(_linear(SKIN), device=device, dtype=torch.float32)
        LUMA = torch.tensor([0.2126, 0.7152, 0.0722], device=device)  # noqa: N806
        deep = torch.tensor(_linear(SKIN_DEEP), device=device, dtype=torch.float32)
        tau = 0.004 * spec["palm"]
        # o polegar pinta só as falanges livres; o metacarpo é palma
        finger_d = [
            torch.stack(f[1:] if g == 0 else f, 0).min(0).values for g, f in enumerate(fingers)
        ]
        soft = torch.softmax(-torch.stack([palm_d, *finger_d], -1) / tau, -1)
        first_caps = [i for i, g in enumerate(sdf.cap_g) if g > 0][::3]
        albedo = skin.expand(q.shape[0], 3).clone()
        for f in range(5):
            if f == 0:
                # a cor nasce em smoothstep no começo de 2→3, com borda suave sobre a palma
                ramp = (h[:, first_caps[0] + 1] / 0.3).clamp(0, 1)
                ramp = ramp * ramp * (3 - 2 * ramp)
                margin = palm_d - finger_d[0] + 0.025 * spec["palm"]
                w = (TINT * ramp * torch.sigmoid(margin / (0.012 * spec["palm"])))[:, None]
            else:
                ramp = ((h[:, first_caps[f]] + 0.1) / 0.25).clamp(0, 1)
                beyond = (torch.stack(fingers[f][1:], 0).min(0).values < fingers[f][0]).float()
                ramp = torch.maximum(ramp, beyond)
                # só a superfície do próprio dedo: a palma ao lado fica com a cor da pele
                surface = (finger_d[f] < palm_d + 0.008 * spec["palm"]).float()
                w = (TINT * soft[:, f + 1] * ramp * surface)[:, None]
            color = torch.tensor(_linear(COLORS[f]), device=device, dtype=torch.float32)
            # TINT pesa a luminância; a croma vem da cor do dedo, para que o vermelho da
            # pele não desvie o matiz
            tinted = (color * ((skin @ LUMA) / (color @ LUMA)).clamp(1, 2) ** 0.25).clamp(max=1)
            gray = (albedo @ LUMA)[:, None] * (1 - w) + (tinted @ LUMA) * w
            chroma_w = (w / TINT).clamp(0, 1)
            chroma = (albedo - (albedo @ LUMA)[:, None]) * (1 - chroma_w) + (
                tinted - tinted @ LUMA
            ) * chroma_w
            albedo = (gray + chroma).clamp(0, 1)
        # luz: principal + espalhamento no terminador + preenchimento quente + ambiente
        ndl = (normal * key).sum(-1)
        diffuse = ndl.clamp(min=0) * shadow
        wrap = ((ndl + 0.45) / 1.45).clamp(min=0) - ndl.clamp(min=0)
        sss = (
            torch.tensor([0.85, 0.32, 0.22], device=device) * (wrap * (0.4 + 0.6 * shadow))[:, None]
        )
        fill_dir = F.normalize(torch.tensor([0.7, 0.45, 0.45], device=device), dim=0)
        fill = (normal * fill_dir).sum(-1).clamp(min=0)[:, None] * deep * 0.55
        sky = (0.5 + 0.5 * normal[:, 2:3]) * torch.tensor([0.20, 0.20, 0.23], device=device)
        light = (
            diffuse[:, None] * torch.tensor([1.55, 1.46, 1.34], device=device)
            + sss * 0.55
            + fill * ao[:, None]
            + sky * ao[:, None]
        )
        color = (
            albedo * light * crevice[:, None] + deep * 0.35 * (1 - crevice[:, None]) * ao[:, None]
        )
        view = torch.tensor([0.0, 0.0, 1.0], device=device)
        half_v = F.normalize(key + view, dim=0)
        spec_term = (normal * half_v).sum(-1).clamp(min=0) ** 36 * 0.16 * shadow
        rim = (1 - normal[:, 2].clamp(0, 1)) ** 4 * 0.10
        color = color + (spec_term[:, None] + rim[:, None] * deep) * ao[:, None]
        color = 1 - torch.exp(-1.15 * color)  # tonemap suave
        rgb[idx] = color.clamp(0, 1) ** (1 / 2.2)
        image = torch.cat([rgb * alpha[:, None], alpha[:, None]], -1).reshape(n, n, 4)
        image = F.avg_pool2d(image.permute(2, 0, 1)[None], 2)[0].permute(1, 2, 0)
        a = image[..., 3:].clamp(min=1e-6)
        out = torch.cat([(image[..., :3] / a).clamp(0, 1), image[..., 3:]], -1)
    return out.cpu().numpy()


def measured_depth(config_path: Path, actor: str) -> np.ndarray:
    """(61, 21) z do MediaPipe em raios RMS (maior = mais perto); NaN onde não casa."""
    import sys

    import cv2

    sys.modules.setdefault("sounddevice", None)  # como em hand_detector
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )

    from research.config import load_config
    from research.data.canonical_hands import FRAME_MIRROR, FRAME_OVERRIDE
    from research.data.class_atlas import representative_keys
    from research.data.preview import _repo_path
    from research.ingestion.hand_detector import DEFAULT_MODEL
    from research.ingestion.lmdb_engine import LMDBStore, unpack_sample

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(DEFAULT_MODEL)),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
    )
    landmarker = HandLandmarker.create_from_options(options)
    config = load_config(config_path)
    data_config = load_config(_repo_path(config["paths"]["data_config"]))
    out = np.full((61, 21), np.nan)
    with LMDBStore(_repo_path(data_config["paths"]["lmdb_path"]), 1 << 30, readonly=True) as store:
        meta = store.read_meta()
        size = int(meta["layout"]["size"])
        keys = representative_keys(meta["samples"], actor)
        for code in range(61):
            key = FRAME_OVERRIDE.get(code + 1, keys[code])
            rgb, _, points = unpack_sample(store.read_sample(key.encode()), size)
            image = np.ascontiguousarray(np.asarray(rgb).transpose(1, 2, 0))
            points = points[:, :2].copy()
            if code + 1 in FRAME_MIRROR:  # como em canonical_hands: x → S − x
                image = np.ascontiguousarray(image[:, ::-1])
                points[:, 0] = size - points[:, 0]
            best = (np.inf, None)
            for scale in (1, 2):  # o recorte é pequeno: ampliado, o detector acha mais mãos
                if best[0] < MATCH_PX:
                    break
                view = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(view))
                result = landmarker.detect(frame)
                for marks, world in zip(result.hand_landmarks, result.hand_world_landmarks):
                    xy = np.array([(m.x * size, m.y * size) for m in marks])
                    error = float(np.linalg.norm(xy - points[:, :2], axis=1).mean())
                    if error < best[0]:
                        best = (error, np.array([(w.x, w.y, w.z) for w in world]))
            if best[0] < MATCH_PX:
                world = best[1] - best[1].mean(0)
                out[code] = -world[:, 2] / np.sqrt((world[:, :2] ** 2).sum(1).mean())
            else:
                print(f"CM {code + 1}: o MediaPipe não casou ({best[0]:.1f} px); z estimado")
    landmarker.close()
    return out


def hands_3d(xyz: np.ndarray, depth: np.ndarray) -> np.ndarray:
    from research.data.canonical_hands import palm_size

    lengths = bone_lengths(xyz[..., :2])
    hands = []
    for code, (points, z) in enumerate(zip(xyz, depth, strict=True)):
        if np.isnan(z).any():
            hands.append(lift(points, lengths))
        else:
            measured = np.c_[points[:, :2], z]
            hands.append(_off_palm(measured, float(palm_size(points[None, :, :2])[0])))
        palm = float(palm_size(points[None, :, :2])[0])
        hands[-1] = _fold_retract(hands[-1])
        if code + 1 in FIST_COMPACT:
            hands[-1] = _compact_fist(hands[-1], FIST_COMPACT[code + 1])
        if code + 1 in THUMB_TUCK:
            hands[-1] = _tuck_thumb(hands[-1], THUMB_TUCK[code + 1])
        if code + 1 in THUMB_BETWEEN:
            hands[-1] = _thumb_between(hands[-1], palm)
        if code + 1 in EVEN_FINGERS:
            hands[-1] = _even_fingers(hands[-1])
        if code + 1 in FINGER_PITCH:
            hands[-1] = _pitch_fingers(hands[-1], FINGER_PITCH[code + 1])
        if code + 1 in CROSSED:
            hands[-1] = _uncross(hands[-1], palm)
    return np.stack(hands)


def _pitch_fingers(p: np.ndarray, angle: float) -> np.ndarray:
    """Gira os 4 dedos em torno do próprio nó: o que apontava para a câmera sobe na tela."""
    p = p.copy()
    _, e2, _ = _frame(p)
    axis = np.cross([0.0, 0.0, 1.0], e2)  # leva +z para e2 (do pulso aos nós)
    axis /= np.linalg.norm(axis)
    cos, sin = np.cos(angle), np.sin(angle)
    for chain in CHAINS[1:]:
        v = p[list(chain[1:])] - p[chain[0]]
        rotated = v * cos + np.cross(axis, v) * sin + np.outer(v @ axis, axis) * (1 - cos)
        p[list(chain[1:])] = p[chain[0]] + rotated
    return p


def _behind(p: np.ndarray, joint: int, chain: tuple[int, ...]) -> tuple[float, float]:
    """Distância na tela da junta à cadeia e o z da cadeia nesse ponto mais próximo."""
    best = (np.inf, 0.0)
    for a, b in zip(chain, chain[1:], strict=False):
        ab = p[b, :2] - p[a, :2]
        t = float(np.clip((p[joint, :2] - p[a, :2]) @ ab / max(ab @ ab, 1e-9), 0, 1))
        d = float(np.linalg.norm(p[a, :2] + t * ab - p[joint, :2]))
        if d < best[0]:
            best = (d, float(p[a, 2] + t * (p[b, 2] - p[a, 2])))
    return best


def _uncross(p: np.ndarray, palm: float) -> np.ndarray:
    """Indicador e médio cruzados: onde se sobrepõem na tela, o da frente passa claramente
    à frente (`CROSS_GAP`), e a ponta do de trás sai de baixo dele no plano (`TIP_GAP`)."""
    p = p.copy()
    index, middle = CHAINS[1], CHAINS[2]
    overlap = 2 * 0.145 * palm  # dois raios de dedo
    pairs = [(j, middle) for j in index[2:]] + [(j, index) for j in middle[2:]]
    hits = [(j, other, *_behind(p, j, other)) for j, other in pairs]
    hits = [(j, other, z) for j, other, d, z in hits if d < overlap]
    if not hits:
        return p
    # quem já está à frente onde se cruzam continua à frente
    ahead = sum((p[j, 2] - z) * (1 if other is middle else -1) for j, other, z in hits)
    front = index if ahead >= 0 else middle
    back = middle if front is index else index
    lack = 0.0
    for j in front[2:]:
        d, z = _behind(p, j, back)
        if d < overlap:
            lack = max(lack, CROSS_GAP * palm - (p[j, 2] - z))
    p[list(front[2:]), 2] += lack
    # a ponta de trás sai de baixo da cadeia da frente, pelo lado em que já está
    tip = back[3]
    distance, _ = _behind(p, tip, front)
    if distance < TIP_GAP * palm:
        side = p[tip, :2] - _nearest(p, tip, front)
        if np.linalg.norm(side) < 1e-6:
            side = p[back[0], :2] - p[front[0], :2]
        shift = side / np.linalg.norm(side) * (TIP_GAP * palm - distance)
        p[list(back[2:]), :2] += np.outer([0.5, 1.0], shift)  # a distal inteira
    return p


def _nearest(p: np.ndarray, joint: int, chain: tuple[int, ...]) -> np.ndarray:
    """Ponto da cadeia, na tela, mais perto da junta."""
    best, point = np.inf, p[chain[0], :2]
    for a, b in zip(chain, chain[1:], strict=False):
        ab = p[b, :2] - p[a, :2]
        t = float(np.clip((p[joint, :2] - p[a, :2]) @ ab / max(ab @ ab, 1e-9), 0, 1))
        q = p[a, :2] + t * ab
        if np.linalg.norm(q - p[joint, :2]) < best:
            best, point = float(np.linalg.norm(q - p[joint, :2])), q
    return point


def build(hands: np.ndarray, device: str) -> np.ndarray:
    sheet = np.zeros((ROWS * TILE, COLUMNS * TILE, 4), np.uint8)
    for code, points in enumerate(hands):
        row, col = divmod(code, COLUMNS)
        cell = (render(points, device) * 255).round().astype(np.uint8)
        sheet[row * TILE : (row + 1) * TILE, col * TILE : (col + 1) * TILE] = cell
    return sheet


def main(argv: list[str] | None = None) -> None:
    import torch
    from PIL import Image

    parser = argparse.ArgumentParser(description="Sprite sheet 3D sintético das 61 mãos.")
    parser.add_argument("--config", type=Path, default=Path("configs/distillation.yaml"))
    parser.add_argument("--actor", default="C4")
    parser.add_argument("--out", type=Path, default=Path("mobile/assets/hands_61.webp"))
    parser.add_argument("--preview", type=Path, help="a mesma folha sobre fundo escuro, em PNG")
    parser.add_argument(
        "--cache", type=Path, help="npz com os pontos canônicos e o z medido (lê se existir)"
    )
    args = parser.parse_args(argv)
    if args.cache and args.cache.exists():
        cached = np.load(args.cache)
        xyz, depth = cached["xyz"], cached["depth"]
    else:
        from research.data.canonical_hands import compute

        xyz = compute(args.config, args.actor, None)[0]
        depth = measured_depth(args.config, args.actor)
        if args.cache:
            np.savez(args.cache, xyz=xyz, depth=depth)
    hands = hands_3d(xyz, depth)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sheet = build(hands, device)
    Image.fromarray(sheet, "RGBA").save(args.out, quality=92, method=6)
    if args.preview:
        back = Image.new("RGBA", (sheet.shape[1], sheet.shape[0]), (0x1E, 0x20, 0x24, 255))
        back.alpha_composite(Image.fromarray(sheet, "RGBA"))
        back.convert("RGB").save(args.preview)
    print(args.out, f"{args.out.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
