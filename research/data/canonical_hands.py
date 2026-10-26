"""Esqueleto canônico de cada CM para o app: os 21 pontos do quadro do atlas.

Para cada CM, os pontos do mesmo quadro de `figuras/atlas.jpg` (o central da primeira
pausa do sinalizante C4, `representative_keys`), normalizados pelo centroide e pelo
raio RMS (a normalização do ramo geométrico). Assim o catálogo do app mostra a mesma
mão da Figura 1.1 do capítulo. `--medoide` usa, em vez dele, o medoide em C1–C5: o
quadro real com a menor mudança de forma média (ΔS) para os demais da CM.

`FRAME_OVERRIDE` troca o quadro do atlas de uma CM por outro, nomeado:

- CM 10: no quadro de C4 o detector põe as juntas 6, 7 e 8 no alto do polegar e o
  indicador parece erguido. Vale o medoide dos quadros com os quatro dedos dobrados
  (ponta mais perto do punho que 1,15 × a base) e a ponta do indicador na altura da
  do médio: 48 de 95, de C1 a C4.
- CM 11: o mesmo erro no quadro de C4. Vale o medoide dos quadros com os quatro dedos
  dobrados, o indicador na altura do médio e a ponta do polegar perto do meio de 6–10:
  só 6 de 93, todos da mesma pausa de C2.
- CM 25: no quadro de C4 a mão está de perfil pela borda ulnar, com os nós 5, 9 e 13
  quase no mesmo ponto. Vale o quadro de C1–C5 com a
  lateralidade da maioria, os outros três dedos fechados e a menor distância entre as
  pontas do indicador e do polegar (0,04 palma): a pinça em gota.

`FRAME_MIRROR` espelha na horizontal o quadro de uma CM (x → S − x), para o polegar
ficar à esquerda como nas outras: só a CM 26, cujo quadro do atlas tem a mão invertida.

O sinal de (5 − 0) × (17 − 0) só escolhe entre quadros: de perfil ele troca de quadro
para quadro (53 de 90 na CM 25), sem erro do detector.

A postura de cada CM vem de `REVIEWED`, revisada no atlas; `describe` (estado de cada
dedo pela distância da ponta ao punho relativa à da base) fica para CMs sem revisão.
Cada ponto ganha ainda um `z` (maior = mais perto da câmera), em raios RMS: a
profundidade do Kinect no quadro (mediana 5 × 5 restrita à mão, relativa à palma, a
~50 mm por raio) somada ao avanço estimado pela flexão (o encurtamento 2D de cada
falange contra o maior comprimento dela nas 61 CMs vira avanço para a câmera,
acumulado da base à ponta). É uma estimativa para o desenho, não uma medida 3D.
Sem revisão linguística. Saída: `mobile/lib/src/logic/canonical_hands.dart`.

    python -m research.data.canonical_hands --config configs/distillation.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FINGERS = {"indicador": (5, 8), "médio": (9, 12), "anelar": (13, 16), "mínimo": (17, 20)}
EXTENDED, FLEXED = 1.7, 1.25
CHAINS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
PALM_POINTS = (0, 5, 9, 13, 17)
MM_PER_RADIUS = 50.0
HAND_BAND_MM = 150.0
Z_LIMIT = 1.5
STRAIGHT = 0.8  # até 20% de encurtamento 2D, a falange conta como reta


# CM → chave do quadro que substitui o do atlas
FRAME_OVERRIDE = {10: "C1_2_9_p0_f018", 11: "C2_2_10_p1_f115", 25: "C3_2_24_p1_f049"}
FRAME_MIRROR = {26}  # CMs com o quadro espelhado na horizontal

# Posturas revisadas no atlas (figuras/atlas.jpg, C4) onde a leitura 2D erra por
# auto-oclusão ou deixa CMs vizinhas com o mesmo texto.
REVIEWED = {
    1: "Mão fechada; polegar encostado na lateral do indicador",
    2: "Mão fechada; polegar estendido para o lado, na horizontal",
    3: "Mínimo estendido para cima; polegar aberto; demais fechados",
    4: "Polegar e mínimo bem abertos para os lados; demais fechados",
    5: "Mínimo estendido; polegar sobre os dedos fechados",
    6: "Mínimo curvo em gancho; polegar sobre os dedos fechados",
    7: "Mão fechada; polegar cruzado à frente dos dedos dobrados",
    8: "Quatro dedos dobrados nas articulações do meio; polegar sob as pontas",
    9: "Mão fechada; polegar estendido para cima e para o lado",
    10: "Quatro dedos dobrados na mesma altura, sobre o polegar",
    11: "Dedos dobrados; ponta do polegar entre o indicador e o médio",
    12: "Indicador erguido e levemente curvo; demais semifechados, polegar sobre o médio",
    13: "Indicador estendido; demais fechados, polegar sobre o médio",
    14: "Indicador estendido; polegar toca a ponta do médio dobrado",
    15: "Indicador estendido; polegar meio aberto, afastado da mão",
    16: "Indicador estendido; polegar cruzado à frente do médio e do anelar",
    17: "Mão de lado; indicador dobrado toca o polegar, demais dobrados atrás",
    18: "Indicador e polegar estendidos e paralelos, em pinça aberta; demais fechados",
    19: "Indicador em gancho acima do polegar aberto; demais fechados",
    20: "Pontas do polegar e do indicador formam um pequeno círculo; demais fechados",
    21: "Pontas de todos os dedos tocam a do polegar, em bico",
    22: "Dedos curvos em gancho sobre o polegar, formando um O",
    23: "Indicador e médio curvos tocam o polegar; anelar e mínimo dobrados",
    24: "Indicador e médio estendidos, juntos e inclinados; polegar toca o médio",
    25: "Indicador curvo tocando a ponta do polegar em gota; demais fechados",
    26: "Dedos juntos e retos, dobrados na base, com as pontas sobre o polegar",
    27: "Mão em C: dedos curvos e juntos, polegar oposto e afastado",
    28: "Mão em C fechado: dedos curvos quase tocam o polegar",
    29: "Mão plana de lado, dedos juntos; polegar afastado para baixo",
    30: "Dedos juntos e retos, dobrados em ângulo reto na base; polegar aberto",
    31: "Dedos abertos em garra, indicador e médio mais erguidos; polegar aberto",
    32: "Indicador estendido; demais dedos curvos em garra; polegar aberto",
    33: "Indicador e médio estendidos e juntos; polegar sobre anelar e mínimo",
    34: "Indicador e médio estendidos e cruzados; polegar sobre anelar e mínimo",
    35: "Indicador e médio estendidos e cruzados; polegar aberto para o lado",
    36: "Indicador e médio estendidos e afastados; polegar aberto para o lado",
    37: "Indicador e médio estendidos, o médio à frente; polegar aberto",
    38: "Indicador estendido para cima e polegar aberto para o lado, em L; demais fechados",
    39: "Indicador e mínimo estendidos; polegar sobre médio e anelar dobrados",
    40: "Indicador e mínimo estendidos; polegar aberto para o lado",
    41: "Mão aberta e afastada, com o médio dobrado à frente",
    42: "Indicador, anelar e mínimo estendidos; médio dobrado toca o polegar",
    43: "Indicador, anelar e mínimo bem afastados; médio e polegar dobrados à frente da palma",
    44: "Polegar e indicador em círculo; médio, anelar e mínimo estendidos e afastados",
    45: "Pinça entre polegar e indicador curvo; médio, anelar e mínimo estendidos",
    46: "Indicador dobrado sobre o polegar; médio, anelar e mínimo estendidos e juntos",
    47: "Polegar toca a ponta do indicador; médio, anelar e mínimo estendidos e afastados",
    48: "Indicador e médio estendidos, levemente afastados; polegar cruzado sobre anelar e mínimo",
    49: "Indicador e médio estendidos em V; polegar sobre anelar e mínimo",
    50: "Indicador e médio em V; polegar apoiado na base do médio",
    51: "Indicador, médio e anelar estendidos e levemente afastados; polegar sobre o mínimo",
    52: "Indicador, médio e anelar estendidos e bem juntos; polegar sobre o mínimo",
    53: "Quatro dedos estendidos e juntos; polegar dobrado sobre a palma",
    54: "Quatro dedos estendidos e afastados; polegar dobrado sobre a palma",
    55: "Quatro dedos afastados, o mínimo curvo; polegar dobrado sobre a palma",
    56: "Quatro dedos estendidos e juntos; polegar estendido junto ao indicador",
    57: "Quatro dedos estendidos e juntos; polegar aberto para o lado",
    58: "Mão em concha: dedos juntos e levemente curvos; polegar aberto",
    59: "Dedos afastados e curvos em garra; polegar aberto",
    60: "Mão aberta, dedos afastados; polegar aberto em ângulo reto",
    61: "Mão aberta e inclinada, dedos bem afastados; polegar estendido para baixo",
}


def normalize(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    radius = np.sqrt((centered**2).sum(axis=1).mean())
    return centered / radius if radius > 0 else centered


def medoid(shapes: np.ndarray) -> int:
    """Índice do quadro com a menor ΔS média para os outros (shapes: N × 21 × 2)."""
    diff = np.linalg.norm(shapes[:, None] - shapes[None], axis=-1).mean(-1)
    return int(diff.mean(1).argmin())


def finger_state(points: np.ndarray, base: int, tip: int) -> str:
    ratio = np.linalg.norm(points[tip] - points[0]) / np.linalg.norm(points[base] - points[0])
    if ratio >= EXTENDED:
        return "estendido"
    return "flexionado" if ratio <= FLEXED else "semiflexionado"


def thumb_state(points: np.ndarray) -> str:
    palm = np.linalg.norm(points[5] - points[17])
    if np.linalg.norm(points[4] - points[8]) < 0.35 * palm:
        return "polegar toca o indicador"
    if np.linalg.norm(points[4] - points[5]) > 0.9 * palm:
        return "polegar aberto"
    return "polegar junto à mão"


def describe(points: np.ndarray) -> str:
    groups: dict[str, list[str]] = {}
    for name, (base, tip) in FINGERS.items():
        groups.setdefault(finger_state(points, base, tip), []).append(name)
    parts = []
    for state in ("estendido", "semiflexionado", "flexionado"):
        names = groups.get(state)
        if not names:
            continue
        if len(names) == 4:
            parts.append(f"quatro dedos {state}s")
        else:
            joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " e " + names[-1]
            parts.append(f"{joined} {state}{'s' if len(names) > 1 else ''}")
    parts.append(thumb_state(points))
    text = "; ".join(parts)
    return text[0].upper() + text[1:]


def sensor_depth(depth: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Profundidade (mm) em cada ponto: mediana 5 × 5 dos pixels válidos da mão."""
    raw = []
    for x, y in np.rint(points).astype(int):
        window = depth[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3].astype(float)
        raw.append(window[window > 0])
    palm = np.median(np.concatenate([raw[i] for i in PALM_POINTS if raw[i].size] or [[0.0]]))
    out = np.full(21, np.nan)
    for i, values in enumerate(raw):
        near = values[np.abs(values - palm) < HAND_BAND_MM]
        if near.size:
            out[i] = np.median(near)
    return palm - np.nan_to_num(out, nan=palm)


def palm_size(hands: np.ndarray) -> np.ndarray:
    """Tamanho da palma em cada quadro: a média de 0–5, 0–17 e 5–17, que quase não muda."""
    pairs = ((0, 5), (0, 17), (5, 17))
    return np.mean([np.linalg.norm(hands[:, a] - hands[:, b], axis=1) for a, b in pairs], axis=0)


def flexion_depth(hands: np.ndarray) -> np.ndarray:
    """Avanço para a câmera pela flexão: encurtamento 2D de cada falange, acumulado.

    O comprimento de cada falange é medido em palmas do próprio quadro (a normalização
    pelo raio RMS encolhe os ossos quando a mão abre), e o comprimento inteiro dela é
    o percentil 95 nas 61 CMs. Até 20% de encurtamento (perspectiva e ruído do
    detector), a falange conta como reta.
    """
    palm = palm_size(hands)
    z = np.zeros(hands.shape[:2])
    for chain in CHAINS:
        for a, b in zip(chain, chain[1:], strict=False):
            length = np.linalg.norm(hands[:, b] - hands[:, a], axis=1) / palm
            full = np.percentile(length, 95)
            seen = np.clip(length / (STRAIGHT * full), 0, 1)
            z[:, b] = z[:, a] + full * np.sqrt(1 - seen**2) * palm
    return z


def compute(
    config_path: Path,
    actor: str,
    medoid_actors: list[str] | None,
) -> tuple:
    from research.config import load_config
    from research.data.class_atlas import representative_keys
    from research.data.preview import _repo_path
    from research.ingestion.lmdb_engine import LMDBStore, unpack_sample

    config = load_config(config_path)
    data_config = load_config(_repo_path(config["paths"]["data_config"]))
    hands = np.zeros((61, 21, 2))
    sensor = np.zeros((61, 21))
    with LMDBStore(_repo_path(data_config["paths"]["lmdb_path"]), 1 << 30, readonly=True) as store:
        meta = store.read_meta()
        size = int(meta["layout"]["size"])

        def read(key: str, mirror: bool = False) -> tuple[np.ndarray, np.ndarray]:
            _, depth, points = unpack_sample(store.read_sample(key.encode()), size)
            sensor = sensor_depth(depth, points[:, :2]) / MM_PER_RADIUS  # independe do espelho
            xy = points[:, :2].copy()
            if mirror:
                xy[:, 0] = size - xy[:, 0]
            return normalize(xy), sensor

        if medoid_actors:
            shapes: dict[int, list[str]] = {}
            for key, sample in meta["samples"].items():
                if sample["actor"] in medoid_actors:
                    shapes.setdefault(int(sample["cls"]), []).append(key)
            for code in range(61):
                frames = [read(k) for k in shapes[code]]
                best = medoid(np.stack([f[0] for f in frames]))
                hands[code], sensor[code] = frames[best]
        else:
            keys = representative_keys(meta["samples"], actor)
            for code in range(61):
                key = FRAME_OVERRIDE.get(code + 1, keys[code])
                hands[code], sensor[code] = read(key, code + 1 in FRAME_MIRROR)
                if code + 1 in FRAME_MIRROR:
                    print(f"CM {code + 1}: quadro {key} espelhado")
                if key != keys[code]:
                    print(f"CM {code + 1}: quadro {key} no lugar do atlas ({keys[code]})")
    z = np.clip(flexion_depth(hands) + sensor, -Z_LIMIT, Z_LIMIT)
    xyz = np.concatenate([hands, z[..., None]], axis=-1)
    return xyz, [REVIEWED.get(code + 1) or describe(hands[code]) for code in range(61)]


def to_dart(hands: np.ndarray, postures: list[str]) -> str:
    rows = ["  [" + ", ".join(f"{v:.3f}" for v in hand.reshape(-1)) + "]," for hand in hands]
    texts = [f"  '{p}'," for p in postures]
    return "\n".join(
        [
            "// Gerado por research/data/canonical_hands.py; não editar à mão.",
            "//",
            "// Esqueleto canônico de cada CM: os 21 pontos (x, y) do quadro do atlas",
            "// (figuras/atlas.jpg, C4), normalizados pelo centroide e pelo raio RMS,",
            "// com y para baixo, e z estimado (maior = mais perto da câmera): 63 valores",
            "// por CM. Posturas revisadas no atlas, sem revisão linguística.",
            "library;",
            "",
            "const List<List<double>> canonicalHands = [",
            *rows,
            "];",
            "",
            "const List<String> canonicalPostures = [",
            *texts,
            "];",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Esqueleto canônico das 61 CMs para o app.")
    parser.add_argument("--config", type=Path, default=Path("configs/distillation.yaml"))
    parser.add_argument("--actor", default="C4", help="sinalizante do atlas")
    parser.add_argument(
        "--medoide", nargs="+", metavar="ATOR", help="usa o medoide desses atores (ex.: C1 … C5)"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("mobile/lib/src/logic/canonical_hands.dart")
    )
    args = parser.parse_args(argv)
    hands, postures = compute(args.config, args.actor, args.medoide)
    args.out.write_text(to_dart(hands, postures), encoding="utf-8")
    for code, posture in enumerate(postures):
        print(f"CM {code + 1:2d}: {posture}")


if __name__ == "__main__":
    main()
