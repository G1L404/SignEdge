"""Aluno: MobileNetV4-Conv-Small (timm) com um ramo para os 21 pontos do MediaPipe.

O app já roda o MediaPipe para recortar a mão, então os pontos chegam de graça.
Eles entram em pixels da vista; o modelo os centra e divide pelo raio médio, o
que deixa o ramo indiferente à posição e ao tamanho da mão. A normalização do
ImageNet também fica dentro do modelo: a entrada é RGB em [0, 1].
"""

from __future__ import annotations

import torch
from torch import nn

from research.networks.teacher import IMAGENET_MEAN, IMAGENET_STD

N_LANDMARKS = 21


def normalize_landmarks(points: torch.Tensor) -> torch.Tensor:
    """(B, 21, 2) em pixels → (B, 42) centrado e em raios médios da mão."""
    centered = points - points.mean(dim=1, keepdim=True)
    radius = centered.pow(2).sum(-1).mean(-1).sqrt().clamp(min=1e-3)
    return (centered / radius[:, None, None]).flatten(1)


class LandmarkEncoder(nn.Module):
    def __init__(self, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * N_LANDMARKS, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(normalize_landmarks(points))


class Student(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        features: int,
        num_classes: int,
        landmarks: LandmarkEncoder | None,
        landmark_dropout: float,
        landmark_noise: float,
        dropout: float,
        aux_head: bool = False,
        rgb_dropout: float = 0.0,
    ) -> None:
        """[aux_head]: cabeça de 61 classes só no vetor dos pontos, treinada junto (perda
        auxiliar) e dobrada na cabeça final no export (`fold_aux`). [rgb_dropout]: zera o
        vetor da MobileNetV4 nessa fração dos exemplos de treino, forçando o ramo dos
        pontos a carregar a decisão sozinho às vezes."""
        super().__init__()
        self.backbone = backbone
        self.landmarks = landmarks
        self.landmark_dropout = landmark_dropout
        self.landmark_noise = landmark_noise
        extra = 0 if landmarks is None else landmarks.net[-2].out_features
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(features + extra, num_classes)
        self.aux_head = nn.Linear(extra, num_classes) if aux_head and extra else None
        self.rgb_dropout = rgb_dropout
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(
        self,
        rgb: torch.Tensor,
        landmarks: torch.Tensor | None = None,
        return_parts: bool = False,
        **_,
    ):
        """Logits; com [return_parts], (logits, vetor fundido, logits auxiliares ou None)."""
        feats = self.backbone((rgb - self.mean) / self.std)
        if self.training and self.rgb_dropout > 0:
            keep = torch.rand(feats.shape[0], 1, device=feats.device) >= self.rgb_dropout
            feats = feats * keep
        aux = None
        if self.landmarks is not None:
            if landmarks is None:
                raise ValueError("aluno com ramo dos pontos chamado sem pontos")
            if self.training and self.landmark_noise > 0:
                scale = landmarks.std(dim=1, keepdim=True).mean(-1, keepdim=True)
                landmarks = landmarks + torch.randn_like(landmarks) * scale * self.landmark_noise
            code = self.landmarks(landmarks)
            if self.training and self.landmark_dropout > 0:
                keep = torch.rand(code.shape[0], 1, device=code.device) >= self.landmark_dropout
                code = code * keep
            if self.aux_head is not None:
                aux = self.aux_head(code)
            feats = torch.cat([feats, code], dim=1)
        logits = self.head(self.dropout(feats))
        return (logits, feats, aux) if return_parts else logits

    @torch.no_grad()
    def fold_aux(self, weight: float) -> None:
        """Dobra `weight` × cabeça auxiliar na cabeça final e a remove.

        As duas leem o mesmo vetor dos pontos, então
        head([f, h]) + w·aux(h) = [W_rgb | W_lm + w·W_aux]·[f, h] + (b + w·b_aux):
        a mesma conta, uma única nn.Linear, o grafo do app sem nenhuma op a mais.
        """
        if self.aux_head is None:
            return
        extra = self.aux_head.in_features
        self.head.weight[:, -extra:] += weight * self.aux_head.weight
        self.head.bias += weight * self.aux_head.bias
        self.aux_head = None

    def head_parameters(self):
        return self.head.parameters()


class LandmarkBaseline(nn.Module):
    """Só os 21 pontos: o piso que os pixels precisam superar."""

    def __init__(self, num_classes: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.encoder = LandmarkEncoder(hidden, dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, landmarks: torch.Tensor, **_) -> torch.Tensor:
        return self.head(self.encoder(landmarks))


def build_student(cfg: dict, num_classes: int, pretrained: bool = True) -> Student:
    """`cfg` é `model.student` de configs/distillation.yaml."""
    import timm

    backbone = timm.create_model(cfg["arch"], pretrained=pretrained, num_classes=0)
    encoder = LandmarkEncoder(int(cfg["landmark_hidden"])) if cfg["landmarks"] else None
    features = getattr(backbone, "head_hidden_size", None) or backbone.num_features
    return Student(
        backbone,
        features,
        num_classes,
        encoder,
        float(cfg["landmark_dropout"]),
        float(cfg["landmark_noise"]),
        float(cfg["dropout"]),
        aux_head=bool(cfg.get("aux_head", False)),
        rgb_dropout=float(cfg.get("rgb_dropout", 0.0)),
    )


def build_landmark_baseline(cfg: dict, num_classes: int) -> LandmarkBaseline:
    """`cfg` é `model.landmark_baseline`."""
    return LandmarkBaseline(num_classes, int(cfg["hidden"]), float(cfg["dropout"]))
