"""Professor: ConvNeXt-Tiny ou ResNet34 do ImageNet, com RGB e, opcionalmente, profundidade.

A profundidade relativa entra como 4º canal do stem. Os pesos desse canal nascem
como a média dos pesos RGB, de modo que o pré-treino continua valendo no começo.
A normalização do ImageNet fica dentro do modelo: a entrada é RGB em [0, 1].
"""

from __future__ import annotations

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _widen_stem(conv: nn.Conv2d) -> nn.Conv2d:
    wide = nn.Conv2d(
        4,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        wide.weight[:, :3] = conv.weight
        wide.weight[:, 3:] = conv.weight.mean(dim=1, keepdim=True)
        if conv.bias is not None:
            wide.bias.copy_(conv.bias)
    return wide


class Teacher(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Linear, use_depth: bool) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.use_depth = use_depth
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor | None = None,
        return_features: bool = False,
        **_,
    ):
        """Logits; com [return_features], (logits, vetor penúltimo)."""
        x = (rgb - self.mean) / self.std
        if self.use_depth:
            if depth is None:
                raise ValueError("professor RGB-D chamado sem profundidade")
            x = torch.cat([x, depth], dim=1)
        features = self.backbone(x)
        logits = self.head(features)
        return (logits, features) if return_features else logits

    def head_parameters(self):
        return self.head.parameters()


def build_teacher(architecture: str, num_classes: int, depth: bool, pretrained: bool = True):
    import torchvision

    if architecture == "convnext_tiny":
        weights = "IMAGENET1K_V1" if pretrained else None
        net = torchvision.models.convnext_tiny(weights=weights)
        if depth:
            net.features[0][0] = _widen_stem(net.features[0][0])
        features = net.classifier[2].in_features
        net.classifier[2] = nn.Identity()
        backbone = nn.Sequential(net.features, net.avgpool, net.classifier)
    elif architecture == "resnet34":
        weights = "IMAGENET1K_V1" if pretrained else None
        net = torchvision.models.resnet34(weights=weights)
        if depth:
            net.conv1 = _widen_stem(net.conv1)
        features = net.fc.in_features
        net.fc = nn.Identity()
        backbone = net
    else:
        raise ValueError(f"professor desconhecido: {architecture}")
    return Teacher(backbone, nn.Linear(features, num_classes), depth)
