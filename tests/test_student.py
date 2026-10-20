import torch
from research.networks.student import LandmarkEncoder, Student
from torch import nn


def _student(**kwargs) -> Student:
    torch.manual_seed(0)
    backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 16))
    return Student(backbone, 16, 5, LandmarkEncoder(8), 0.0, 0.0, 0.0, **kwargs).eval()


def _inputs():
    torch.manual_seed(1)
    return torch.rand(4, 3, 8, 8), torch.rand(4, 21, 2) * 224


def test_fold_aux_is_the_same_sum_in_one_linear():
    model = _student(aux_head=True)
    rgb, points = _inputs()
    logits, _, aux = model(rgb, points, return_parts=True)
    model.fold_aux(0.3)
    assert model.aux_head is None
    torch.testing.assert_close(model(rgb, points), logits + 0.3 * aux)


def test_rgb_dropout_only_in_training():
    model = _student(rgb_dropout=1.0)
    rgb, points = _inputs()
    with torch.no_grad():
        evaluated = model(rgb, points)
        model.train()
        _, fused, _ = model(rgb, points, return_parts=True)
    assert torch.all(fused[:, :16] == 0)  # treino: o vetor RGB zerado
    model.rgb_dropout = 0.0
    with torch.no_grad():
        torch.testing.assert_close(evaluated, model.eval()(rgb, points))  # avaliação: intacto
