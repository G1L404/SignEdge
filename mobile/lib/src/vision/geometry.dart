/// Geometria do recorte: o mesmo contrato do Python (crop.py + EvalView).
///
/// Coordenadas são contínuas: o pixel i cobre [i, i + 1), como os pontos do
/// MediaPipe. Uma [Affine] leva coordenadas da saída (pixels de um tensor de
/// lado `size`) para coordenadas da imagem.
library;

import 'dart:math' as math;

/// Ponto 2D em pixels contínuos.
class Pt {
  const Pt(this.x, this.y);
  final double x;
  final double y;
}

/// x_img = a·u + b·v + c, y_img = d·u + e·v + f.
class Affine {
  const Affine(this.a, this.b, this.c, this.d, this.e, this.f);

  final double a, b, c, d, e, f;

  Pt apply(double u, double v) => Pt(a * u + b * v + c, d * u + e * v + f);

  /// Quadrado de lado [side] (pixels da imagem) centrado em ([cx], [cy]),
  /// girado de [rotation] radianos, amostrado num tensor de lado [size].
  factory Affine.square(
    double cx,
    double cy,
    double side,
    double rotation,
    int size,
  ) {
    final cos = math.cos(rotation), sin = math.sin(rotation);
    final k = side / size, half = size / 2;
    return Affine(
      cos * k,
      -sin * k,
      cx - (cos * half - sin * half) * k,
      sin * k,
      cos * k,
      cy - (sin * half + cos * half) * k,
    );
  }

  /// Compõe: primeiro `this` (saída → imagem), depois [then] (imagem → outra).
  Affine andThen(Affine then) => Affine(
    then.a * a + then.b * d,
    then.a * b + then.b * e,
    then.a * c + then.b * f + then.c,
    then.d * a + then.e * d,
    then.d * b + then.e * e,
    then.d * c + then.e * f + then.f,
  );
}

/// Parâmetros da caixa da mão, iguais a configs/data_pipeline.yaml e distillation.yaml.
class CropRule {
  static const double handScale = 1.4;
  static const double minBoxFrac = 0.233333;
  static const double viewContext = 1.5;
  static const int inputSize = 224;
}

/// Caixa justa da mão: quadrado centrado nos 21 pontos (hand_detector.hand_square).
///
/// Lado = maior extensão × 1,4, nunca abaixo de 0,2333 × o lado menor da foto
/// (no Kinect, 112 de 480 px). Devolve (x0, y0, lado) inteiros, como no Python.
({int x0, int y0, int side}) handSquare(List<Pt> points, int frameShortSide) {
  var minX = double.infinity, minY = double.infinity;
  var maxX = -double.infinity, maxY = -double.infinity;
  for (final p in points) {
    minX = math.min(minX, p.x);
    minY = math.min(minY, p.y);
    maxX = math.max(maxX, p.x);
    maxY = math.max(maxY, p.y);
  }
  final span = math.max(maxX - minX, maxY - minY);
  final minSide = _roundHalfEven(frameShortSide * CropRule.minBoxFrac);
  final side = math.max(
    math.max(_roundHalfEven(span * CropRule.handScale), minSide),
    1,
  );
  final x0 = _roundHalfEven((minX + maxX) / 2 - side / 2);
  final y0 = _roundHalfEven((minY + maxY) / 2 - side / 2);
  return (x0: x0, y0: y0, side: side);
}

/// A vista do modelo (EvalView): lado = 1,5 × caixa justa, centro = meio da
/// extensão dos pontos, sem giro nem espelho.
class ModelView {
  ModelView._(this.center, this.side, this.box);

  factory ModelView.fromLandmarks(List<Pt> points, int frameShortSide) {
    var minX = double.infinity, minY = double.infinity;
    var maxX = -double.infinity, maxY = -double.infinity;
    for (final p in points) {
      minX = math.min(minX, p.x);
      minY = math.min(minY, p.y);
      maxX = math.max(maxX, p.x);
      maxY = math.max(maxY, p.y);
    }
    final box = handSquare(points, frameShortSide);
    return ModelView._(
      Pt((minX + maxX) / 2, (minY + maxY) / 2),
      CropRule.viewContext * box.side,
      box,
    );
  }

  final Pt center;
  final double side;
  final ({int x0, int y0, int side}) box;

  Affine get affine =>
      Affine.square(center.x, center.y, side, 0, CropRule.inputSize);

  /// Pontos da imagem para pixels da vista 224 (augment.map_landmarks).
  List<Pt> mapLandmarks(List<Pt> points) {
    final k = CropRule.inputSize / side, half = CropRule.inputSize / 2;
    return [
      for (final p in points)
        Pt((p.x - center.x) * k + half, (p.y - center.y) * k + half),
    ];
  }
}

/// round() do Python 3: metade vai para o par.
int _roundHalfEven(double value) {
  final floor = value.floorToDouble();
  final diff = value - floor;
  if (diff > 0.5) return floor.toInt() + 1;
  if (diff < 0.5) return floor.toInt();
  return floor.toInt().isEven ? floor.toInt() : floor.toInt() + 1;
}
