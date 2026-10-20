/// Pós-processamento do MediaPipe Hands, em Dart puro.
///
/// Port da referência numpy validada contra o HandLandmarker oficial (erro
/// mediano de 1,3 px em frames de 288 px): âncoras SSD do detector de palma,
/// NMS ponderado, ROI girada (palma: escala 2,6, desvio −0,5; pontos: escala
/// 2,0, desvio −0,1) e a volta dos 21 pontos para a imagem.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'geometry.dart';

const int palmInput = 192;
const int landmarkInput = 224;
const double palmScoreThreshold = 0.5;

/// A presença do modelo de pontos já sai como probabilidade (0,99 com mão, ~0,01
/// sem): não passa por sigmoid. Com sigmoid, tudo ficava ≥ 0,5 e o rastreio
/// nunca soltava um falso positivo.
const double handPresenceThreshold = 0.5;

/// Menor extensão dos 21 pontos, em fração do lado menor do frame (~19 px em 480).
/// Pontos colapsados geram uma ROI ainda menor no frame seguinte, e o rastreio se
/// prende num zoom de fundo; abaixo disso, a mão é dada como perdida.
const double minHandSpanFrac = 0.04;

/// Os pontos valem como mão: presença acima do limiar e tamanho plausível.
bool plausibleHand(double presence, List<Pt> points, int frameShortSide) {
  if (presence < handPresenceThreshold) return false;
  var minX = double.infinity, minY = double.infinity;
  var maxX = -double.infinity, maxY = -double.infinity;
  for (final p in points) {
    minX = math.min(minX, p.x);
    minY = math.min(minY, p.y);
    maxX = math.max(maxX, p.x);
    maxY = math.max(maxY, p.y);
  }
  return math.max(maxX - minX, maxY - minY) >= minHandSpanFrac * frameShortSide;
}

/// Quadrado girado na imagem: centro, lado em pixels e rotação em radianos.
class Roi {
  const Roi(this.cx, this.cy, this.side, this.rotation);
  final double cx, cy, side, rotation;

  Affine affine(int size) => Affine.square(cx, cy, side, rotation, size);
}

/// As 2016 âncoras do detector de palma de 192 px: grade 24×24 com 2 por célula
/// (passo 8) e 12×12 com 6 (as três camadas de passo 16 juntas).
final Float32List palmAnchors = () {
  final out = <double>[];
  for (final (stride, perCell) in [(8, 2), (16, 6)]) {
    final n = palmInput ~/ stride;
    for (var y = 0; y < n; y++) {
      for (var x = 0; x < n; x++) {
        for (var k = 0; k < perCell; k++) {
          out
            ..add((x + 0.5) / n)
            ..add((y + 0.5) / n);
        }
      }
    }
  }
  return Float32List.fromList(out);
}();

double sigmoid(double x) => 1 / (1 + math.exp(-x.clamp(-100.0, 100.0)));

/// Letterbox do detector: o quadrado do lado maior do frame, centrado.
Affine palmLetterbox(int width, int height) {
  final side = math.max(width, height).toDouble();
  return Affine.square(width / 2, height / 2, side, 0, palmInput);
}

Roi _transform(
  double cx,
  double cy,
  double w,
  double h,
  double rotation,
  double scale,
  double shiftY,
) {
  final x = cx - h * shiftY * math.sin(rotation);
  final y = cy + h * shiftY * math.cos(rotation);
  return Roi(x, y, math.max(w, h) * scale, rotation);
}

double _rotationTo(double x0, double y0, double x1, double y1) =>
    math.pi / 2 - math.atan2(-(y1 - y0), x1 - x0);

/// Melhor palma do frame ([raw] 2016×18, [scores] 2016 logits) já como ROI dos
/// pontos, em pixels do frame. Null se nenhuma passa de 0,5.
({Roi roi, double score})? decodePalm(
  Float32List raw,
  Float32List scores,
  int width,
  int height,
) {
  const stride = 18;
  var best = -1;
  var bestScore = palmScoreThreshold;
  for (var i = 0; i < scores.length; i++) {
    final s = sigmoid(scores[i]);
    if (s >= bestScore) {
      bestScore = s;
      best = i;
    }
  }
  if (best < 0) return null;

  List<double> box(int i) {
    final ax = palmAnchors[2 * i], ay = palmAnchors[2 * i + 1];
    final o = i * stride;
    final cx = raw[o] / palmInput + ax, cy = raw[o + 1] / palmInput + ay;
    final w = raw[o + 2] / palmInput, h = raw[o + 3] / palmInput;
    return [
      cx - w / 2,
      cy - h / 2,
      cx + w / 2,
      cy + h / 2,
      for (var k = 0; k < 7; k++) ...[
        raw[o + 4 + 2 * k] / palmInput + ax,
        raw[o + 5 + 2 * k] / palmInput + ay,
      ],
    ];
  }

  // NMS ponderado: média, pelo score, das caixas que cobrem a melhor (IoU > 0,3).
  final ref = box(best);
  final acc = List<double>.filled(18, 0);
  var total = 0.0;
  for (var i = 0; i < scores.length; i++) {
    final s = sigmoid(scores[i]);
    if (s < palmScoreThreshold) continue;
    final b = box(i);
    final ix =
        math.max(0.0, math.min(b[2], ref[2]) - math.max(b[0], ref[0])) *
        math.max(0.0, math.min(b[3], ref[3]) - math.max(b[1], ref[1]));
    final union =
        (b[2] - b[0]) * (b[3] - b[1]) +
        (ref[2] - ref[0]) * (ref[3] - ref[1]) -
        ix;
    if (union <= 0 || ix / union <= 0.3) continue;
    for (var k = 0; k < 18; k++) {
      acc[k] += b[k] * s;
    }
    total += s;
  }
  final b = [for (final value in acc) value / total];

  // Do quadrado de 192 normalizado para pixels do frame.
  final side = math.max(width, height).toDouble();
  final padX = (side - width) / 2, padY = (side - height) / 2;
  double px(double value) => value * side - padX;
  double py(double value) => value * side - padY;
  final x0 = px(b[0]), y0 = py(b[1]), x1 = px(b[2]), y1 = py(b[3]);
  // Ponto-chave 0 = punho, 2 = base do dedo médio.
  final rotation = _rotationTo(px(b[4]), py(b[5]), px(b[8]), py(b[9]));
  final roi = _transform(
    (x0 + x1) / 2,
    (y0 + y1) / 2,
    x1 - x0,
    y1 - y0,
    rotation,
    2.6,
    -0.5,
  );
  return (roi: roi, score: bestScore);
}

/// ROI do próximo frame a partir dos pontos deste (rastreamento sem palma).
Roi roiFromLandmarks(List<Pt> points) {
  const partial = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18];
  final p = [for (final i in partial) points[i]];
  final x1 = ((p[4].x + p[8].x) / 2 + p[6].x) / 2;
  final y1 = ((p[4].y + p[8].y) / 2 + p[6].y) / 2;
  final rotation = _rotationTo(p[0].x, p[0].y, x1, y1);
  final cos = math.cos(rotation), sin = math.sin(rotation);
  var minX = double.infinity, minY = double.infinity;
  var maxX = -double.infinity, maxY = -double.infinity;
  for (final q in p) {
    // Gira de −rotation para alinhar a mão aos eixos.
    final rx = q.x * cos + q.y * sin;
    final ry = -q.x * sin + q.y * cos;
    minX = math.min(minX, rx);
    maxX = math.max(maxX, rx);
    minY = math.min(minY, ry);
    maxY = math.max(maxY, ry);
  }
  final mx = (minX + maxX) / 2, my = (minY + maxY) / 2;
  final cx = mx * cos - my * sin, cy = mx * sin + my * cos;
  return _transform(cx, cy, maxX - minX, maxY - minY, rotation, 2.0, -0.1);
}

/// Amortecimento da ROI de rastreio. A ROI do frame t+1 sai dos pontos crus do t:
/// sem isso, uma oscilação pequena nos pontos move a ROI, que faz os pontos pularem
/// de volta, e o esqueleto treme em realimentação com a mão parada.
const double roiDeadbandPx = 3;
const double roiDeadbandScale = 0.03;
const double roiDeadbandAngle = 0.05; // ~3°
const double roiSmoothing = 0.5;

/// Zona morta + EMA em centro, lado e ângulo: abaixo da zona, a ROI não se mexe.
Roi dampRoi(Roi previous, Roi next) {
  final dx = next.cx - previous.cx, dy = next.cy - previous.cy;
  final moved = math.sqrt(dx * dx + dy * dy) > roiDeadbandPx;
  final cx = moved ? previous.cx + roiSmoothing * dx : previous.cx;
  final cy = moved ? previous.cy + roiSmoothing * dy : previous.cy;
  final scaled = (next.side / previous.side - 1).abs() > roiDeadbandScale;
  final side = scaled
      ? previous.side + roiSmoothing * (next.side - previous.side)
      : previous.side;
  // Diferença de ângulo em (−π, π]: a volta de 360° não pode virar um giro.
  var dr = (next.rotation - previous.rotation) % (2 * math.pi);
  if (dr > math.pi) dr -= 2 * math.pi;
  final rotation = dr.abs() > roiDeadbandAngle
      ? previous.rotation + roiSmoothing * dr
      : previous.rotation;
  return Roi(cx, cy, side, rotation);
}

/// Centro e maior extensão da caixa dos pontos.
(Pt, double) _extent(List<Pt> points) {
  var minX = double.infinity, minY = double.infinity;
  var maxX = -double.infinity, maxY = -double.infinity;
  for (final p in points) {
    minX = math.min(minX, p.x);
    minY = math.min(minY, p.y);
    maxX = math.max(maxX, p.x);
    maxY = math.max(maxY, p.y);
  }
  return (
    Pt((minX + maxX) / 2, (minY + maxY) / 2),
    math.max(maxX - minX, maxY - minY),
  );
}

/// Salto absurdo entre dois frames rastreados (o centro anda mais que meia mão,
/// ou o tamanho muda mais de 60%): os pontos pularam de lugar, e a palma decide.
bool implausibleJump(List<Pt> previous, List<Pt> next) {
  final (c0, s0) = _extent(previous);
  final (c1, s1) = _extent(next);
  if (s0 <= 0) return true;
  final dx = c1.x - c0.x, dy = c1.y - c0.y;
  final ratio = s1 / s0;
  return math.sqrt(dx * dx + dy * dy) > 0.5 * s0 || ratio < 0.6 || ratio > 1.6;
}

/// A mão perto demais da câmera: algum ponto encosta na borda do frame, ou os
/// pontos ocupam mais de 75% do lado menor. A vista 224 sair da imagem não conta:
/// a 40 cm do celular ela já tem ~500 px, e a borda replicada dá conta.
bool handTooClose(List<Pt> points, int width, int height) {
  final short = math.min(width, height).toDouble();
  final margin = 0.02 * short;
  final (_, span) = _extent(points);
  if (span > 0.75 * short) return true;
  for (final p in points) {
    if (p.x < margin ||
        p.y < margin ||
        p.x > width - margin ||
        p.y > height - margin) {
      return true;
    }
  }
  return false;
}

/// Saída do modelo de pontos (63 floats em pixels do recorte 224) para o frame.
List<Pt> landmarksToFrame(Float32List output, Roi roi) {
  final m = roi.affine(landmarkInput);
  return [
    for (var i = 0; i < 21; i++) m.apply(output[3 * i], output[3 * i + 1]),
  ];
}

/// Ligações dos 21 pontos, para o desenho do esqueleto.
const List<(int, int)> handConnections = [
  (0, 1),
  (1, 2),
  (2, 3),
  (3, 4),
  (0, 5),
  (5, 6),
  (6, 7),
  (7, 8),
  (5, 9),
  (9, 10),
  (10, 11),
  (11, 12),
  (9, 13),
  (13, 14),
  (14, 15),
  (15, 16),
  (13, 17),
  (0, 17),
  (17, 18),
  (18, 19),
  (19, 20),
];
