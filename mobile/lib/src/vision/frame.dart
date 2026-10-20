/// Amostragem bilinear de um frame em pé (já rotacionado) por uma [Affine].
///
/// Nunca convertemos o frame inteiro para RGB: cada modelo pede só o seu
/// recorte (192², 224², 224²), e cada pixel do tensor lê quatro vizinhos do
/// YUV do sensor. Mesma convenção do `grid_sample(align_corners=False)`: o
/// centro do pixel u da saída é u + 0,5.
library;

import 'dart:isolate';
import 'dart:typed_data';

import 'geometry.dart';

abstract class FrameImage {
  /// Largura e altura do frame em pé (depois da rotação do sensor).
  int get width;
  int get height;

  /// Preenche [out] (NHWC, [size]×[size]×3, valores em [0, 1]) com o quadrado
  /// [view]. Fora da foto: borda replicada, ou preto com [zeroBorder]
  /// (letterbox do detector de palma).
  void sample(
    Affine view,
    int size,
    Float32List out, {
    bool zeroBorder = false,
  });

  /// Como [sample], com o vizinho mais próximo e borda replicada: 3 leituras por
  /// pixel em vez de 9. Só para a ROI do modelo de pontos (a ROI tem 200–400 px e
  /// vira 224, quase 1:1); o SignEdge recebe o bilinear do treino.
  void sampleNearest(Affine view, int size, Float32List out);
}

/// Rotação do sensor para o frame em pé, em coordenadas contínuas.
///
/// [rotationDegrees] é o quanto girar a imagem do sensor no sentido horário
/// (o `sensorOrientation` da câmera com o aparelho em retrato). Devolve a
/// afim em pé → sensor.
Affine uprightToSensor(int rotationDegrees, int sensorWidth, int sensorHeight) {
  final w = sensorWidth.toDouble(), h = sensorHeight.toDouble();
  switch (rotationDegrees % 360) {
    case 90:
      return Affine(0, 1, 0, -1, 0, h); // sx = y', sy = h − x'
    case 180:
      return Affine(-1, 0, w, 0, -1, h);
    case 270:
      return Affine(0, -1, w, 1, 0, 0); // sx = w − y', sy = x'
    default:
      return const Affine(1, 0, 0, 0, 1, 0);
  }
}

/// Frame YUV 4:2:0 do Android (YUV_420_888), lido sem conversão prévia.
class YuvFrame implements FrameImage {
  YuvFrame({
    required this.sensorWidth,
    required this.sensorHeight,
    required this.rotationDegrees,
    required this.y,
    required this.u,
    required this.v,
    required this.yRowStride,
    required this.uvRowStride,
    required this.uvPixelStride,
  }) : _toSensor = uprightToSensor(rotationDegrees, sensorWidth, sensorHeight);

  final int sensorWidth, sensorHeight, rotationDegrees;
  final Uint8List y, u, v;
  final int yRowStride, uvRowStride, uvPixelStride;
  final Affine _toSensor;

  bool get _swap => rotationDegrees % 180 != 0;

  @override
  int get width => _swap ? sensorHeight : sensorWidth;

  @override
  int get height => _swap ? sensorWidth : sensorHeight;

  @override
  void sample(
    Affine view,
    int size,
    Float32List out, {
    bool zeroBorder = false,
    bool exact = false,
  }) {
    final m = view.andThen(_toSensor);
    final w = sensorWidth, h = sensorHeight;
    final yRow = yRowStride, uvRow = uvRowStride, uvPix = uvPixelStride;
    final lumaPlane = y, uPlane = u, vPlane = v;
    // Ponto fixo 16.16 nas coordenadas do sensor: sem floor() de double por pixel.
    const one = 1 << 16;
    final stepX = (m.a * one).round(), stepY = (m.d * one).round();
    var o = 0;
    for (var row = 0; row < size; row++) {
      final v0 = row + 0.5;
      final x = m.a * 0.5 + m.b * v0 + m.c - 0.5;
      final yy = m.d * 0.5 + m.e * v0 + m.f - 0.5;
      var xf = (x * one).round(), yf = (yy * one).round();
      for (var col = 0; col < size; col++, xf += stepX, yf += stepY) {
        final x0 = xf >> 16, y0 = yf >> 16;
        // [exact] desliga o caminho rápido: o recorte de referência do diagnóstico.
        if (!exact && x0 >= 0 && y0 >= 0 && x0 + 1 < w && y0 + 1 < h) {
          // Caminho rápido (quase todo pixel): luma bilinear em inteiros; a cor, que
          // o sensor já guarda em meia resolução, vem do vizinho mais próximo.
          final fx = xf & 0xFFFF, fy = yf & 0xFFFF;
          final p = y0 * yRow + x0;
          final top = lumaPlane[p] * (one - fx) + lumaPlane[p + 1] * fx;
          final bottom =
              lumaPlane[p + yRow] * (one - fx) + lumaPlane[p + yRow + 1] * fx;
          // Luma × 256 (8 bits de fração).
          final lum = (top * (one - fy) + bottom * fy) >> 24;
          final cx = fx < 0x8000 ? x0 : x0 + 1, cy = fy < 0x8000 ? y0 : y0 + 1;
          final uv = (cy >> 1) * uvRow + (cx >> 1) * uvPix;
          final dU = uPlane[uv] - 128, dV = vPlane[uv] - 128;
          // BT.601 de faixa cheia com coeficientes × 256: 1,402 · 0,344 · 0,714 · 1,772.
          final r = lum + 359 * dV;
          final g = lum - 88 * dU - 183 * dV;
          final b = lum + 454 * dU;
          out[o++] = r <= 0 ? 0 : (r >= _white ? 1 : r * _toUnit);
          out[o++] = g <= 0 ? 0 : (g >= _white ? 1 : g * _toUnit);
          out[o++] = b <= 0 ? 0 : (b >= _white ? 1 : b * _toUnit);
          continue;
        }
        final xs = xf / one, ys = yf / one;
        final fx = xs - x0, fy = ys - y0;
        var lum = 0.0, cb = 0.0, cr = 0.0;
        for (var k = 0; k < 4; k++) {
          final ix = x0 + (k & 1), iy = y0 + (k >> 1);
          final weight =
              ((k & 1) == 1 ? fx : 1 - fx) * ((k >> 1) == 1 ? fy : 1 - fy);
          if (weight == 0) continue;
          int cx = ix, cy = iy;
          if (ix < 0 || iy < 0 || ix >= w || iy >= h) {
            if (zeroBorder) {
              cb += 128 * weight;
              cr += 128 * weight;
              continue;
            }
            cx = ix < 0 ? 0 : (ix >= w ? w - 1 : ix);
            cy = iy < 0 ? 0 : (iy >= h ? h - 1 : iy);
          }
          final uv = (cy >> 1) * uvRowStride + (cx >> 1) * uvPixelStride;
          lum += y[cy * yRowStride + cx] * weight;
          cb += u[uv] * weight;
          cr += v[uv] * weight;
        }
        // BT.601 de faixa cheia (JFIF), o YUV das câmeras do Android.
        final dU = cb - 128, dV = cr - 128;
        out[o++] = _unit(lum + 1.402 * dV);
        out[o++] = _unit(lum - 0.344136 * dU - 0.714136 * dV);
        out[o++] = _unit(lum + 1.772 * dU);
      }
    }
  }

  @override
  void sampleNearest(Affine view, int size, Float32List out) {
    final m = view.andThen(_toSensor);
    final wMax = sensorWidth - 1, hMax = sensorHeight - 1;
    final yRow = yRowStride, uvRow = uvRowStride, uvPix = uvPixelStride;
    final lumaPlane = y, uPlane = u, vPlane = v;
    const one = 1 << 16, half = 1 << 15;
    final stepX = (m.a * one).round(), stepY = (m.d * one).round();
    var o = 0;
    for (var row = 0; row < size; row++) {
      final v0 = row + 0.5;
      // + meio pixel: o arredondamento vira um shift (vizinho mais próximo).
      var xf = ((m.a * 0.5 + m.b * v0 + m.c - 0.5) * one).round() + half;
      var yf = ((m.d * 0.5 + m.e * v0 + m.f - 0.5) * one).round() + half;
      for (var col = 0; col < size; col++, xf += stepX, yf += stepY) {
        var x = xf >> 16, yy = yf >> 16;
        if (x < 0) {
          x = 0;
        } else if (x > wMax) {
          x = wMax;
        }
        if (yy < 0) {
          yy = 0;
        } else if (yy > hMax) {
          yy = hMax;
        }
        final lum = lumaPlane[yy * yRow + x] << 8;
        final uv = (yy >> 1) * uvRow + (x >> 1) * uvPix;
        final dU = uPlane[uv] - 128, dV = vPlane[uv] - 128;
        final r = lum + 359 * dV;
        final g = lum - 88 * dU - 183 * dV;
        final b = lum + 454 * dU;
        out[o++] = r <= 0 ? 0 : (r >= _white ? 1 : r * _toUnit);
        out[o++] = g <= 0 ? 0 : (g >= _white ? 1 : g * _toUnit);
        out[o++] = b <= 0 ? 0 : (b >= _white ? 1 : b * _toUnit);
      }
    }
  }

  static const int _white = 255 * 256;
  static const double _toUnit = 1 / _white;

  static double _unit(double value) =>
      value <= 0 ? 0 : (value >= 255 ? 1 : value * (1 / 255));
}

/// Frame RGB em [0, 1] (H×W×3), para testes e imagens estáticas.
class RgbFrame implements FrameImage {
  RgbFrame(this.width, this.height, this.pixels);

  @override
  final int width;
  @override
  final int height;
  final Float32List pixels;

  @override
  void sample(
    Affine view,
    int size,
    Float32List out, {
    bool zeroBorder = false,
  }) {
    final m = view;
    var o = 0;
    for (var row = 0; row < size; row++) {
      final v0 = row + 0.5;
      for (var col = 0; col < size; col++) {
        final u0 = col + 0.5;
        final x = m.a * u0 + m.b * v0 + m.c - 0.5;
        final yy = m.d * u0 + m.e * v0 + m.f - 0.5;
        final x0 = x.floor(), y0 = yy.floor();
        final fx = x - x0, fy = yy - y0;
        var r = 0.0, g = 0.0, b = 0.0;
        for (var k = 0; k < 4; k++) {
          final ix = x0 + (k & 1), iy = y0 + (k >> 1);
          final weight =
              ((k & 1) == 1 ? fx : 1 - fx) * ((k >> 1) == 1 ? fy : 1 - fy);
          if (weight == 0) continue;
          int cx = ix, cy = iy;
          if (ix < 0 || iy < 0 || ix >= width || iy >= height) {
            if (zeroBorder) continue;
            cx = ix < 0 ? 0 : (ix >= width ? width - 1 : ix);
            cy = iy < 0 ? 0 : (iy >= height ? height - 1 : iy);
          }
          final p = (cy * width + cx) * 3;
          r += pixels[p] * weight;
          g += pixels[p + 1] * weight;
          b += pixels[p + 2] * weight;
        }
        out[o++] = r;
        out[o++] = g;
        out[o++] = b;
      }
    }
  }

  @override
  void sampleNearest(Affine view, int size, Float32List out) {
    var o = 0;
    for (var row = 0; row < size; row++) {
      for (var col = 0; col < size; col++) {
        final p = view.apply(col + 0.5, row + 0.5);
        final x = (p.x - 0.5).round().clamp(0, width - 1);
        final y = (p.y - 0.5).round().clamp(0, height - 1);
        final q = (y * width + x) * 3;
        out[o++] = pixels[q];
        out[o++] = pixels[q + 1];
        out[o++] = pixels[q + 2];
      }
    }
  }
}

/// Os planos de um frame da câmera, prontos para mandar a outro isolate.
///
/// Cada [TransferableTypedData] só pode ser materializado uma vez: cada isolate
/// recebe a sua cópia (~0,4 ms para 640×480 no aparelho).
class YuvPlanes {
  YuvPlanes({
    required Uint8List y,
    required Uint8List u,
    required Uint8List v,
    required this.width,
    required this.height,
    required this.rotation,
    required this.yRowStride,
    required this.uvRowStride,
    required this.uvPixelStride,
  }) : _y = TransferableTypedData.fromList([y]),
       _u = TransferableTypedData.fromList([u]),
       _v = TransferableTypedData.fromList([v]);

  final TransferableTypedData _y, _u, _v;
  final int width, height, rotation, yRowStride, uvRowStride, uvPixelStride;

  YuvFrame materialize() => YuvFrame(
    sensorWidth: width,
    sensorHeight: height,
    rotationDegrees: rotation,
    y: _y.materialize().asUint8List(),
    u: _u.materialize().asUint8List(),
    v: _v.materialize().asUint8List(),
    yRowStride: yRowStride,
    uvRowStride: uvRowStride,
    uvPixelStride: uvPixelStride,
  );
}
