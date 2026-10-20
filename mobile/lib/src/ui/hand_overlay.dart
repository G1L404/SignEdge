import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../vision/geometry.dart';
import '../vision/hand_math.dart';

/// Esqueleto da mão, a caixa justa e a vista 224 que o modelo enxerga.
///
/// Recebe coordenadas do frame em pé; na câmera frontal espelha o desenho
/// para acompanhar o preview (o modelo recebe o frame sem espelho).
class HandOverlay extends CustomPainter {
  HandOverlay({
    required this.frameWidth,
    required this.frameHeight,
    required this.landmarks,
    required this.view,
    required this.mirror,
    required this.accent,
    this.warning = const Color(0xFFFFB020),
  });

  final int frameWidth, frameHeight;
  final Float32List? landmarks;
  final Float32List? view;
  final bool mirror;
  final Color accent, warning;

  @override
  void paint(Canvas canvas, Size size) {
    final points = landmarks;
    if (points == null || frameWidth == 0) return;
    final scale = size.width / frameWidth;
    Offset at(double x, double y) =>
        Offset((mirror ? frameWidth - x : x) * scale, y * scale);

    final v = view;
    if (v != null) {
      final side = v[2] * scale;
      final viewRect = Rect.fromCenter(
        center: at(v[0], v[1]),
        width: side,
        height: side,
      );
      // O aviso olha a mão (os 21 pontos), não a vista 224: a ~40 cm do celular
      // a vista já passa dos 480 px de largura, e a borda replicada dá conta.
      final tooClose = handTooClose(
        [for (var i = 0; i < 21; i++) Pt(points[2 * i], points[2 * i + 1])],
        frameWidth,
        frameHeight,
      );
      final boxColor = tooClose ? warning : accent;
      canvas.drawRect(
        viewRect,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2
          ..color = boxColor,
      );
      if (tooClose) {
        final hint = TextPainter(
          text: TextSpan(
            text: ' Afaste um pouco a mão ',
            style: TextStyle(
              color: Colors.black,
              backgroundColor: warning,
              fontSize: 14,
              fontWeight: FontWeight.w700,
            ),
          ),
          textDirection: TextDirection.ltr,
        )..layout();
        hint.paint(
          canvas,
          Offset((size.width - hint.width) / 2, size.height - hint.height - 12),
        );
      }
      final inner = side / CropRule.viewContext;
      canvas.drawRect(
        Rect.fromCenter(center: viewRect.center, width: inner, height: inner),
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1
          ..color = accent.withValues(alpha: 0.45),
      );
      final label = TextPainter(
        text: TextSpan(
          text: ' vista 224 ',
          style: TextStyle(
            color: Colors.black,
            backgroundColor: boxColor,
            fontSize: 11,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      label.paint(canvas, viewRect.topLeft - Offset(0, label.height));
    }

    final bone = Paint()
      ..color = Colors.white.withValues(alpha: 0.85)
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round;
    for (final (a, b) in handConnections) {
      canvas.drawLine(
        at(points[2 * a], points[2 * a + 1]),
        at(points[2 * b], points[2 * b + 1]),
        bone,
      );
    }
    final joint = Paint()..color = accent;
    for (var i = 0; i < 21; i++) {
      canvas.drawCircle(
        at(points[2 * i], points[2 * i + 1]),
        i == 0 ? 5 : 3.5,
        joint,
      );
    }
  }

  @override
  bool shouldRepaint(HandOverlay old) =>
      old.landmarks != landmarks || old.view != view || old.mirror != mirror;
}
