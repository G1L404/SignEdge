// Contrato geométrico contra valores gerados no Python (torch.grid_sample,
// hand_square e map_landmarks do treino; ROI da referência validada contra o
// MediaPipe oficial). Regenerar: ver o cabeçalho de test/fixtures.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:signedge_mobile/src/vision/frame.dart';
import 'package:signedge_mobile/src/vision/geometry.dart';
import 'package:signedge_mobile/src/vision/hand_math.dart';

Map<String, dynamic> _fixture(String name) =>
    jsonDecode(File('test/fixtures/$name').readAsStringSync())
        as Map<String, dynamic>;

List<Pt> _points(List<dynamic> raw) => [
  for (final p in raw) Pt((p[0] as num).toDouble(), (p[1] as num).toDouble()),
];

void main() {
  group('vista do modelo (EvalView)', () {
    final gold = _fixture('view_golden.json');
    final width = gold['W'] as int, height = gold['H'] as int;
    final points = _points(gold['landmarks'] as List);

    test('caixa justa e lado da vista iguais ao hand_square', () {
      final view = ModelView.fromLandmarks(
        points,
        height < width ? height : width,
      );
      final box = (gold['box'] as List).cast<int>();
      expect([
        view.box.x0,
        view.box.y0,
        view.box.x0 + view.box.side,
        view.box.y0 + view.box.side,
      ], box);
      expect(view.side, closeTo((gold['side'] as num).toDouble(), 1e-9));
      final center = (gold['center'] as List).cast<num>();
      expect(view.center.x, closeTo(center[0].toDouble(), 1e-9));
      expect(view.center.y, closeTo(center[1].toDouble(), 1e-9));
    });

    test('recorte bilinear com borda replicada igual ao grid_sample', () {
      final pixels = Float32List(width * height * 3);
      for (var y = 0; y < height; y++) {
        for (var x = 0; x < width; x++) {
          for (var c = 0; c < 3; c++) {
            pixels[(y * width + x) * 3 + c] =
                ((x * 7 + y * 13 + c * 29) % 256) / 255;
          }
        }
      }
      final view = ModelView.fromLandmarks(points, height);
      final out = Float32List(224 * 224 * 3);
      RgbFrame(width, height, pixels).sample(view.affine, 224, out);
      for (final probe in gold['probe'] as List) {
        final x = probe[0] as int, y = probe[1] as int;
        final rgb = (probe[2] as List).cast<num>();
        for (var c = 0; c < 3; c++) {
          expect(
            out[(y * 224 + x) * 3 + c],
            closeTo(rgb[c].toDouble(), 1e-5),
            reason: '($x, $y, $c)',
          );
        }
      }
      final sum = out.fold<double>(0, (a, b) => a + b);
      expect(sum, closeTo((gold['sum'] as num).toDouble(), 0.05));
    });

    test('pontos em pixels da vista 224 iguais ao map_landmarks', () {
      final view = ModelView.fromLandmarks(points, height);
      final mapped = view.mapLandmarks(points);
      final expected = _points(gold['mapped'] as List);
      for (var i = 0; i < 21; i++) {
        expect(mapped[i].x, closeTo(expected[i].x, 1e-6));
        expect(mapped[i].y, closeTo(expected[i].y, 1e-6));
      }
    });
  });

  group('MediaPipe em Dart', () {
    final gold = _fixture('roi_golden.json');

    test('2016 âncoras do detector de palma', () {
      expect(palmAnchors.length, 2 * (gold['n_anchors'] as int));
      for (final (key, index) in [
        ('anchor_0', 0),
        ('anchor_1151', 1151),
        ('anchor_1152', 1152),
      ]) {
        final a = (gold[key] as List).cast<num>();
        expect(palmAnchors[2 * index], closeTo(a[0].toDouble(), 1e-6));
        expect(palmAnchors[2 * index + 1], closeTo(a[1].toDouble(), 1e-6));
      }
    });

    test('palma: âncoras, NMS ponderado e ROI iguais à referência', () {
      final palm = _fixture('palm_golden.json');
      Float32List bin(String name) =>
          File('test/fixtures/$name').readAsBytesSync().buffer.asFloat32List();
      final result = decodePalm(
        bin('palm_raw.bin'),
        bin('palm_scores.bin'),
        palm['width'] as int,
        palm['height'] as int,
      )!;
      final expected = (palm['roi'] as List).cast<num>();
      expect(result.score, closeTo((palm['score'] as num).toDouble(), 1e-5));
      expect(result.roi.cx, closeTo(expected[0].toDouble(), 1e-3));
      expect(result.roi.cy, closeTo(expected[1].toDouble(), 1e-3));
      expect(result.roi.side, closeTo(expected[2].toDouble(), 1e-3));
      expect(result.roi.rotation, closeTo(expected[3].toDouble(), 1e-5));
    });

    test('ROI de rastreamento a partir dos pontos', () {
      final roi = roiFromLandmarks(_points(gold['landmarks'] as List));
      final expected = (gold['roi'] as List).cast<num>();
      expect(roi.cx, closeTo(expected[0].toDouble(), 1e-6));
      expect(roi.cy, closeTo(expected[1].toDouble(), 1e-6));
      expect(roi.side, closeTo(expected[2].toDouble(), 1e-6));
      expect(roi.rotation, closeTo(expected[3].toDouble(), 1e-9));
      final m = roi.affine(landmarkInput);
      final em = (gold['matrix'] as List)
          .map((r) => (r as List).cast<num>())
          .toList();
      for (final (got, want) in [
        (m.a, em[0][0]),
        (m.b, em[0][1]),
        (m.c, em[0][2]),
        (m.d, em[1][0]),
        (m.e, em[1][1]),
        (m.f, em[1][2]),
      ]) {
        expect(got, closeTo(want.toDouble(), 1e-6));
      }
    });
  });

  group('rastreio solta o falso positivo', () {
    // Mão de 120 px num frame 480×640, e a mesma mão colapsada num ponto.
    final hand = [
      for (var i = 0; i < 21; i++) Pt(200 + 6.0 * i, 300 + 3.0 * (i % 7)),
    ];
    final collapsed = [
      for (var i = 0; i < 21; i++) Pt(240 + 0.3 * i, 500 + 0.2 * i),
    ];

    test('presença já é probabilidade: fundo (0,01) não passa', () {
      // Saídas cruas medidas no hand_landmarks_detector: mão 0,99; fundo ≤ 0,11.
      expect(plausibleHand(0.99, hand, 480), isTrue);
      expect(plausibleHand(0.11, hand, 480), isFalse);
      expect(plausibleHand(0.006, hand, 480), isFalse);
    });

    test('pontos colapsados não seguram o rastreio', () {
      expect(plausibleHand(0.99, collapsed, 480), isFalse);
    });
  });

  group('estabilidade do rastreio', () {
    test(
      'ROI: tremor abaixo da zona morta não move; mudança real anda pela metade',
      () {
        const a = Roi(240, 320, 300, 0.10);
        final still = dampRoi(a, const Roi(241.5, 319, 305, 0.12));
        expect(
          [still.cx, still.cy, still.side, still.rotation],
          [240, 320, 300, 0.10],
        );
        final moved = dampRoi(a, const Roi(260, 320, 360, 0.30));
        expect(moved.cx, closeTo(250, 1e-9));
        expect(moved.side, closeTo(330, 1e-9));
        expect(moved.rotation, closeTo(0.20, 1e-9));
      },
    );

    test('ROI: o ângulo dá a volta pelo caminho curto', () {
      final r = dampRoi(const Roi(0, 0, 100, 3.1), const Roi(0, 0, 100, -3.1));
      // De 3,1 a −3,1 são ~0,08 rad pelo caminho curto, não 6,2.
      expect(
        r.rotation,
        closeTo(3.1 + 0.5 * (2 * 3.141592653589793 - 6.2), 1e-6),
      );
    });

    test('salto absurdo e mão perto demais', () {
      final hand = [
        for (var i = 0; i < 21; i++) Pt(200 + 5.0 * i, 300 + 4.0 * (i % 5)),
      ];
      final shifted = [for (final p in hand) Pt(p.x + 80, p.y)];
      final nudged = [for (final p in hand) Pt(p.x + 6, p.y + 3)];
      expect(implausibleJump(hand, nudged), isFalse);
      expect(implausibleJump(hand, shifted), isTrue);
      expect(handTooClose(hand, 480, 640), isFalse);
      final edge = [...hand.sublist(1), const Pt(3, 300)];
      expect(handTooClose(edge, 480, 640), isTrue);
      final huge = [for (var i = 0; i < 21; i++) Pt(40 + 19.0 * i, 300)];
      expect(handTooClose(huge, 480, 640), isTrue);
    });
  });

  group('rotação do sensor', () {
    test('90° e 270° levam os cantos do frame em pé aos do sensor', () {
      // Sensor 640×480; em pé 480×640.
      final back = uprightToSensor(90, 640, 480);
      final topLeft = back.apply(0, 0), topRight = back.apply(480, 0);
      expect([topLeft.x, topLeft.y], [0, 480]);
      expect([topRight.x, topRight.y], [0, 0]);
      final front = uprightToSensor(270, 640, 480);
      final origin = front.apply(0, 0);
      expect([origin.x, origin.y], [640, 0]);
    });

    test('caminho rápido do YUV (90°) igual ao RgbFrame de referência', () {
      // Sensor 40×30 → em pé 30×40. Imagem em tons de cinza (U = V = 128).
      const sw = 40, sh = 30;
      double g(int col, int row) =>
          (37 + 3 * col + 5 * row + (col * row) % 11) % 256 + 0.0;
      final yPlane = Uint8List(sw * sh);
      for (var row = 0; row < sw; row++) {
        for (var col = 0; col < sh; col++) {
          // Pixel em pé (col, row) vem do sensor (sx = row, sy = sh − 1 − col).
          yPlane[(sh - 1 - col) * sw + row] = g(col, row).toInt();
        }
      }
      final uv = Uint8List(sw * sh ~/ 2)..fillRange(0, sw * sh ~/ 2, 128);
      final yuv = YuvFrame(
        sensorWidth: sw,
        sensorHeight: sh,
        rotationDegrees: 90,
        y: yPlane,
        u: uv,
        v: uv,
        yRowStride: sw,
        uvRowStride: sw,
        uvPixelStride: 2,
      );
      final rgb = Float32List(sh * sw * 3);
      for (var row = 0; row < sw; row++) {
        for (var col = 0; col < sh; col++) {
          final value = g(col, row).toInt() / 255;
          rgb.fillRange((row * sh + col) * 3, (row * sh + col) * 3 + 3, value);
        }
      }
      // Quadrado girado, todo dentro do frame (só caminho rápido).
      final view = Affine.square(15, 20, 14, 0.4, 16);
      final a = Float32List(16 * 16 * 3), b = Float32List(16 * 16 * 3);
      yuv.sample(view, 16, a);
      RgbFrame(sh, sw, rgb).sample(view, 16, b);
      for (var i = 0; i < a.length; i++) {
        // Ponto fixo: luma com 8 bits de fração (erro < 1/256 de nível, ~1,5e-5).
        expect(a[i], closeTo(b[i], 5e-5), reason: 'índice $i');
      }
    });

    test('YUV cinza vira RGB cinza e a vista respeita a rotação', () {
      const w = 8, h = 6;
      final y = Uint8List(w * h)..fillRange(0, w * h, 128);
      final uv = Uint8List(w * h ~/ 2)..fillRange(0, w * h ~/ 2, 128);
      final frame = YuvFrame(
        sensorWidth: w,
        sensorHeight: h,
        rotationDegrees: 90,
        y: y,
        u: uv,
        v: uv,
        yRowStride: w,
        uvRowStride: w,
        uvPixelStride: 2,
      );
      expect([frame.width, frame.height], [h, w]);
      final out = Float32List(4 * 4 * 3);
      frame.sample(Affine.square(3, 4, 4, 0, 4), 4, out);
      for (final value in out) {
        expect(value, closeTo(128 / 255, 1e-6));
      }
    });
  });
}
