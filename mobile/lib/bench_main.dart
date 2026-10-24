// Benchmark isolado do classificador no aparelho, em release e fora do app normal:
//   flutter build apk --release -t lib/bench_main.dart
// Mesmo tflite_flutter e mesmo XNNPackDelegate padrão do app (openVerified), sem
// câmera, recorte nem MediaPipe. Os modelos vão por adb para a pasta do app
// (Android/data/<pacote>/files/bench/<nome>.tflite); a saída é uma linha
// BENCH_JSON por modelo no logcat.
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'src/vision/pipeline.dart';

const _dir =
    '/storage/emulated/0/Android/data/br.ufpr.signedge.signedge_mobile/files/bench';
// Só os de pontos em float: os Int8 integrais não abrem no LiteRT 1.4 do Android
// (SQRT sem núcleo Int8), o mesmo motivo de o app usar model_int8_lmfloat.
const _models = ['denso_lmfloat', 'lamp_lmfloat', 'denso_float', 'lamp_float'];
const _warmup = 30, _blocks = 5, _runs = 200;

/// Cópia alinhada a 16 bytes, como a memória do asset: o inline_buffers alinha os
/// pesos a 16 bytes contando do início do arquivo.
Uint8List _aligned(Uint8List bytes) {
  final out = Int32x4List(
    (bytes.length + 15) ~/ 16,
  ).buffer.asUint8List(0, bytes.length);
  out.setAll(0, bytes);
  return out;
}

class _Bench {
  _Bench(this.name, Uint8List bytes) {
    final opened = openVerified(bytes, const [
      Backend.xnnpack,
      Backend.gpu,
      Backend.cpu,
    ]);
    interpreter = opened.$1;
    backend = opened.$2.name;
    final inputs = interpreter.getInputTensors();
    rgbIndex = inputs.indexWhere((t) => t.name.contains('rgb'));
    landmarksIndex = inputs.indexWhere((t) => t.name.contains('landmarks'));
    final random = math.Random(0);
    for (var i = 0; i < rgb.length; i++) {
      rgb[i] = random.nextDouble();
    }
    for (var i = 0; i < landmarks.length; i++) {
      landmarks[i] = random.nextDouble() * 224;
    }
  }

  final String name;
  late final Interpreter interpreter;
  late final String backend;
  late final int rgbIndex, landmarksIndex;
  final rgb = Float32List(224 * 224 * 3);
  final landmarks = Float32List(42);
  final logits = Float32List(61);
  final invokeMs = <double>[], fullMs = <double>[];

  void _setInputs() {
    interpreter.getInputTensor(rgbIndex).setTo(rgb.buffer.asUint8List());
    interpreter
        .getInputTensor(landmarksIndex)
        .setTo(landmarks.buffer.asUint8List());
  }

  void warmup() {
    for (var i = 0; i < _warmup; i++) {
      _setInputs();
      interpreter.invoke();
    }
  }

  /// Um bloco: média do invoke sozinho e do invoke com as cópias do app.
  void block() {
    _setInputs();
    final clock = Stopwatch()..start();
    for (var i = 0; i < _runs; i++) {
      interpreter.invoke();
    }
    invokeMs.add(clock.elapsedMicroseconds / 1000 / _runs);
    clock
      ..reset()
      ..start();
    for (var i = 0; i < _runs; i++) {
      _setInputs();
      interpreter.invoke();
      interpreter.getOutputTensor(0).copyTo(logits.buffer.asUint8List());
    }
    fullMs.add(clock.elapsedMicroseconds / 1000 / _runs);
  }

  static Map<String, double> _summary(List<double> v) {
    final s = [...v]..sort();
    double r(double x) => (x * 1000).round() / 1000;
    return {
      'mediana': r(s[s.length ~/ 2]),
      'min': r(s.first),
      'max': r(s.last),
    };
  }

  Map<String, Object> toJson() => {
    'modelo': name,
    'backend': backend,
    'invoke_ms': _summary(invokeMs),
    'invoke_com_copias_ms': _summary(fullMs),
    'blocos': invokeMs.length,
    'execucoes_por_bloco': _runs,
  };
}

/// Roda num isolate de fundo: a UI não trava e o Android não acusa ANR.
List<String> _run(Map<String, Uint8List> files) {
  final benches = [for (final m in _models) _Bench(m, _aligned(files[m]!))];
  for (final b in benches) {
    b.warmup();
  }
  // Modelos alternados em cada bloco, com a ordem girando: aquecimento e
  // temperatura pesam igual para todos.
  for (var k = 0; k < _blocks; k++) {
    for (var i = 0; i < benches.length; i++) {
      benches[(i + k) % benches.length].block();
    }
  }
  final out = [for (final b in benches) jsonEncode(b.toJson())];
  for (final b in benches) {
    b.interpreter.close();
  }
  return out;
}

Future<String> _benchmark() async {
  final missing = [
    for (final m in _models)
      if (!File('$_dir/$m.tflite').existsSync()) m,
  ];
  if (missing.isNotEmpty) return 'BENCH_ERRO faltam: ${missing.join(', ')}';
  final files = {
    for (final m in _models) m: File('$_dir/$m.tflite').readAsBytesSync(),
  };
  debugPrint('BENCH_INICIO');
  try {
    final lines = await Isolate.run(() => _run(files));
    for (final l in lines) {
      debugPrint('BENCH_JSON $l');
    }
    debugPrint('BENCH_FIM');
    return lines.join('\n\n');
  } catch (error) {
    debugPrint('BENCH_ERRO $error');
    return 'BENCH_ERRO $error';
  }
}

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const _BenchApp());
}

class _BenchApp extends StatelessWidget {
  const _BenchApp();

  @override
  Widget build(BuildContext context) => MaterialApp(
    home: Scaffold(
      body: SafeArea(
        child: FutureBuilder<String>(
          future: _benchmark(),
          builder: (context, snapshot) => Padding(
            padding: const EdgeInsets.all(16),
            child: SingleChildScrollView(
              child: Text(
                snapshot.data ?? 'Benchmark do classificador em andamento…',
                style: const TextStyle(fontSize: 12),
              ),
            ),
          ),
        ),
      ),
    ),
  );
}
