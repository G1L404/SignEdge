/// As duas etapas de um frame, cada uma no seu isolate:
///
/// - [HandStage] (GPU): palma (só quando perde a mão) → 21 pontos → recorte da
///   vista do modelo;
/// - [ClassifierStage] (CPU, XNNPACK): SignEdge Int8 sobre esse recorte.
///
/// Separadas, a GPU já rastreia o frame t+1 enquanto a CPU classifica o t: a
/// vazão é a da etapa mais lenta, não a soma das duas.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:tflite_flutter/tflite_flutter.dart';

import 'frame.dart';
import 'geometry.dart';
import 'hand_math.dart';

enum Backend { gpu, xnnpack, cpu }

/// O primeiro backend de [order] que abre o modelo e passa num invoke.
///
/// Cada interpretador só é aceito depois de um invoke de aquecimento: se o
/// backend falha no aparelho, cai para o próximo. O construtor do Interpreter
/// já aloca os tensores; não alocar de novo.
(Interpreter, Backend) openVerified(
  Uint8List bytes,
  List<Backend> order, {
  int threads = 2,
}) {
  Object? last;
  for (final backend in order) {
    Interpreter? interpreter;
    try {
      final options = InterpreterOptions();
      switch (backend) {
        case Backend.gpu:
          options.addDelegate(
            GpuDelegateV2(
              options: GpuDelegateOptionsV2(isPrecisionLossAllowed: true),
            ),
          );
        case Backend.xnnpack:
          // Sem opções: as padrão do XNNPACK, com os kernels Int8 ligados.
          options.addDelegate(XNNPackDelegate());
        case Backend.cpu:
          options.threads = threads;
      }
      interpreter = Interpreter.fromBuffer(bytes, options: options);
      interpreter.invoke();
      return (interpreter, backend);
    } catch (error) {
      interpreter?.close();
      last = error;
    }
  }
  throw StateError('nenhum backend abriu o modelo: $last');
}

/// Invoke com o nome do modelo no erro: o log nativo do TFLite não diz qual falhou.
void invokeNamed(Interpreter interpreter, String model) {
  try {
    interpreter.invoke();
  } catch (error) {
    throw StateError('modelo $model: $error');
  }
}

/// Cronômetro por etapa de um frame, devolvido junto com o resultado.
class StageClock {
  final Stopwatch _clock = Stopwatch()..start();
  final Map<String, double> stages = {};
  int _mark = 0;

  int get micros => _clock.elapsedMicroseconds;

  /// Zera o cronômetro; com [keep], mantém as etapas (só reinicia a contagem).
  void start({bool keep = false}) {
    if (!keep) stages.clear();
    _mark = micros;
  }

  /// Fecha a etapa [name]: o tempo desde a etapa anterior entra no perfil.
  void lap(String name) {
    final now = micros;
    stages[name] = (stages[name] ?? 0) + (now - _mark) / 1000;
    _mark = now;
  }
}

/// Saída do isolate da mão, em coordenadas do frame em pé.
class HandResult {
  const HandResult({
    required this.landmarks,
    required this.view,
    required this.viewPoints,
    required this.handMs,
    required this.usedPalm,
    required this.handBackends,
    required this.stages,
  });

  /// 21 pontos (x, y intercalados) ou null sem mão.
  final Float32List? landmarks;

  /// Vista do modelo: (cx, cy, lado) em pixels do frame.
  final Float32List? view;

  /// Os pontos em pixels da vista 224, a segunda entrada do SignEdge.
  final Float32List? viewPoints;

  bool get hasHand => landmarks != null;

  final double handMs;
  final bool usedPalm;

  /// Onde rodaram palma e pontos (ex.: "GPU · XNNPACK").
  final String handBackends;

  /// Milissegundos por etapa (recorte e modelo de cada rede).
  final Map<String, double> stages;
}

class HandStage {
  /// Palma e pontos na GPU: os modelos do MediaPipe têm pesos fp16 e são feitos
  /// para ela (24 e 21 ms contra 88 e 72 ms na CPU de um Helio G95).
  HandStage(Uint8List palm, Uint8List landmarks)
    : _palmBytes = palm,
      _landmarksBytes = landmarks {
    final opened1 = _open(palm, _order);
    _palm = opened1.$1;
    _palmBackend = opened1.$2;
    final opened2 = _open(landmarks, landmarksOrder);
    _landmarks = opened2.$1;
    _landmarksBackend = opened2.$2;
    _palmRaw = _outputIndex(_palm, 'Identity');
    _palmScore = _outputIndex(_palm, 'Identity_1');
    _lmPoints = _outputIndex(_landmarks, 'Identity');
    _lmPresence = _outputIndex(_landmarks, 'Identity_1');
  }

  static const _order = [Backend.gpu, Backend.xnnpack, Backend.cpu];

  /// Ordem de backends do modelo de pontos (lite). GPU primeiro: na CPU ele leva
  /// 12 ms sozinho, mas 32–58 ms dentro do app, disputando os dois núcleos A76 com
  /// o SignEdge, que roda em paralelo no XNNPACK.
  static const landmarksOrder = [Backend.gpu, Backend.xnnpack, Backend.cpu];

  static (Interpreter, Backend) _open(Uint8List bytes, List<Backend> order) =>
      openVerified(bytes, order);

  static int _outputIndex(Interpreter interpreter, String name) {
    final index = interpreter.getOutputTensors().indexWhere(
      (t) => t.name == name,
    );
    if (index < 0) throw StateError('saída $name ausente');
    return index;
  }

  final Uint8List _palmBytes, _landmarksBytes;
  late Interpreter _palm, _landmarks;
  late Backend _palmBackend, _landmarksBackend;
  late final int _palmRaw, _palmScore, _lmPoints, _lmPresence;

  String get handBackends =>
      '${_palmBackend.name.toUpperCase()} · ${_landmarksBackend.name.toUpperCase()}';

  final _palmIn = Float32List(palmInput * palmInput * 3);
  final _palmBoxes = Float32List(2016 * 18);
  final _palmScores = Float32List(2016);
  final _lmIn = Float32List(landmarkInput * landmarkInput * 3);
  final _lmOut = Float32List(63);
  final _presence = Float32List(1);
  final StageClock _clock = StageClock();

  /// ROI rastreada do frame anterior; null força o detector de palma.
  Roi? _tracked;

  /// Pontos do frame anterior, para detectar salto absurdo no rastreio.
  List<Pt>? _previousPoints;

  void reset() {
    _tracked = null;
    _previousPoints = null;
  }

  HandResult process(FrameImage frame) {
    _clock.start();
    final start = _clock.micros;
    var usedPalm = false;
    var roi = _tracked;
    if (roi == null) {
      roi = _detectPalm(frame);
      usedPalm = true;
    }
    List<Pt>? points = roi == null ? null : _runLandmarks(frame, roi);
    final previous = _previousPoints;
    if (points != null && !usedPalm && previous != null) {
      if (implausibleJump(previous, points)) points = null;
    }
    if (points == null && !usedPalm) {
      // Perdeu a mão rastreada: tenta a palma no mesmo frame.
      usedPalm = true;
      roi = _detectPalm(frame);
      points = roi == null ? null : _runLandmarks(frame, roi);
    }
    if (points == null) {
      _tracked = null;
      _previousPoints = null;
      return HandResult(
        landmarks: null,
        view: null,
        viewPoints: null,
        handMs: (_clock.micros - start) / 1000,
        usedPalm: usedPalm,
        handBackends: handBackends,
        stages: Map.of(_clock.stages),
      );
    }
    // Rastreio: a ROI nova é amortecida contra a que foi usada neste frame. Vinda
    // da palma, recomeça sem amortecer.
    final next = roiFromLandmarks(points);
    _tracked = usedPalm || roi == null ? next : dampRoi(roi, next);
    _previousPoints = points;

    final view = ModelView.fromLandmarks(
      points,
      math.min(frame.width, frame.height),
    );
    // O recorte da vista fica com o isolate do SignEdge, que estaria parado
    // esperando: aqui só vão a vista e os pontos nela.
    final mapped = view.mapLandmarks(points);
    return HandResult(
      landmarks: Float32List.fromList([
        for (final p in points) ...[p.x, p.y],
      ]),
      view: Float32List.fromList([view.center.x, view.center.y, view.side]),
      viewPoints: Float32List.fromList([
        for (final p in mapped) ...[p.x, p.y],
      ]),
      handMs: (_clock.micros - start) / 1000,
      usedPalm: usedPalm,
      handBackends: handBackends,
      stages: Map.of(_clock.stages),
    );
  }

  /// Diagnóstico da palma no logcat (via perfil): estatísticas da entrada e maior
  /// score das 2016 âncoras. [compareReference] também gera o recorte pelo caminho
  /// exato e mede a diferença (custa ~10 ms por chamada; no Redmi Note 10S deu
  /// no máximo 0,055, só da cor pelo vizinho mais próximo). Desligado fora da
  /// depuração: percorre os 110.592 valores da entrada a cada frame de palma.
  static const bool palmDiagnostics = false;
  static const bool compareReference = false;
  final _palmReference = Float32List(palmInput * palmInput * 3);

  Roi? _detectPalm(FrameImage frame) {
    final letterbox = palmLetterbox(frame.width, frame.height);
    frame.sample(letterbox, palmInput, _palmIn, zeroBorder: true);
    _clock.lap('palma.recorte');
    if (palmDiagnostics) {
      if (compareReference && frame is YuvFrame) {
        frame.sample(
          letterbox,
          palmInput,
          _palmReference,
          zeroBorder: true,
          exact: true,
        );
      }
      var low = 1.0, high = 0.0, sum = 0.0, diff = 0.0;
      for (var i = 0; i < _palmIn.length; i++) {
        final value = _palmIn[i];
        if (value < low) low = value;
        if (value > high) high = value;
        sum += value;
        if (compareReference) {
          final d = (value - _palmReference[i]).abs();
          if (d > diff) diff = d;
        }
      }
      _clock.stages['diag.palma_min'] = low;
      _clock.stages['diag.palma_max'] = high;
      _clock.stages['diag.palma_media'] = sum / _palmIn.length;
      if (compareReference) _clock.stages['diag.rapido_vs_ref'] = diff;
      _clock.start(keep: true);
    }
    _palm.getInputTensor(0).setTo(_palmIn.buffer.asUint8List());
    if (!_invokeOnGpu(_palm, _palmBackend)) {
      _palm.close();
      final opened3 = _open(_palmBytes, _order.sublist(1));
      _palm = opened3.$1;
      _palmBackend = opened3.$2;
      // A entrada foi copiada para o interpretador fechado.
      _palm.getInputTensor(0).setTo(_palmIn.buffer.asUint8List());
      invokeNamed(_palm, 'palma');
    }
    _palm.getOutputTensor(_palmRaw).copyTo(_palmBoxes.buffer.asUint8List());
    _palm.getOutputTensor(_palmScore).copyTo(_palmScores.buffer.asUint8List());
    _clock.lap('palma.modelo');
    if (palmDiagnostics) {
      var best = -double.infinity;
      for (final s in _palmScores) {
        if (s > best) best = s;
      }
      _clock.stages['diag.palma_score'] = sigmoid(best);
    }
    final roi = decodePalm(
      _palmBoxes,
      _palmScores,
      frame.width,
      frame.height,
    )?.roi;
    _clock.lap('palma.decode');
    return roi;
  }

  List<Pt>? _runLandmarks(FrameImage frame, Roi roi) {
    frame.sampleNearest(roi.affine(landmarkInput), landmarkInput, _lmIn);
    _clock.lap('pontos.recorte');
    _landmarks.getInputTensor(0).setTo(_lmIn.buffer.asUint8List());
    if (!_invokeOnGpu(_landmarks, _landmarksBackend)) {
      _landmarks.close();
      final opened4 = _open(_landmarksBytes, [
        for (final b in landmarksOrder)
          if (b != _landmarksBackend) b,
      ]);
      _landmarks = opened4.$1;
      _landmarksBackend = opened4.$2;
      _landmarks.getInputTensor(0).setTo(_lmIn.buffer.asUint8List());
      invokeNamed(_landmarks, 'pontos');
    }
    _landmarks.getOutputTensor(_lmPoints).copyTo(_lmOut.buffer.asUint8List());
    _landmarks
        .getOutputTensor(_lmPresence)
        .copyTo(_presence.buffer.asUint8List());
    _clock.lap('pontos.modelo');
    final points = landmarksToFrame(_lmOut, roi);
    final shortSide = math.min(frame.width, frame.height);
    return plausibleHand(_presence[0], points, shortSide) ? points : null;
  }

  /// Na GPU, falha devolve false (o chamador reabre na CPU); fora dela, propaga.
  static bool _invokeOnGpu(Interpreter interpreter, Backend backend) {
    if (backend != Backend.gpu) {
      invokeNamed(interpreter, backend.name);
      return true;
    }
    try {
      interpreter.invoke();
      return true;
    } catch (_) {
      return false;
    }
  }
}

/// Pedido ao isolate do SignEdge. [generation] muda quando a mão some, para a
/// UI descartar classificações de uma mão que já saiu.
class ClassifierRequest {
  const ClassifierRequest(
    this.generation,
    this.planes,
    this.view,
    this.viewPoints,
  );
  final int generation;

  /// O mesmo frame que a mão usou, a vista (cx, cy, lado) e os pontos nela.
  final YuvPlanes planes;
  final Float32List view;
  final Float32List viewPoints;
}

class ClassifierResult {
  const ClassifierResult({
    required this.generation,
    required this.logits,
    required this.ms,
    required this.cropMs,
    required this.backend,
  });
  final int generation;
  final Float32List logits;

  /// Recorte + modelo, e só o recorte.
  final double ms, cropMs;

  /// Onde o SignEdge roda (xnnpack, gpu ou cpu).
  final String backend;
}

class ClassifierStage {
  /// SignEdge Int8 na CPU com o XNNPACK aplicado explicitamente (15–17 ms). O
  /// LiteRT 1.4 do tflite_flutter, no ARM, falha com o XNNPACK padrão nesse modelo
  /// ("Input tensor N lacks data"), e o delegate explícito roda.
  ClassifierStage(Uint8List bytes) {
    final opened5 = openVerified(bytes, const [
      Backend.xnnpack,
      Backend.gpu,
      Backend.cpu,
    ]);
    _interpreter = opened5.$1;
    backend = opened5.$2;
    final inputs = _interpreter.getInputTensors();
    _landmarks = inputs.indexWhere((t) => t.name.contains('landmarks'));
    _rgb = inputs.indexWhere((t) => t.name.contains('rgb'));
    if (_landmarks < 0 || _rgb < 0) {
      throw StateError('model_int8.tflite sem as entradas rgb e landmarks');
    }
  }

  late final Interpreter _interpreter;
  late final Backend backend;
  late final int _landmarks, _rgb;
  final _logits = Float32List(61);
  final _crop = Float32List(CropRule.inputSize * CropRule.inputSize * 3);
  final Stopwatch _clock = Stopwatch()..start();

  ClassifierResult process(ClassifierRequest request) {
    final start = _clock.elapsedMicroseconds;
    final frame = request.planes.materialize();
    final v = request.view;
    final view = Affine.square(v[0], v[1], v[2], 0, CropRule.inputSize);
    frame.sample(view, CropRule.inputSize, _crop);
    final cropped = _clock.elapsedMicroseconds;
    _interpreter.getInputTensor(_rgb).setTo(_crop.buffer.asUint8List());
    _interpreter
        .getInputTensor(_landmarks)
        .setTo(request.viewPoints.buffer.asUint8List());
    invokeNamed(_interpreter, 'SignEdge');
    _interpreter.getOutputTensor(0).copyTo(_logits.buffer.asUint8List());
    return ClassifierResult(
      generation: request.generation,
      logits: Float32List.fromList(_logits),
      ms: (_clock.elapsedMicroseconds - start) / 1000,
      cropMs: (cropped - start) / 1000,
      backend: backend.name,
    );
  }
}
