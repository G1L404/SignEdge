/// Estado do app: câmera, os dois isolates de inferência, telemetria e decisão
/// sobre as 61 configurações de mão.
///
/// Pipeline em dois estágios: a mão (GPU) recebe um frame novo assim que devolve
/// o anterior; o SignEdge (CPU) classifica o recorte mais recente. Se ele ainda
/// está ocupado, só o último recorte espera: a fila nunca cresce.
library;

import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter/services.dart';

import 'logic/decision.dart';
import 'vision/inference_isolate.dart';
import 'vision/frame.dart';
import 'vision/pipeline.dart';

class AppController extends ChangeNotifier {
  AppController._(this._hand, this._classifier, this._cameras, this.modelBytes);

  static Future<AppController> create() async {
    Future<Uint8List> bytes(String asset) async =>
        (await rootBundle.load(asset)).buffer.asUint8List();
    final models = ModelBytes(
      palm: await bytes('assets/hand_detector.tflite'),
      landmarks: await bytes('assets/hand_landmark_lite.tflite'),
      classifier: await bytes('assets/model_int8.tflite'),
    );
    final hand = await Worker.spawn(handWorker, (
      models.palm,
      models.landmarks,
    ));
    final classifier = await Worker.spawn(classifierWorker, models.classifier);
    final controller = AppController._(
      hand,
      classifier,
      await availableCameras(),
      models.classifier.lengthInBytes,
    );
    await controller._open(CameraLensDirection.back);
    return controller;
  }

  /// Tamanho do model_int8.tflite embarcado, para o selo da telemetria.
  final int modelBytes;
  final Worker _hand, _classifier;
  final List<CameraDescription> _cameras;
  CameraController? camera;

  /// EMA contínua nos logits (0,35 para o frame novo), limpa no frame em que a mão
  /// some, e a histerese por cima: entrar pede 2 frames seguidos acima do limiar;
  /// sair, cair abaixo de limiar − 0,12. Assim o card não pisca com a mão respirando.
  final LogitEma smoother = LogitEma(alpha: 0.35);
  final Stabilizer stabilizer = Stabilizer(enterFrames: 2, release: 0.12);
  final Stopwatch _clock = Stopwatch()..start();

  /// Limiar de confiança (probabilidade calibrada) para aceitar uma CM.
  ///
  /// 0,45: nas 1.160 pausas fora do fold do modelo podado (T = 0,48), o app
  /// responde em 86,3% das pausas com 80,2% de acerto entre as respondidas (0,30:
  /// 94,7% e 77,5%; 0,70: 70,1% e 86,5%). O slider ajusta ao vivo.
  double threshold = 0.45;

  // Último frame processado, em coordenadas do frame em pé.
  int frameWidth = 0, frameHeight = 0;
  Float32List? landmarks;
  Float32List? view;
  Decision decision = Decision.neutral;

  // Telemetria.
  double fps = 0, handMs = 0, classifierMs = 0, classifierCropMs = 0;
  bool usedPalm = false;
  String handBackends = '';
  String classifierBackend = '';
  String? error;

  bool _handBusy = false, _classifierBusy = false;
  ClassifierRequest? _pendingCrop;

  /// Muda quando a mão some: classificações de uma mão que já saiu são descartadas.
  int _generation = 0;
  Duration? _lastFrame;

  bool get isFront =>
      camera?.description.lensDirection == CameraLensDirection.front;

  bool get canSwitch => _cameras.length > 1;

  Future<void> _open(CameraLensDirection lens) async {
    final description = _cameras.firstWhere(
      (c) => c.lensDirection == lens,
      orElse: () => _cameras.first,
    );
    final controller = CameraController(
      description,
      // 640×480: a mesma resolução do Kinect da base.
      ResolutionPreset.medium,
      // Sem `fps` fixo: no ISP MediaTek a faixa [30, 30] trava o auto-exposure e a
      // imagem fica escura demais para o detector. O driver escolhe a faixa.
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    await controller.initialize();
    for (final setup in [
      () => controller.setExposureMode(ExposureMode.auto),
      () => controller.setFocusMode(FocusMode.auto),
    ]) {
      try {
        await setup();
      } catch (_) {
        // Câmera sem esse controle (ex.: frontal de foco fixo): segue no padrão.
      }
    }
    camera = controller;
    unawaited(_hand.call(const ResetTracking()));
    _forgetHand();
    await controller.startImageStream(_onImage);
    notifyListeners();
  }

  Future<void> switchCamera() async {
    final old = camera;
    if (old == null || !canSwitch) return;
    // A direção nova sai da câmera atual, antes de zerar `camera` (senão `isFront`
    // já seria falso e a troca voltaria sempre para a frontal).
    final next = old.description.lensDirection == CameraLensDirection.front
        ? CameraLensDirection.back
        : CameraLensDirection.front;
    camera = null;
    notifyListeners();
    await old.stopImageStream();
    await old.dispose();
    await _open(next);
  }

  void setThreshold(double value) {
    // A decisão nova sai no próximo frame (≤ 50 ms): recalcular aqui contaria um
    // frame a mais na histerese.
    threshold = value;
    notifyListeners();
  }

  final _Profile _profile = _Profile();

  void _onImage(CameraImage image) {
    _profile.delivered++;
    final current = camera;
    // Durante a troca de câmera ainda podem chegar frames da anterior.
    if (_handBusy || current == null || image.planes.length < 3) return;
    _handBusy = true;
    // Aparelho travado em retrato: girar o sensor por sensorOrientation deixa o frame
    // em pé. Na frontal o frame cru não vem espelhado; só o preview é espelho.
    final rotation = current.description.sensorOrientation;
    final sent = _clock.elapsedMicroseconds;
    // Uma cópia dos planos para cada isolate: a mão agora, o SignEdge se houver mão.
    YuvPlanes planes() => YuvPlanes(
      y: image.planes[0].bytes,
      u: image.planes[1].bytes,
      v: image.planes[2].bytes,
      width: image.width,
      height: image.height,
      rotation: rotation,
      yRowStride: image.planes[0].bytesPerRow,
      uvRowStride: image.planes[1].bytesPerRow,
      uvPixelStride: image.planes[1].bytesPerPixel ?? 1,
    );
    final request = planes();
    final copied = _clock.elapsedMicroseconds;
    final swap = rotation % 180 != 0;
    _hand
        .call(request)
        .then((reply) {
          final result = reply as HandResult;
          frameWidth = swap ? image.height : image.width;
          frameHeight = swap ? image.width : image.height;
          error = null;
          _applyHand(result, planes);
          _profile.addHand(
            result,
            copyMs: (copied - sent) / 1000,
            totalMs: (_clock.elapsedMicroseconds - sent) / 1000,
            nowUs: _clock.elapsedMicroseconds,
          );
        })
        .catchError((Object e) {
          error = '$e';
          debugPrint('SignEdge: $e');
        })
        .whenComplete(() {
          _handBusy = false;
          _requestUi();
        });
  }

  void _applyHand(HandResult result, YuvPlanes Function() planes) {
    final now = _clock.elapsed;
    if (_lastFrame != null) {
      final dt = (now - _lastFrame!).inMicroseconds / 1e6;
      if (dt > 0) fps = fps == 0 ? 1 / dt : 0.9 * fps + 0.1 / dt;
    }
    _lastFrame = now;
    handMs = 0.8 * handMs + 0.2 * result.handMs;
    usedPalm = result.usedPalm;
    handBackends = result.handBackends;
    // EMA só no desenho e só no rastreio: a palma redetectada recomeça do zero, e o
    // SignEdge recebe os pontos crus.
    final smooth = !result.usedPalm;
    landmarks = _ease(landmarks, result.landmarks, smooth);
    view = _ease(view, result.view, smooth);
    if (!result.hasHand) {
      // Sempre: também descarta a classificação que ainda está a caminho.
      _forgetHand();
      _decide();
      return;
    }
    final request = ClassifierRequest(
      _generation,
      planes(),
      result.view!,
      result.viewPoints!,
    );
    if (_classifierBusy) {
      _pendingCrop = request; // só o mais recente espera
    } else {
      _classify(request);
    }
  }

  /// Peso do frame novo no EMA visual: 0,6 tira o tremor sem arrastar o esqueleto
  /// (meia-vida de ~1 frame).
  static const double drawSmoothing = 0.6;

  static Float32List? _ease(Float32List? old, Float32List? next, bool smooth) {
    if (next == null || old == null || !smooth || old.length != next.length) {
      return next;
    }
    return Float32List.fromList([
      for (var i = 0; i < next.length; i++)
        old[i] + drawSmoothing * (next[i] - old[i]),
    ]);
  }

  void _forgetHand() {
    _generation++;
    _pendingCrop = null;
    smoother.clear();
    stabilizer.reset();
  }

  void _classify(ClassifierRequest request) {
    _classifierBusy = true;
    _classifier
        .call(request)
        .then((reply) {
          final result = reply as ClassifierResult;
          _profile.addClassifier(result);
          classifierCropMs = 0.8 * classifierCropMs + 0.2 * result.cropMs;
          classifierBackend = result.backend;
          classifierMs = 0.8 * classifierMs + 0.2 * result.ms;
          if (result.generation != _generation) return;
          smoother.add(result.logits);
          _decide();
        })
        .catchError((Object e) {
          error = '$e';
          debugPrint('SignEdge: $e');
        })
        .whenComplete(() {
          _classifierBusy = false;
          final next = _pendingCrop;
          _pendingCrop = null;
          if (next != null) _classify(next);
          _requestUi();
        });
  }

  void _decide() {
    final logits = smoother.value;
    if (logits == null) {
      decision = Decision.neutral;
      return;
    }
    final probs = softmax(logits);
    decision = stabilizer.step(decide(probs, threshold), probs, threshold);
  }

  bool _uiScheduled = false;

  /// No máximo uma reconstrução da tela por quadro do display: a mão e o SignEdge
  /// terminam em momentos diferentes, e cada um pedia a sua (a GPU que desenha a
  /// tela é a mesma que roda o modelo de pontos).
  void _requestUi() {
    if (_uiScheduled) return;
    _uiScheduled = true;
    SchedulerBinding.instance.scheduleFrameCallback((_) {
      _uiScheduled = false;
      notifyListeners();
    });
  }

  @override
  void dispose() {
    camera?.dispose();
    _hand.dispose();
    _classifier.dispose();
    super.dispose();
  }
}

/// Perfil do pipeline no aparelho, impresso no logcat a cada 2 s (tag "flutter").
class _Profile {
  int delivered = 0;
  int _hands = 0, _classified = 0;
  int _since = -1;
  double _copy = 0, _total = 0, _classifierMs = 0;
  final Map<String, double> _stages = {};
  final Map<String, (double, int, double)> _diag = {}; // soma, n, máximo

  double _classifierCropMs = 0;

  void addClassifier(ClassifierResult result) {
    _classified++;
    _classifierMs += result.ms;
    _classifierCropMs += result.cropMs;
  }

  void addHand(
    HandResult result, {
    required double copyMs,
    required double totalMs,
    required int nowUs,
  }) {
    if (_since < 0) _since = nowUs;
    _hands++;
    _copy += copyMs;
    _total += totalMs;
    result.stages.forEach((k, v) {
      if (k.startsWith('diag.')) {
        final (sum, n, top) = _diag[k] ?? (0.0, 0, -double.infinity);
        _diag[k] = (sum + v, n + 1, v > top ? v : top);
      } else {
        _stages[k] = (_stages[k] ?? 0) + v;
      }
    });
    final seconds = (nowUs - _since) / 1e6;
    if (seconds < 2) return;
    String ms(double v, int n) => n == 0 ? '-' : (v / n).toStringAsFixed(1);
    String rate(int n) => (n / seconds).toStringAsFixed(1);
    debugPrint(
      'PERFIL câmera ${rate(delivered)} fps, mão ${rate(_hands)} fps, '
      'SignEdge ${rate(_classified)} fps (${ms(_classifierMs, _classified)} ms, '
      'recorte ${ms(_classifierCropMs, _classified)}) | '
      'mão ponta a ponta ${ms(_total, _hands)} ms, cópia ${ms(_copy, _hands)} | '
      '${_stages.entries.map((e) => '${e.key} ${ms(e.value, _hands)}').join(', ')}',
    );
    if (_diag.isNotEmpty) {
      debugPrint(
        'DIAG ${_diag.entries.map((e) => '${e.key.substring(5)} média ${(e.value.$1 / e.value.$2).toStringAsFixed(3)} máx ${e.value.$3.toStringAsFixed(3)}').join(' | ')}',
      );
    }
    delivered = _hands = _classified = 0;
    _since = nowUs;
    _copy = _total = _classifierMs = _classifierCropMs = 0;
    _stages.clear();
    _diag.clear();
  }
}
