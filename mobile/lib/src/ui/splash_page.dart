import 'dart:async';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter/services.dart';

/// Abertura do app: a mão soletra E-D-G-E em LIBRAS enquanto os modelos
/// carregam. Só chama [onDone] depois que [ready] completa (com sucesso ou
/// erro) e o ciclo de letras em andamento termina, para não cortar o gesto.
class SplashPage extends StatefulWidget {
  const SplashPage({super.key, required this.ready, required this.onDone});

  final Future<Object?> ready;
  final VoidCallback onDone;

  @override
  State<SplashPage> createState() => _SplashPageState();
}

const _letters = ['E', 'D', 'G', 'E'];

const _bg = Color(0xFF090D0C);
const _bgTop = Color(0xFF16221F);
const _accent = Color(0xFF4FD1BE);
const _muted = Color(0xFF8DA19C);
const _title = Color(0xFFE6EFEC);
const _font = 'Atkinson';

class _SplashPageState extends State<SplashPage> {
  int _index = 0;
  bool _spelled = false; // mostra o nome no lugar dos blocos
  bool _loaded = false;
  bool _visible = false; // fade-in no primeiro frame
  bool _alive = true;
  bool _finished = false;

  @override
  void initState() {
    super.initState();
    // Erro também libera a abertura: a tela seguinte é quem mostra a falha.
    widget.ready.then<void>(
      (_) => _loaded = true,
      onError: (_) => _loaded = true,
    );
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) setState(() => _visible = true);
    });
    unawaited(_run());
  }

  Future<void> _run() async {
    while (_alive && !_finished) {
      for (var i = 0; i < _letters.length; i++) {
        if (!_alive || _finished) return;
        setState(() {
          _index = i;
          _spelled = false;
        });
        await Future<void>.delayed(
          i == 0
              ? const Duration(milliseconds: 550)
              : const Duration(seconds: 1),
        );
      }
      if (!_alive || _finished) return;
      setState(() => _spelled = true);
      await Future<void>.delayed(const Duration(milliseconds: 1350));
      if (_loaded) {
        _finish();
        return;
      }
    }
  }

  void _finish() {
    if (_finished || !_alive) return;
    _finished = true;
    widget.onDone();
  }

  @override
  void dispose() {
    _alive = false;
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final size = (MediaQuery.sizeOf(context).width * 0.62).clamp(180.0, 300.0);
    final hand = SizedBox(
      width: size * 0.98,
      height: size * 0.98,
      child: SpellingHand(letter: _letters[_index]),
    );
    // Tela cheia: o gradiente vai por trás das barras do sistema, que ficam
    // transparentes com ícones claros; só o conteúdo respeita a SafeArea.
    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: const SystemUiOverlayStyle(
        statusBarColor: Colors.transparent,
        statusBarIconBrightness: Brightness.light,
        statusBarBrightness: Brightness.dark,
        systemNavigationBarColor: Colors.transparent,
        systemNavigationBarIconBrightness: Brightness.light,
        systemNavigationBarContrastEnforced: false,
      ),
      child: Material(
        type: MaterialType.transparency,
        child: SizedBox.expand(
          child: DecoratedBox(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [_bgTop, _bg],
              ),
            ),
            child: SafeArea(
              child: AnimatedOpacity(
                opacity: _visible ? 1 : 0,
                curve: const Cubic(0, 0, 0.58, 1),
                duration: const Duration(milliseconds: 380),
                child: Column(
                  children: [
                    const Spacer(flex: 5),
                    _HandHalo(size: size, spelled: _spelled, child: hand),
                    const SizedBox(height: 36),
                    SizedBox(
                      height: 60,
                      child: AnimatedSwitcher(
                        duration: const Duration(milliseconds: 420),
                        switchInCurve: const Interval(
                          0.35,
                          1,
                          curve: Cubic(0.215, 0.61, 0.355, 1),
                        ),
                        switchOutCurve: Curves.easeIn,
                        transitionBuilder: (child, animation) => FadeTransition(
                          opacity: animation,
                          child: ScaleTransition(
                            scale: animation.drive(Tween(begin: 0.94, end: 1)),
                            child: child,
                          ),
                        ),
                        child: _spelled
                            ? const _AppName(key: ValueKey('nome'))
                            : _LetterBlocks(
                                key: const ValueKey('blocos'),
                                letters: _letters,
                                current: _index,
                              ),
                      ),
                    ),
                    const Spacer(flex: 4),
                    const _LoadingStatus(),
                    const SizedBox(height: 28),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _HandHalo extends StatelessWidget {
  const _HandHalo({
    required this.size,
    required this.spelled,
    required this.child,
  });

  final double size;
  final bool spelled;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    final inner = size * 0.86;
    return SizedBox(
      width: size,
      height: size,
      child: Stack(
        alignment: Alignment.center,
        children: [
          AnimatedContainer(
            curve: const Cubic(0, 0, 0.58, 1),
            duration: const Duration(milliseconds: 500),
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: RadialGradient(
                radius: 0.5,
                colors: [
                  _accent.withValues(alpha: spelled ? 0.2 : 0.1),
                  _accent.withValues(alpha: spelled ? 0.06 : 0.03),
                  _accent.withValues(alpha: 0),
                ],
                stops: const [0, 0.62, 1],
              ),
            ),
          ),
          Container(
            width: inner,
            height: inner,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: Colors.white.withValues(alpha: 0.025),
              border: Border.all(color: _accent.withValues(alpha: 0.14)),
            ),
          ),
          child,
        ],
      ),
    );
  }
}

class _LetterBlocks extends StatelessWidget {
  const _LetterBlocks({
    super.key,
    required this.letters,
    required this.current,
  });

  final List<String> letters;
  final int current;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        for (var i = 0; i < letters.length; i++)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 5),
            child: _LetterBlock(
              letter: letters[i],
              state: i < current
                  ? _BlockState.done
                  : (i == current ? _BlockState.current : _BlockState.pending),
            ),
          ),
      ],
    );
  }
}

enum _BlockState { pending, done, current }

class _LetterBlock extends StatelessWidget {
  const _LetterBlock({required this.letter, required this.state});

  final String letter;
  final _BlockState state;

  @override
  Widget build(BuildContext context) {
    const duration = Duration(milliseconds: 260);
    final current = state == _BlockState.current;
    final done = state == _BlockState.done;
    return AnimatedScale(
      scale: current ? 1.08 : 1,
      curve: const Cubic(0.175, 0.885, 0.32, 1.275),
      duration: duration,
      child: AnimatedContainer(
        width: 46,
        height: 54,
        alignment: Alignment.center,
        curve: const Cubic(0, 0, 0.58, 1),
        duration: duration,
        decoration: BoxDecoration(
          color: current
              ? _accent
              : (done ? const Color(0xFF0B2420) : Colors.transparent),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            width: 1.5,
            color: current
                ? _accent
                : (done
                      ? _accent.withValues(alpha: 0.55)
                      : Colors.white.withValues(alpha: 0.16)),
          ),
          boxShadow: current
              ? [
                  BoxShadow(
                    color: _accent.withValues(alpha: 0.35),
                    blurRadius: 18,
                    spreadRadius: -2,
                  ),
                ]
              : const [],
        ),
        child: AnimatedDefaultTextStyle(
          duration: const Duration(milliseconds: 200),
          style: TextStyle(
            fontFamily: _font,
            fontSize: 24,
            fontWeight: FontWeight.w700,
            height: 1,
            color: current
                ? _bg
                : (done ? _accent : Colors.white.withValues(alpha: 0.38)),
          ),
          child: Text(letter),
        ),
      ),
    );
  }
}

class _AppName extends StatelessWidget {
  const _AppName({super.key});

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Text.rich(
        TextSpan(
          children: [
            TextSpan(text: 'Sign'),
            TextSpan(
              text: 'Edge',
              style: TextStyle(color: _accent),
            ),
          ],
        ),
        style: TextStyle(
          fontFamily: _font,
          fontSize: 34,
          fontWeight: FontWeight.w700,
          letterSpacing: -0.5,
          color: _title,
        ),
      ),
    );
  }
}

class _LoadingStatus extends StatelessWidget {
  const _LoadingStatus();

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        const Text(
          'Carregando modelos…',
          style: TextStyle(
            fontFamily: _font,
            fontSize: 14,
            letterSpacing: 0.2,
            color: _muted,
          ),
        ),
        const SizedBox(height: 12),
        ClipRRect(
          borderRadius: BorderRadius.circular(2),
          child: const SizedBox(
            width: 96,
            height: 3,
            child: LinearProgressIndicator(
              color: _accent,
              backgroundColor: Color(0x1F4FD1BE),
              semanticsLabel: 'Carregando modelos',
            ),
          ),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Mão desenhada (vista de frente, palma para o observador).

/// Pose da mão: flexão (graus) das 3 falanges de cada dedo (indicador ao
/// mínimo), ângulos absolutos dos 3 segmentos do polegar e a profundidade do
/// polegar, que decide se ele é desenhado atrás ou na frente dos dedos.
@immutable
class HandPose {
  const HandPose(this.fingers, this.thumb, this.thumbDepth);

  final List<List<double>> fingers;
  final List<double> thumb;
  final double thumbDepth;
}

const _curled = [55.0, 100.0, 25.0];
const _straight = [0.0, 4.0, 2.0];
const _fist = [70.0, 100.0, 70.0];

const handPoses = <String, HandPose>{
  'E': HandPose([_curled, _curled, _curled, _curled], [74, 80, 90], 6),
  'D': HandPose(
    [
      _straight,
      [35, 95, 50],
      _fist,
      _fist,
    ],
    [36, 38, 52],
    22,
  ),
  'G': HandPose([_straight, _fist, _fist, _fist], [60, 63, 68], 22),
};

double _smoothstep(double a, double b, double x) {
  final t = ((x - a) / (b - a)).clamp(0.0, 1.0);
  return t * t * (3 - 2 * t);
}

const _fingerDelay = [0.0, 0.08, 0.15, 0.22];
// O polegar se recolhe no meio da troca para não atravessar os dedos.
const _thumbTuck = [-16.0, -30.0, -34.0];

HandPose _lerpPose(HandPose a, HandPose b, double t) {
  final fingers = List.generate(4, (i) {
    final ft = _smoothstep(_fingerDelay[i], _fingerDelay[i] + 0.74, t);
    return List.generate(
      3,
      (j) => a.fingers[i][j] + (b.fingers[i][j] - a.fingers[i][j]) * ft,
    );
  });
  final n = _smoothstep(0.04, 0.78, t);
  final tuck = math.sin(math.pi * n);
  final thumb = List.generate(
    3,
    (j) => a.thumb[j] + (b.thumb[j] - a.thumb[j]) * n + _thumbTuck[j] * tuck,
  );
  final depth =
      a.thumbDepth + (b.thumbDepth - a.thumbDepth) * _smoothstep(0.35, 0.65, n);
  return HandPose(fingers, thumb, depth);
}

/// Leve oscilação para a mão não parecer congelada.
HandPose _breathe(HandPose p, double seconds) {
  final fingers = List.generate(4, (i) {
    return List.generate(3, (j) {
      final v = p.fingers[i][j];
      final s = math.sin(seconds / 0.9 + i * 1.3 + j * 0.7);
      return v + s * (v > 20 ? 1.6 : 0.8);
    });
  });
  final thumb = List.generate(
    3,
    (j) => p.thumb[j] + math.sin(seconds / 1.1 + j) * 0.8,
  );
  return HandPose(fingers, thumb, p.thumbDepth);
}

class SpellingHand extends StatefulWidget {
  const SpellingHand({
    super.key,
    required this.letter,
    this.duration = const Duration(milliseconds: 450),
  });

  final String letter;
  final Duration duration;

  @override
  State<SpellingHand> createState() => _SpellingHandState();
}

class _SpellingHandState extends State<SpellingHand>
    with SingleTickerProviderStateMixin {
  late final ValueNotifier<HandPose> _pose = ValueNotifier(
    _poseFor(widget.letter),
  );
  late HandPose _from = _pose.value;
  late HandPose _to = _pose.value;
  double _t = 1;
  Duration _now = Duration.zero;
  Duration _changedAt = Duration.zero;
  late final Ticker _ticker;

  @override
  void initState() {
    super.initState();
    _ticker = createTicker(_tick)..start();
  }

  static HandPose _poseFor(String letter) =>
      handPoses[letter] ?? handPoses['E']!;

  HandPose get _current => _t < 1 ? _lerpPose(_from, _to, _t) : _to;

  @override
  void didUpdateWidget(SpellingHand old) {
    super.didUpdateWidget(old);
    if (old.letter != widget.letter) {
      _from = _current;
      _to = _poseFor(widget.letter);
      _t = 0;
      _changedAt = _now;
    }
  }

  void _tick(Duration elapsed) {
    _now = elapsed;
    final reduceMotion = MediaQuery.maybeDisableAnimationsOf(context) ?? false;
    if (_t < 1) {
      final ms = widget.duration.inMilliseconds;
      _t = reduceMotion || ms <= 0
          ? 1
          : ((elapsed - _changedAt).inMicroseconds / 1000 / ms).clamp(0.0, 1.0);
      if (_t >= 1) _from = _to;
    }
    var pose = _current;
    if (!reduceMotion) pose = _breathe(pose, elapsed.inMicroseconds / 1e6);
    _pose.value = pose;
  }

  @override
  void dispose() {
    _ticker.dispose();
    _pose.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Semantics(
      label: 'Mão fazendo a letra ${widget.letter} em LIBRAS',
      image: true,
      child: CustomPaint(size: Size.infinite, painter: _HandPainter(_pose)),
    );
  }
}

class _Finger {
  const _Finger(this.x, this.y, this.angle, this.width, this.lengths);

  final double x, y, angle, width;
  final List<double> lengths;
}

const _fingerBases = [
  _Finger(39.5, 47.5, -5, 10.4, [15, 11, 9.5]),
  _Finger(50, 46.5, -1, 11, [16.5, 12, 10]),
  _Finger(60, 47.5, 3, 10.4, [15.5, 11.5, 9.5]),
  _Finger(67.4, 49.6, 2.5, 9.4, [12.5, 9, 8]),
];
const _fingerTaper = [0.5, 0.465, 0.43, 0.39];
const _thumbBase = Offset(33.6, 73.8);
const _thumbLengths = [4.0, 14.0, 11.0];
const _thumbRadii = [4.6, 5.6, 5.0, 4.4];

const _skinLight = Color(0xFFF4D2BA);
const _skinMid = Color(0xFFE5B08F);
const _skinDark = Color(0xFFC98A69);
const _outline = Color(0x996E3A28);
const _shadow = Color(0x55301206);
const _innerShade = Color(0x226E2E18);
const _palmShade = Color(0x2A6E2E18);
const _crease = Color(0x40A0624A);
const _knuckle = Color(0x66A0624A);
const _nail = Color(0xFFF2D5C7);
const _nailEdge = Color(0x80B4765E);
const _skinColors = [_skinLight, _skinMid, _skinDark];

class _Part {
  _Part(
    this.path,
    this.depth,
    this.points,
    this.angles,
    this.cumulative,
    this.radii,
    this.isFinger,
  );

  final Path path;
  final double depth;
  final List<Offset> points;
  final List<double> angles;
  final List<double> cumulative;
  final List<double> radii;
  final bool isFinger;
}

/// Segmento afunilado: dois círculos ligados pelas tangentes externas.
Path _capsule(Offset a, double ra, Offset b, double rb) {
  final d = b - a;
  final len = d.distance;
  final dr = ra - rb;
  if (len <= dr.abs() + 0.01) {
    return Path()..addOval(
      Rect.fromCircle(center: ra >= rb ? a : b, radius: math.max(ra, rb)),
    );
  }
  final base = math.atan2(d.dy, d.dx);
  final q = math.acos(dr / len);
  Offset polar(double ang, double r) =>
      Offset(r * math.cos(ang), r * math.sin(ang));
  final start = a + polar(base + q, ra);
  final end = b + polar(base - q, rb);
  return Path()
    ..moveTo(start.dx, start.dy)
    ..arcTo(
      Rect.fromCircle(center: a, radius: ra),
      base + q,
      2 * math.pi - 2 * q,
      false,
    )
    ..lineTo(end.dx, end.dy)
    ..arcTo(Rect.fromCircle(center: b, radius: rb), base - q, 2 * q, false)
    ..close();
}

Path _union(List<Path> paths) =>
    paths.reduce((a, b) => Path.combine(PathOperation.union, a, b));

class _HandPainter extends CustomPainter {
  _HandPainter(this.pose) : super(repaint: pose);

  final ValueNotifier<HandPose> pose;

  Path? _thumbJoint;
  Shader? _palmShader;
  late Offset _thumbFrom, _thumbTo;

  @override
  void paint(Canvas canvas, Size size) {
    final scale = math.min(size.height / 94, size.width / 60);
    canvas
      ..save()
      ..translate(size.width / 2, size.height / 2)
      ..scale(scale)
      ..translate(-52, -53);
    _paintHand(canvas, pose.value);
    canvas.restore();
  }

  void _paintHand(Canvas canvas, HandPose p) {
    final parts = <_Part>[];
    for (var o = 0; o < 4; o++) {
      final f = _fingerBases[o];
      final a = f.angle * math.pi / 180;
      final dx = math.sin(a), dy = -math.cos(a);
      final pts = [Offset(f.x, f.y)];
      final cum = <double>[];
      var h = 0.0, g = 0.0, depth = 0.0;
      for (var e = 0; e < 3; e++) {
        h += p.fingers[o][e];
        final r = h * math.pi / 180;
        final proj = f.lengths[e] * math.cos(r);
        g += f.lengths[e] * math.sin(r);
        depth = math.max(depth, g);
        pts.add(pts.last + Offset(dx * proj, dy * proj));
        cum.add(h);
      }
      final radii = [for (final t in _fingerTaper) t * f.width];
      parts.add(
        _Part(
          _union([
            for (var e = 0; e < 3; e++)
              _capsule(pts[e], radii[e], pts[e + 1], radii[e + 1]),
          ]),
          depth,
          pts,
          p.fingers[o],
          cum,
          radii,
          true,
        ),
      );
    }

    final t = [_thumbBase];
    for (var e = 0; e < 3; e++) {
      final ang = (p.thumb[e] - 90) * math.pi / 180;
      t.add(
        t.last +
            Offset(
              _thumbLengths[e] * math.cos(ang),
              _thumbLengths[e] * math.sin(ang),
            ),
      );
    }
    final mid = t[1] + (t[2] - t[1]) * 0.4;
    final thumbBase = _union([
      _capsule(t[0], 4.6, t[1], 5.6),
      _capsule(t[1], 5.6, mid, 5.4),
    ]);
    _thumbJoint = Path()..addOval(Rect.fromCircle(center: t[1], radius: 6.44));
    _thumbFrom = t[1];
    _thumbTo = t[2] + (t[3] - t[2]) * 0.2;
    parts.add(
      _Part(
        _union([_capsule(t[1], 5.6, t[2], 5), _capsule(t[2], 5, t[3], 4.4)]),
        p.thumbDepth,
        t,
        const [],
        const [],
        _thumbRadii,
        false,
      ),
    );

    // Pulso.
    const wrist = Rect.fromLTRB(39, 78, 65, 99.5);
    canvas.drawRRect(
      RRect.fromRectAndCorners(
        wrist,
        topLeft: const Radius.circular(4),
        topRight: const Radius.circular(4),
      ),
      Paint()
        ..shader = LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [_skinDark, _skinDark, _skinDark.withValues(alpha: 0)],
          stops: const [0, 0.4, 1],
        ).createShader(wrist),
    );

    // Dedos estendidos ficam atrás da palma; os dobrados, na frente.
    final behind = parts.where((e) => e.depth <= 2).toList();
    final front = parts.where((e) => e.depth > 2).toList()
      ..sort((a, b) => a.depth.compareTo(b.depth));
    for (final part in behind) {
      _paintPart(canvas, part, false);
    }

    final palm = Path()
      ..moveTo(32, 53)
      ..cubicTo(32, 47.5, 35.5, 45, 41, 45)
      ..lineTo(65, 45)
      ..cubicTo(70.5, 45, 73, 48, 73, 53.5)
      ..lineTo(72.2, 71)
      ..cubicTo(71.5, 80, 66.5, 86, 58.5, 87)
      ..lineTo(45, 87)
      ..cubicTo(37.5, 86.5, 33, 82, 31.5, 76)
      ..cubicTo(28.5, 72, 28.5, 64, 32, 59)
      ..close();
    final hand = Path.combine(PathOperation.union, palm, thumbBase);
    _palmShader = const RadialGradient(
      center: Alignment(-0.2, -0.15),
      radius: 0.85,
      colors: _skinColors,
      stops: [0, 0.6, 1],
    ).createShader(hand.getBounds());
    canvas
      ..drawPath(
        hand,
        Paint()
          ..color = _outline
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.4,
      )
      ..drawPath(hand, Paint()..shader = _palmShader)
      ..save()
      ..clipPath(hand)
      ..drawPath(
        hand,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 5
          ..color = _palmShade
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 2.5),
      )
      ..restore();

    final crease = Paint()
      ..color = _crease
      ..style = PaintingStyle.stroke
      ..strokeWidth = 0.8
      ..strokeCap = StrokeCap.round;
    canvas
      ..drawPath(
        Path()
          ..moveTo(36, 72)
          ..quadraticBezierTo(46, 67.5, 60, 70.5),
        crease,
      )
      ..drawPath(
        Path()
          ..moveTo(44, 85.6)
          ..quadraticBezierTo(52, 87.2, 60, 85.6),
        crease,
      );

    for (final part in front) {
      _paintPart(canvas, part, true);
    }
  }

  /// O polegar some perto da articulação para se fundir à base na palma.
  Shader _thumbFade(Color c) => ui.Gradient.linear(
    _thumbFrom,
    _thumbTo,
    [c.withValues(alpha: 0), c.withValues(alpha: 0), c],
    const [0, 0.22, 0.62],
  );

  void _paintPart(Canvas canvas, _Part part, bool inFront) {
    final path = part.path;
    final bounds = path.getBounds();
    final joint = part.isFinger ? null : _thumbJoint;
    final isThumb = joint != null;

    if (inFront) {
      final shadow = Paint()
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 1.6);
      if (isThumb) {
        shadow.shader = _thumbFade(_shadow);
      } else {
        shadow.color = _shadow;
      }
      canvas.drawPath(path.shift(const Offset(0.4, 1.3)), shadow);
    }

    final stroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2;
    if (isThumb) {
      stroke.shader = _thumbFade(_outline);
    } else {
      stroke.color = _outline;
    }
    canvas.drawPath(path, stroke);

    canvas.drawPath(
      path,
      Paint()
        ..shader = isThumb && _palmShader != null
            ? _palmShader
            : const LinearGradient(
                begin: Alignment.centerLeft,
                end: Alignment.centerRight,
                colors: _skinColors,
                stops: [0.05, 0.55, 1],
              ).createShader(bounds),
    );

    canvas
      ..save()
      ..clipPath(path);
    if (isThumb) {
      canvas.clipPath(
        Path.combine(
          PathOperation.difference,
          Path()..addRect(bounds.inflate(10)),
          joint,
        ),
      );
    }
    canvas
      ..drawPath(
        path,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3
          ..color = _innerShade
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 1.2),
      )
      ..restore();

    if (!part.isFinger) {
      _paintThumbNail(canvas, part);
      return;
    }

    final knuckle = Paint()
      ..color = _knuckle
      ..style = PaintingStyle.stroke
      ..strokeWidth = 0.6
      ..strokeCap = StrokeCap.round;
    final pts = part.points;
    for (var n = 1; n < 3; n++) {
      if (part.angles[n] > 30) continue;
      final d = pts[n + 1] - pts[n];
      final len = d.distance;
      if (len < 1) continue;
      final w = part.radii[n] * 0.55;
      final perp = Offset(-d.dy / len * w, d.dx / len * w);
      canvas.drawLine(pts[n] + perp, pts[n] - perp, knuckle);
    }

    // Unha aparece quando a ponta do dedo dobra de volta para o observador.
    final alpha = _smoothstep(120, 170, part.cumulative[2]);
    if (alpha <= 0.01) return;
    final tip = pts[3];
    var dir = tip - pts[2];
    final len = dir.distance;
    dir = len < 0.5 ? const Offset(0, 1) : dir / len;
    final r = part.radii[3];
    final half = math.max(1.6, math.min(len * 0.5, r * 0.95));
    _paintNail(
      canvas,
      tip - dir * (half * 0.55),
      dir,
      half * 0.85,
      r * 0.6,
      alpha * 0.85,
    );
  }

  void _paintThumbNail(Canvas canvas, _Part part) {
    final a = part.points[2], b = part.points[3];
    var dir = b - a;
    final len = dir.distance;
    if (len < 0.5) return;
    dir = dir / len;
    final half = math.min(len * 0.36, 2.8);
    _paintNail(canvas, b - dir * (half * 0.9), dir, half, 2.552, 0.8);
  }

  void _paintNail(
    Canvas canvas,
    Offset center,
    Offset dir,
    double halfLength,
    double halfWidth,
    double alpha,
  ) {
    final nail = RRect.fromRectAndRadius(
      Rect.fromCenter(
        center: Offset.zero,
        width: halfLength * 2,
        height: halfWidth * 2,
      ),
      Radius.circular(halfWidth),
    );
    canvas
      ..save()
      ..translate(center.dx, center.dy)
      ..rotate(math.atan2(dir.dy, dir.dx))
      ..drawRRect(nail, Paint()..color = _nail.withValues(alpha: alpha * 0.95))
      ..drawRRect(
        nail,
        Paint()
          ..color = _nailEdge.withValues(alpha: 0.5 * alpha)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 0.5,
      )
      ..restore();
  }

  @override
  bool shouldRepaint(_HandPainter old) => old.pose != pose;
}
