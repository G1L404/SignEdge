import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../logic/canonical_hands.dart';
import '../logic/decision.dart';

/// Cores por dedo: polegar, indicador, médio, anelar e mínimo.
const List<Color> fingerColors = [
  Color(0xFFF97316),
  Color(0xFF2563EB),
  Color(0xFF16A34A),
  Color(0xFF9333EA),
  Color(0xFFDB2777),
];
const List<String> fingerNames = [
  'polegar',
  'indicador',
  'médio',
  'anelar',
  'mínimo',
];

/// Falanges de cada dedo, da base à ponta.
const List<List<(int, int)>> _phalanges = [
  [(1, 2), (2, 3), (3, 4)],
  [(5, 6), (6, 7), (7, 8)],
  [(9, 10), (10, 11), (11, 12)],
  [(13, 14), (14, 15), (15, 16)],
  [(17, 18), (18, 19), (19, 20)],
];

/// Meia largura da janela, em raios RMS: cabe a mão mais aberta das 61 (2,06).
const double _window = 2.2;

const Color _skin = Color(0xFFE0AA8C);

/// Desenho vetorial leve de uma CM, para quando o sprite sheet não foi carregado:
/// palma em tom de pele e as 15 falanges na cor do dedo, da mais distante à mais
/// próxima (z). A escala é a mesma para as 61 CMs.
class CanonicalHandPainter extends CustomPainter {
  CanonicalHandPainter(this.code);

  final int code;

  @override
  void paint(Canvas canvas, Size size) {
    final p = canonicalHands[code];
    final side = math.min(size.width, size.height);
    final scale = side / (2 * _window);
    Offset at(int i) =>
        size.center(Offset.zero) + Offset(p[3 * i], p[3 * i + 1]) * scale;

    final palm = Path()
      ..addPolygon([
        for (final i in [0, 1, 5, 9, 13, 17]) at(i),
      ], true);
    canvas.drawPath(palm, Paint()..color = _skin);
    canvas.drawPath(
      palm,
      Paint()
        ..color = _skin
        ..style = PaintingStyle.stroke
        ..strokeWidth = side / 9
        ..strokeJoin = StrokeJoin.round,
    );
    final segments = [
      for (var f = 0; f < 5; f++)
        for (final (a, b) in _phalanges[f])
          (p[3 * a + 2] + p[3 * b + 2], f, a, b),
    ]..sort((s, t) => s.$1.compareTo(t.$1));
    for (final (_, f, a, b) in segments) {
      canvas.drawLine(
        at(a),
        at(b),
        Paint()
          ..color = fingerColors[f]
          ..strokeWidth = side / 10
          ..strokeCap = StrokeCap.round,
      );
    }
  }

  @override
  bool shouldRepaint(CanonicalHandPainter old) => old.code != code;
}

/// As 61 mãos em render 3D sintético, com os dedos tingidos (`assets/hands_61.webp`,
/// gerado por research/data/hand_sprites.py): 11 × 6 ladrilhos de 160 px. Carregado
/// uma vez no início; desenhar um ladrilho é só copiar pixels.
abstract final class HandSprites {
  static const int columns = 11, tile = 160;
  static ui.Image? sheet;

  static Future<void> load() async {
    try {
      final data = await rootBundle.load('assets/hands_61.webp');
      final codec = await ui.instantiateImageCodec(data.buffer.asUint8List());
      sheet = (await codec.getNextFrame()).image;
    } catch (_) {
      sheet = null; // sem o asset (testes), o desenho vetorial assume
    }
  }
}

class _SpritePainter extends CustomPainter {
  _SpritePainter(this.code, this.sheet);

  final int code;
  final ui.Image sheet;

  @override
  void paint(Canvas canvas, Size size) {
    const t = HandSprites.tile;
    final source = Rect.fromLTWH(
      (code % HandSprites.columns) * t.toDouble(),
      (code ~/ HandSprites.columns) * t.toDouble(),
      t.toDouble(),
      t.toDouble(),
    );
    canvas.drawImageRect(
      sheet,
      source,
      Offset.zero & size,
      Paint()..filterQuality = FilterQuality.medium,
    );
  }

  @override
  bool shouldRepaint(_SpritePainter old) =>
      old.code != code || old.sheet != sheet;
}

/// A mão de uma CM numa caixa quadrada: o ladrilho do sprite sheet, ou o desenho
/// vetorial quando ele não foi carregado.
class CanonicalHand extends StatelessWidget {
  const CanonicalHand(this.code, {super.key, this.size = 72, this.opacity = 1});

  final int code;
  final double size;
  final double opacity;

  @override
  Widget build(BuildContext context) {
    final sheet = HandSprites.sheet;
    final hand = SizedBox.square(
      dimension: size,
      child: CustomPaint(
        painter: sheet == null
            ? CanonicalHandPainter(code)
            : _SpritePainter(code, sheet),
      ),
    );
    return opacity < 1 ? Opacity(opacity: opacity, child: hand) : hand;
  }
}

class _FingerLegend extends StatelessWidget {
  const _FingerLegend();

  @override
  Widget build(BuildContext context) => Wrap(
    spacing: 10,
    children: [
      for (var f = 0; f < 5; f++)
        Text(
          fingerNames[f],
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w700,
            color: fingerColors[f],
          ),
        ),
    ],
  );
}

/// Catálogo das 61 CMs: a mão canônica de cada uma, com a postura ao tocar.
Future<void> showCatalog(BuildContext context, {int? active}) =>
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (context) => DraggableScrollableSheet(
        expand: false,
        initialChildSize: 0.85,
        maxChildSize: 0.95,
        builder: (context, scroll) => _Catalog(scroll: scroll, active: active),
      ),
    );

class _Catalog extends StatelessWidget {
  const _Catalog({required this.scroll, this.active});

  final ScrollController scroll;
  final int? active;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return CustomScrollView(
      controller: scroll,
      slivers: [
        SliverPadding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
          sliver: SliverToBoxAdapter(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'As 61 configurações de mão',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 4),
                Text(
                  'Mão desenhada a partir dos pontos do atlas do capítulo. Toque numa CM para '
                  'ver a postura dos dedos; ⚠ marca as CMs mais confundidas.',
                  style: TextStyle(
                    fontSize: 12,
                    color: scheme.onSurfaceVariant,
                  ),
                ),
                const SizedBox(height: 6),
                const _FingerLegend(),
              ],
            ),
          ),
        ),
        SliverPadding(
          padding: const EdgeInsets.fromLTRB(12, 0, 12, 24),
          sliver: SliverGrid.count(
            crossAxisCount: 4,
            mainAxisSpacing: 6,
            crossAxisSpacing: 6,
            childAspectRatio: 0.82,
            children: [
              for (var code = 0; code < 61; code++)
                _CatalogTile(code: code, active: code == active),
            ],
          ),
        ),
      ],
    );
  }
}

class _CatalogTile extends StatelessWidget {
  const _CatalogTile({required this.code, required this.active});

  final int code;
  final bool active;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final confused = frequentConfusion.contains(code);
    return Material(
      color: active ? scheme.primaryContainer : scheme.surfaceContainerHighest,
      borderRadius: BorderRadius.circular(10),
      child: InkWell(
        borderRadius: BorderRadius.circular(10),
        onTap: () => _showDetail(context, code),
        child: Padding(
          padding: const EdgeInsets.all(4),
          child: Column(
            children: [
              Expanded(
                child: LayoutBuilder(
                  builder: (context, box) => CanonicalHand(
                    code,
                    size: math.min(box.maxWidth, box.maxHeight),
                  ),
                ),
              ),
              Text(
                '${confused ? '⚠ ' : ''}CM ${code + 1}',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  color: confused ? scheme.tertiary : scheme.onSurface,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

Future<void> _showDetail(BuildContext context, int code) => showDialog<void>(
  context: context,
  builder: (context) {
    final scheme = Theme.of(context).colorScheme;
    return AlertDialog(
      title: Text('CM ${code + 1}'),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          CanonicalHand(code, size: 200),
          const SizedBox(height: 8),
          const _FingerLegend(),
          const SizedBox(height: 8),
          Text(canonicalPostures[code]),
          if (frequentConfusion.contains(code)) ...[
            const SizedBox(height: 6),
            Text(
              'Confusão frequente com uma CM vizinha: confira o top-3.',
              style: TextStyle(fontSize: 12, color: scheme.tertiary),
            ),
          ],
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Fechar'),
        ),
      ],
    );
  },
);
