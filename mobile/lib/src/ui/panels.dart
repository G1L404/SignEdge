import 'package:flutter/material.dart';

import '../app_controller.dart';
import '../logic/decision.dart';

/// Telemetria Edge AI: FPS, latências e o selo do modelo embarcado.
class TelemetryBar extends StatelessWidget {
  const TelemetryBar({super.key, required this.app});

  final AppController app;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    Widget metric(String label, String value) => Padding(
      padding: const EdgeInsets.only(right: 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            value,
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w700,
              fontFeatures: [FontFeature.tabularFigures()],
            ),
          ),
          Text(
            label,
            style: TextStyle(fontSize: 10, color: scheme.onSurfaceVariant),
          ),
        ],
      ),
    );
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 8, 4),
      child: Row(
        children: [
          metric('FPS', app.fps.toStringAsFixed(0)),
          metric(
            'mão ${app.handBackends} (${app.usedPalm ? 'palma' : 'rastreio'})',
            '${app.handMs.toStringAsFixed(0)} ms',
          ),
          metric(
            'SignEdge ${app.classifierBackend.toUpperCase()}',
            '${app.classifierMs.toStringAsFixed(1)} ms',
          ),
          const Spacer(),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: scheme.primaryContainer,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              'Int8 offline\n${(app.modelBytes / 1048576).toStringAsFixed(2).replaceAll('.', ',')} MiB',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 10,
                height: 1.2,
                color: scheme.onPrimaryContainer,
              ),
            ),
          ),
          IconButton(
            tooltip: 'Trocar câmera',
            onPressed: app.canSwitch ? app.switchCamera : null,
            icon: const Icon(Icons.cameraswitch_outlined),
          ),
        ],
      ),
    );
  }
}

/// Modo soletrador: a configuração parada por 0,75 s vira caractere.
class SpellerBar extends StatelessWidget {
  const SpellerBar({super.key, required this.app});

  final AppController app;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final text = app.speller.text;
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 0, 8, 6),
      child: Column(
        children: [
          Row(
            children: [
              Expanded(
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 10,
                  ),
                  decoration: BoxDecoration(
                    color: scheme.surfaceContainerHighest,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(
                    text.isEmpty
                        ? 'Segure uma configuração para digitar…'
                        : text,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      fontSize: 18,
                      letterSpacing: 1.5,
                      color: text.isEmpty
                          ? scheme.onSurfaceVariant
                          : scheme.onSurface,
                    ),
                  ),
                ),
              ),
              IconButton(
                tooltip: 'Espaço',
                onPressed: () {
                  app.speller.space();
                  app.notifySpeller();
                },
                icon: const Icon(Icons.space_bar),
              ),
              IconButton(
                tooltip: 'Apagar',
                onPressed: () {
                  app.speller.backspace();
                  app.notifySpeller();
                },
                icon: const Icon(Icons.backspace_outlined),
              ),
              IconButton(
                tooltip: 'Limpar',
                onPressed: () {
                  app.speller.clear();
                  app.notifySpeller();
                },
                icon: const Icon(Icons.clear_all),
              ),
            ],
          ),
          const SizedBox(height: 4),
          LinearProgressIndicator(
            value: app.spellProgress,
            minHeight: 4,
            borderRadius: BorderRadius.circular(2),
          ),
        ],
      ),
    );
  }
}

/// Card principal. Quatro estados, para a tela nunca contradizer as barras:
/// sem mão; em transição (a mais provável abaixo do limiar); CM fora da demo
/// (reconhecida com confiança, mas fora do soletrador); e a CM da demo aceita,
/// com a variante que a rede viu quando não é a canônica.
class ResultCard extends StatelessWidget {
  const ResultCard({super.key, required this.app});

  final AppController app;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final decision = app.decision;
    final accepted = decision.accepted;
    final shape = accepted == null ? null : app.catalog[accepted];
    final recognized = decision.recognized;
    // A descrição é da CM que a rede viu (a variante tem a sua, se houver); a
    // letra é a da canônica, que é o que o soletrador digita.
    final described =
        (recognized == null ? null : app.catalog[recognized]) ?? shape;
    final percent = '${(decision.probability * 100).round()}%';
    final limit = '${(app.threshold * 100).round()}%';

    final (
      String title,
      Color color,
      String? subtitle,
    ) = switch (decision.state) {
      DecisionState.noHand => ('Mostre a mão', scheme.onSurfaceVariant, null),
      DecisionState.belowThreshold => (
        'Em transição',
        scheme.onSurfaceVariant,
        'Nenhuma CM passa do limiar ($percent < $limit)',
      ),
      DecisionState.outsideDemo => (
        'CM ${recognized! + 1}',
        scheme.tertiary,
        'Configuração reconhecida ($percent), fora do vocabulário do soletrador',
      ),
      DecisionState.accepted => (
        'CM ${accepted! + 1}',
        scheme.primary,
        recognized != null && recognized != accepted
            ? 'Reconhecida como CM ${recognized + 1}, variante desta ($percent)'
            : null,
      ),
    };
    final large =
        decision.state == DecisionState.accepted ||
        decision.state == DecisionState.outsideDemo;

    return Card(
      margin: const EdgeInsets.fromLTRB(12, 8, 12, 12),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 14, 16, 8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    fontSize: large ? 40 : 24,
                    fontWeight: FontWeight.w800,
                    color: color,
                  ),
                ),
                const SizedBox(width: 12),
                if (shape?.letter != null)
                  Chip(
                    label: Text(
                      shape!.alt == null
                          ? shape.letter!
                          : '${shape.letter} · ${shape.alt}',
                      style: const TextStyle(fontWeight: FontWeight.w700),
                    ),
                    visualDensity: VisualDensity.compact,
                  ),
                if (decision.state == DecisionState.outsideDemo)
                  Chip(
                    label: const Text('fora da demo'),
                    visualDensity: VisualDensity.compact,
                    side: BorderSide(color: scheme.tertiary),
                  ),
              ],
            ),
            if (subtitle != null)
              Text(
                subtitle,
                style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant),
              ),
            // Confirmação da letra no próprio card: enche em 0,75 s e digita.
            if (decision.state == DecisionState.accepted &&
                shape?.letter != null &&
                app.spellProgress > 0) ...[
              const SizedBox(height: 6),
              Row(
                children: [
                  Text(
                    'Digitando ${shape!.letter}',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                      color: scheme.primary,
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: LinearProgressIndicator(
                      value: app.spellProgress,
                      minHeight: 8,
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 4),
            ],
            if (shape != null) ...[
              Text(described!.shape, style: TextStyle(color: scheme.onSurface)),
              const SizedBox(height: 4),
              Text(
                shape.examples.isNotEmpty && app.catalog.reviewed
                    ? 'Sinais com esta CM: ${shape.examples.join(' · ')}'
                    : 'Exemplos de sinais: aguardando revisão (INES / intérprete)',
                style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant),
              ),
            ],
            const SizedBox(height: 10),
            for (final r in decision.top)
              _ProbabilityBar(
                ranked: r,
                threshold: app.threshold,
                spellable: app.spellableCodes.contains(r.code),
              ),
            Row(
              children: [
                Text(
                  'Limiar ${(app.threshold * 100).round()}%',
                  style: TextStyle(
                    fontSize: 12,
                    color: scheme.onSurfaceVariant,
                  ),
                ),
                Expanded(
                  child: Slider(
                    value: app.threshold,
                    min: 0.3,
                    max: 0.9,
                    divisions: 12,
                    label: '${(app.threshold * 100).round()}%',
                    onChanged: app.setThreshold,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _ProbabilityBar extends StatelessWidget {
  const _ProbabilityBar({
    required this.ranked,
    required this.threshold,
    required this.spellable,
  });

  final Ranked ranked;
  final double threshold;

  /// CM da demo ou variante dela (vai para o soletrador).
  final bool spellable;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final strong = ranked.probability >= threshold;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          SizedBox(
            width: 64,
            child: Text(
              'CM ${ranked.cm}',
              style: TextStyle(
                fontWeight: FontWeight.w600,
                color: spellable ? scheme.onSurface : scheme.onSurfaceVariant,
              ),
            ),
          ),
          Expanded(
            child: LinearProgressIndicator(
              value: ranked.probability,
              minHeight: 10,
              borderRadius: BorderRadius.circular(5),
              // Acima do limiar: verde se vai para o soletrador, outra cor se não.
              color: strong
                  ? (spellable ? scheme.primary : scheme.tertiary)
                  : scheme.outline,
              backgroundColor: scheme.surfaceContainerHighest,
            ),
          ),
          SizedBox(
            width: 48,
            child: Text(
              '${(ranked.probability * 100).round()}%',
              textAlign: TextAlign.right,
            ),
          ),
        ],
      ),
    );
  }
}
