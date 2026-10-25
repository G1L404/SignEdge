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
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w700,
              fontFeatures: [FontFeature.tabularFigures()],
            ),
          ),
          Text(
            label,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
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
          // O rótulo da mão é o mais longo ("GPU · GPU (rastreio)"): ocupa a sobra
          // e corta com reticências em telas de ~380 dp, em vez de estourar a linha.
          Expanded(
            child: metric(
              'mão ${app.handBackends} (${app.usedPalm ? 'palma' : 'rastreio'})',
              '${app.handMs.toStringAsFixed(0)} ms',
            ),
          ),
          metric(
            'SignEdge ${app.classifierBackend.toUpperCase()}',
            '${app.classifierMs.toStringAsFixed(1)} ms',
          ),
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

/// Card principal. Três estados, para a tela nunca contradizer as barras: sem
/// mão; em transição (nenhuma CM ativa); e a CM ativa, com o top-3 das 61.
class ResultCard extends StatelessWidget {
  const ResultCard({super.key, required this.app});

  final AppController app;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final decision = app.decision;
    final code = decision.code;
    final percent = '${(decision.probability * 100).round()}%';
    final limit = '${(app.threshold * 100).round()}%';
    final accepted = decision.state == DecisionState.accepted;
    // Leitura da CM ativa primeiro, e a das outras do top-3 que dividem a
    // confiança com ela ("C ou O").
    final readings = accepted
        ? app.catalog.readingsAmong([
            Ranked(code!, decision.probability),
            ...decision.top.where((r) => r.code != code),
          ])
        : const <String>[];
    final shape = accepted ? app.catalog.shapeOf(code! + 1) : null;

    final (
      String title,
      Color color,
      String? subtitle,
    ) = switch (decision.state) {
      DecisionState.noHand => ('Mostre a mão', scheme.onSurfaceVariant, null),
      DecisionState.belowThreshold => (
        'Em transição',
        scheme.onSurfaceVariant,
        'Nenhuma CM ativa ($percent; limiar $limit)',
      ),
      DecisionState.accepted => ('CM ${code! + 1}', scheme.primary, percent),
    };

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
                    fontSize: accepted ? 40 : 24,
                    fontWeight: FontWeight.w800,
                    color: color,
                  ),
                ),
                const SizedBox(width: 12),
                if (readings.isNotEmpty)
                  Flexible(
                    child: Chip(
                      label: Text(
                        'lembra ${readings.join(' ou ')}',
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      visualDensity: VisualDensity.compact,
                    ),
                  ),
                if (accepted && frequentConfusion.contains(code))
                  Flexible(
                    child: Chip(
                      label: const Text(
                        'confusão frequente',
                        overflow: TextOverflow.ellipsis,
                      ),
                      visualDensity: VisualDensity.compact,
                      side: BorderSide(color: scheme.tertiary),
                    ),
                  ),
              ],
            ),
            if (subtitle != null)
              Text(
                subtitle,
                style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant),
              ),
            if (shape != null)
              Text(shape, style: TextStyle(color: scheme.onSurface)),
            if (readings.isNotEmpty && !app.catalog.reviewed)
              Text(
                'Forma da mão, não a letra: leitura aproximada, aguardando '
                'revisão (INES / intérprete)',
                style: TextStyle(fontSize: 11, color: scheme.onSurfaceVariant),
              ),
            if (decision.top.isNotEmpty) ...[
              const SizedBox(height: 6),
              Text(
                decision.top
                    .map(
                      (r) => 'CM ${r.cm} (${(r.probability * 100).round()}%)',
                    )
                    .join(' · '),
                style: TextStyle(fontSize: 13, color: scheme.onSurface),
              ),
            ],
            const SizedBox(height: 8),
            for (final r in decision.top)
              _ProbabilityBar(
                ranked: r,
                threshold: app.threshold,
                active: accepted && r.code == code,
              ),
            Row(
              children: [
                Text(
                  'Limiar $limit',
                  style: TextStyle(
                    fontSize: 12,
                    color: scheme.onSurfaceVariant,
                  ),
                ),
                Expanded(
                  child: Slider(
                    value: app.threshold,
                    min: AppController.minThreshold,
                    max: AppController.maxThreshold,
                    divisions: 12,
                    label: limit,
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
    required this.active,
  });

  final Ranked ranked;
  final double threshold;

  /// A CM ativa no card.
  final bool active;

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
                color: active ? scheme.onSurface : scheme.onSurfaceVariant,
              ),
            ),
          ),
          Expanded(
            child: LinearProgressIndicator(
              value: ranked.probability,
              minHeight: 10,
              borderRadius: BorderRadius.circular(5),
              color: active
                  ? scheme.primary
                  : (strong ? scheme.tertiary : scheme.outline),
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
