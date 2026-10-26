import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:signedge_mobile/src/logic/canonical_hands.dart';
import 'package:signedge_mobile/src/ui/canonical_hand.dart';

void main() {
  test(
    '61 mãos de 21 pontos (x, y, z), com x e y normalizados por centroide e raio RMS',
    () {
      expect(canonicalHands.length, 61);
      expect(canonicalPostures.length, 61);
      for (final hand in canonicalHands) {
        expect(hand.length, 63);
        var cx = 0.0, cy = 0.0;
        for (var i = 0; i < 21; i++) {
          cx += hand[3 * i] / 21;
          cy += hand[3 * i + 1] / 21;
        }
        var r2 = 0.0;
        for (var i = 0; i < 21; i++) {
          r2 +=
              (math.pow(hand[3 * i] - cx, 2) +
                  math.pow(hand[3 * i + 1] - cy, 2)) /
              21;
        }
        expect(cx.abs(), lessThan(0.01));
        expect(cy.abs(), lessThan(0.01));
        expect(math.sqrt(r2), closeTo(1, 0.01));
      }
    },
  );

  testWidgets('catálogo abre com as 61 CMs e mostra a postura ao tocar', (
    tester,
  ) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Builder(
          builder: (context) => Scaffold(
            body: TextButton(
              onPressed: () => showCatalog(context, active: 37),
              child: const Text('abrir'),
            ),
          ),
        ),
      ),
    );
    await tester.tap(find.text('abrir'));
    await tester.pumpAndSettle();
    expect(find.text('As 61 configurações de mão'), findsOneWidget);
    expect(find.text('CM 1'), findsOneWidget);
    await tester.tap(find.text('CM 1'));
    await tester.pumpAndSettle();
    expect(find.text(canonicalPostures[0]), findsOneWidget);
  });
}
