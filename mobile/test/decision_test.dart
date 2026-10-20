import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:signedge_mobile/src/logic/decision.dart';
import 'package:signedge_mobile/src/logic/handshapes.dart';

Float32List _onehot(int code, {double high = 5}) {
  final out = Float32List(61);
  out[code] = high;
  return out;
}

void main() {
  test('vocabulário da demo é o do YAML (14 CMs)', () {
    expect(demoCodes.map((c) => c + 1).toList(), [
      1,
      2,
      3,
      5,
      8,
      26,
      33,
      38,
      39,
      40,
      44,
      51,
      53,
      61,
    ]);
  });

  test('média móvel guarda só a janela', () {
    final smoother = LogitSmoother(window: 3);
    for (var i = 0; i < 5; i++) {
      smoother.add(_onehot(0, high: i.toDouble()));
    }
    expect(smoother.length, 3);
    expect(smoother.mean()![0], closeTo((2 + 3 + 4) / 3, 1e-6));
  });

  test('estados: aceita da demo, fora da demo e abaixo do limiar', () {
    final inside = decide(softmax(_onehot(37, high: 8)), 0.6); // CM38
    expect(inside.state, DecisionState.accepted);
    expect(inside.accepted, 37);
    expect(inside.top.first.cm, 38);

    // CM10 com confiança alta: reconhecida, mas fora do soletrador (não é "transição").
    final outside = decide(softmax(_onehot(9, high: 8)), 0.6);
    expect(outside.state, DecisionState.outsideDemo);
    expect(outside.accepted, isNull);
    expect(outside.recognized, 9);

    final weak = decide(softmax(_onehot(37, high: 1)), 0.6);
    expect(weak.state, DecisionState.belowThreshold);
    expect(weak.accepted, isNull);
  });

  test(
    'variantes somam no grupo: OK de dedos abertos (CM47) vira F (CM44)',
    () {
      const variants = {
        43: [44, 45, 46],
      };
      // O caso real do aparelho: CM47 93%, CM46 2%.
      final probs = Float32List(61)..fillRange(0, 61, 0.05 / 59);
      probs[46] = 0.93;
      probs[45] = 0.02;
      final ok = decide(probs, 0.45, variants: variants);
      expect(ok.state, DecisionState.accepted);
      expect(ok.accepted, 43);
      expect(ok.recognized, 46);
      expect(ok.probability, closeTo(0.95 + 2 * 0.05 / 59, 1e-6));

      // Espalhada entre as variantes, nenhuma passa sozinha; o grupo passa.
      final spread = Float32List(61)..fillRange(0, 61, 0.2 / 57);
      for (final c in [43, 44, 45, 46]) {
        spread[c] = 0.2;
      }
      expect(decide(spread, 0.45).state, DecisionState.belowThreshold);
      expect(decide(spread, 0.45, variants: variants).accepted, 43);
    },
  );

  test('soletrador digita depois de 1,2 s parado e não repete', () {
    final speller = Speller(hold: const Duration(milliseconds: 1200));
    String token(int code) => code == 37 ? 'L' : '[CM${code + 1}]';
    Duration ms(int v) => Duration(milliseconds: v);
    expect(speller.update(37, ms(0), token), isNull);
    expect(speller.progress(ms(600)), closeTo(0.5, 1e-9));
    expect(speller.update(37, ms(1100), token), isNull);
    expect(speller.update(37, ms(1250), token), 'L');
    expect(speller.update(37, ms(3000), token), isNull);
    speller.update(null, ms(3100), token);
    speller.update(4, ms(3200), token);
    expect(speller.update(4, ms(4500), token), '[CM5]');
    expect(speller.text, 'L[CM5]');
    speller.backspace();
    expect(speller.text, 'L');
  });

  test('EMA nos logits: primeiro frame é ele mesmo, depois 0,35 do novo', () {
    final ema = LogitEma(alpha: 0.35);
    ema.add(_onehot(0, high: 10));
    expect(ema.value![0], closeTo(10, 1e-6));
    ema.add(_onehot(0, high: 0));
    expect(ema.value![0], closeTo(6.5, 1e-6));
    ema.clear();
    expect(ema.value, isNull);
  });

  group('histerese', () {
    Float32List probs(int code, double p) {
      final out = Float32List(61)..fillRange(0, 61, (1 - p) / 60);
      out[code] = p;
      return out;
    }

    Decision step(Stabilizer s, int code, double p) {
      final x = probs(code, p);
      return s.step(decide(x, 0.45), x, 0.45);
    }

    test('entra só com 2 frames seguidos acima do limiar', () {
      final s = Stabilizer();
      expect(step(s, 37, 0.60).state, DecisionState.belowThreshold);
      expect(step(s, 37, 0.60).accepted, 37);
    });

    test('ativa, fica travada até limiar − 0,12; abaixo disso, sai', () {
      final s = Stabilizer();
      step(s, 37, 0.6);
      step(s, 37, 0.6);
      expect(
        step(s, 37, 0.40).accepted,
        37,
      ); // respirou: abaixo de 0,45, acima de 0,33
      expect(step(s, 37, 0.34).accepted, 37);
      expect(step(s, 37, 0.30).state, DecisionState.belowThreshold);
    });

    test('troca de CM também pede 2 frames da nova', () {
      final s = Stabilizer();
      step(s, 37, 0.6);
      step(s, 37, 0.6);
      // Um frame de outra CM forte não derruba a ativa (ela ainda está ≥ 0,33).
      final x = Float32List(61)..fillRange(0, 61, 0.1 / 59);
      x[0] = 0.5;
      x[37] = 0.4;
      expect(s.step(decide(x, 0.45), x, 0.45).accepted, 37);
      expect(s.step(decide(x, 0.45), x, 0.45).accepted, 0);
    });
  });

  test('soletrador padrão digita em 0,75 s', () {
    final speller = Speller();
    Duration ms(int v) => Duration(milliseconds: v);
    speller.update(0, ms(0), (_) => 'A');
    expect(speller.update(0, ms(700), (_) => 'A'), isNull);
    expect(speller.update(0, ms(760), (_) => 'A'), 'A');
  });

  test('catálogo: caracteres do soletrador da Tabela V', () {
    final catalog = HandshapeCatalog.parse(
      File('assets/handshapes.json').readAsStringSync(),
    );
    // As 14 da demo, mais entradas de variantes com descrição própria.
    expect(catalog.byCode.keys.toSet().containsAll(demoCodes), isTrue);
    expect(catalog[0]!.letter, 'A');
    expect(catalog[37]!.letter, 'L');
    expect(catalog[2]!.letter, 'Y');
    expect(catalog.token(1), '1');
    expect(catalog[37]!.alt, '2');
    expect(catalog.token(38), '🤘');
    expect(catalog.lettersSource, contains('Porfirio'));
    expect(catalog.token(9), '[CM10]'); // fora da demo
    expect(catalog.reviewed, isFalse);
    expect(catalog.variantCodes, {
      13: [15], // D = 14 + 16
      20: [19], // O = 21 + 20
      25: [21], // C = 26 + 22
      37: [14], // L = 38 + 15
      43: [44, 45, 46], // F = 44 + 45/46/47
      48: [49], // V = 49 + 50
      52: [55], // B = 53 + 56
    });
    // 17 canônicas: as 14 da demo científica + O, V, D; 46 e 47 são só descrição.
    expect(catalog.canonicalCodes, {...demoCodes, 20, 48, 13});
    expect(catalog.canonicalCodes.contains(46), isFalse);
    expect(catalog.token(20), 'O');
    expect(catalog[20]!.alt, '0');
  });
}
