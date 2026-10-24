import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:signedge_mobile/src/logic/decision.dart';

Float32List _onehot(int code, {double high = 5}) {
  final out = Float32List(61);
  out[code] = high;
  return out;
}

Float32List _probs(int code, double p) {
  final out = Float32List(61)..fillRange(0, 61, (1 - p) / 60);
  out[code] = p;
  return out;
}

void main() {
  test(
    'estados: aceita qualquer das 61 acima do limiar, abaixo é transição',
    () {
      final strong = decide(softmax(_onehot(9, high: 8)), 0.45); // CM10
      expect(strong.state, DecisionState.accepted);
      expect(strong.code, 9);
      expect(strong.top.first.cm, 10);

      final weak = decide(softmax(_onehot(37, high: 1)), 0.45);
      expect(weak.state, DecisionState.belowThreshold);
      expect(weak.code, 37);
    },
  );

  test('top-3 em ordem decrescente, com as probabilidades', () {
    final x = Float32List(61)..fillRange(0, 61, 0.1 / 58);
    x[45] = 0.52; // CM46
    x[46] = 0.31; // CM47
    x[43] = 0.07; // CM44
    final d = decide(x, 0.45);
    expect(d.top.map((r) => r.cm).toList(), [46, 47, 44]);
    expect(d.top.first.probability, closeTo(0.52, 1e-6));
    expect(d.probability, closeTo(0.52, 1e-6));
  });

  test('CMs de confusão frequente: recall < 50% fora do fold', () {
    expect(frequentConfusion.map((c) => c + 1).toSet(), {
      25,
      27,
      32,
      38,
      42,
      47,
    });
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
    Decision step(Stabilizer s, Float32List x) =>
        s.step(decide(x, 0.45), x, 0.45);

    test('entra só com 2 frames seguidos acima do limiar', () {
      final s = Stabilizer();
      expect(step(s, _probs(37, 0.60)).state, DecisionState.belowThreshold);
      final d = step(s, _probs(37, 0.60));
      expect(d.state, DecisionState.accepted);
      expect(d.code, 37);
    });

    test('ativa, fica travada até limiar − 0,12; abaixo disso, sai', () {
      final s = Stabilizer();
      step(s, _probs(37, 0.6));
      step(s, _probs(37, 0.6));
      // respirou: abaixo de 0,45, acima de 0,33
      expect(step(s, _probs(37, 0.40)).code, 37);
      expect(step(s, _probs(37, 0.34)).state, DecisionState.accepted);
      expect(step(s, _probs(37, 0.30)).state, DecisionState.belowThreshold);
    });

    test('troca de CM também pede 2 frames da nova', () {
      final s = Stabilizer();
      step(s, _probs(37, 0.6));
      step(s, _probs(37, 0.6));
      // Um frame de outra CM forte não derruba a ativa (ela ainda está ≥ 0,33).
      final x = Float32List(61)..fillRange(0, 61, 0.1 / 59);
      x[0] = 0.5;
      x[37] = 0.4;
      expect(step(s, x).code, 37);
      final d = step(s, x);
      expect(d.code, 0);
      expect(d.probability, closeTo(0.5, 1e-6));
    });

    test('reset esquece a ativa', () {
      final s = Stabilizer();
      step(s, _probs(5, 0.9));
      step(s, _probs(5, 0.9));
      s.reset();
      expect(step(s, _probs(5, 0.9)).state, DecisionState.belowThreshold);
    });
  });
}
