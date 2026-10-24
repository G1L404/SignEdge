/// Da sequência de logits à resposta da tela: média móvel, softmax, limiar e
/// histerese sobre as 61 configurações de mão.
///
/// Os logits do model_int8.tflite já vêm divididos por T (T = 0,48, ajustada
/// depois da poda nas pausas fora do fold do LOSO), então o softmax aqui é a
/// probabilidade calibrada.
library;

import 'dart:math' as math;
import 'dart:typed_data';

/// Códigos (CM − 1) com recall abaixo de 50% nas 1.160 pausas fora do fold do
/// modelo podado (T = 0,48): CM 25, 27, 32, 38, 42 e 47. O card avisa que a
/// resposta costuma ser trocada com vizinhas.
const Set<int> frequentConfusion = {24, 26, 31, 37, 41, 46};

/// EMA contínua dos logits dos frames com mão: o frame novo pesa [alpha]. Responde
/// já no primeiro frame e não tem o degrau de uma janela fixa. Limpa quando a mão
/// some.
class LogitEma {
  LogitEma({this.alpha = 0.35});

  final double alpha;
  Float32List? _value;
  int _frames = 0;

  int get length => _frames;

  Float32List? get value => _value;

  void add(Float32List logits) {
    final v = _value;
    if (v == null) {
      _value = Float32List.fromList(logits);
    } else {
      for (var i = 0; i < v.length; i++) {
        v[i] += alpha * (logits[i] - v[i]);
      }
    }
    _frames++;
  }

  void clear() {
    _value = null;
    _frames = 0;
  }
}

Float32List softmax(Float32List logits) {
  final top = logits.reduce(math.max);
  final out = Float32List(logits.length);
  var sum = 0.0;
  for (var i = 0; i < logits.length; i++) {
    out[i] = math.exp(logits[i] - top);
    sum += out[i];
  }
  for (var i = 0; i < out.length; i++) {
    out[i] /= sum;
  }
  return out;
}

class Ranked {
  const Ranked(this.code, this.probability);
  final int code;
  final double probability;
  int get cm => code + 1;
}

/// O que o card mostra.
enum DecisionState {
  /// Sem mão (ou nenhum logit ainda).
  noHand,

  /// Nenhuma CM ativa: a mais provável não chega ao limiar, ou ainda não ficou
  /// os frames seguidos que a histerese pede.
  belowThreshold,

  /// Uma CM ativa, acima do limiar.
  accepted,
}

/// O que a tela mostra num instante.
class Decision {
  const Decision({
    required this.state,
    required this.top,
    this.code,
    this.probability = 0,
  });

  final DecisionState state;

  /// CM ativa (código CM − 1) no estado [DecisionState.accepted]; nos outros, a
  /// mais provável, para o subtítulo.
  final int? code;

  /// Probabilidade de [code].
  final double probability;

  /// As três mais prováveis entre as 61, em ordem.
  final List<Ranked> top;

  static const neutral = Decision(state: DecisionState.noHand, top: []);
}

/// As [k] mais prováveis entre as 61, em ordem decrescente.
List<Ranked> topK(Float32List probabilities, [int k = 3]) {
  final order = List<int>.generate(probabilities.length, (i) => i)
    ..sort((a, b) => probabilities[b].compareTo(probabilities[a]));
  return [for (final i in order.take(k)) Ranked(i, probabilities[i])];
}

/// Decisão crua do frame: argmax nas 61, aceita se passa do [threshold].
///
/// Nas pausas fora do fold do modelo podado, o limiar 0,45 responde em 86,3% das
/// pausas, com 80,2% de acerto entre as respondidas; a verdadeira está entre as
/// três mostradas em 92,4% das pausas.
Decision decide(Float32List probabilities, double threshold) {
  final top = topK(probabilities);
  final best = top.first;
  return Decision(
    state: best.probability >= threshold
        ? DecisionState.accepted
        : DecisionState.belowThreshold,
    top: top,
    code: best.code,
    probability: best.probability,
  );
}

/// Histerese (Schmitt trigger) sobre a decisão crua de cada frame.
///
/// Entrar (vindo de "em transição" ou trocando de CM) exige a mesma top-1, acima
/// do limiar, por [enterFrames] frames seguidos. Uma vez ativa, a CM fica no card
/// enquanto a probabilidade dela estiver ≥ limiar − [release]: a mão respirando
/// não faz o card piscar.
class Stabilizer {
  Stabilizer({this.enterFrames = 2, this.release = 0.12});

  final int enterFrames;
  final double release;
  int? _active;
  int? _candidate;
  int _streak = 0;

  void reset() {
    _active = null;
    _candidate = null;
    _streak = 0;
  }

  Decision step(Decision raw, Float32List probabilities, double threshold) {
    final strong = raw.state == DecisionState.accepted;
    if (strong && raw.code != _active) {
      _streak = raw.code == _candidate ? _streak + 1 : 1;
      _candidate = raw.code;
      if (_streak >= enterFrames) {
        _active = raw.code;
        _candidate = null;
        _streak = 0;
      }
    } else if (!strong || raw.code == _active) {
      _candidate = null;
      _streak = 0;
    }

    final active = _active;
    if (active != null && probabilities[active] < threshold - release) {
      _active = null;
    }
    if (_active != null) {
      return Decision(
        state: DecisionState.accepted,
        top: raw.top,
        code: _active,
        probability: probabilities[_active!],
      );
    }
    // Nada ativo: "em transição", mesmo que a crua já passe do limiar (ainda sem
    // os frames seguidos).
    return Decision(
      state: raw.state == DecisionState.noHand
          ? DecisionState.noHand
          : DecisionState.belowThreshold,
      top: raw.top,
      code: raw.code,
      probability: raw.probability,
    );
  }
}
