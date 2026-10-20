/// Da sequência de logits à resposta da tela: média móvel, softmax, vocabulário
/// da demo com limiar, e o soletrador.
///
/// Os logits do model_int8.tflite já vêm divididos por T (calibração fora do
/// fold do LOSO), então o softmax aqui é a probabilidade calibrada.
library;

import 'dart:math' as math;
import 'dart:typed_data';

/// Códigos da base (CM − 1) do vocabulário científico da demo, `vocabularies.demo`
/// do YAML: CM 1, 2, 3, 5, 8, 26, 33, 38, 39, 40, 44, 51, 53, 61. É a ele que as
/// métricas do C6 se referem. O soletrador do app usa o de assets/handshapes.json
/// (estas 14 + O, V, D, com variantes), passado em `decide(vocabulary: ...)`.
const List<int> demoCodes = [0, 1, 2, 4, 7, 25, 32, 37, 38, 39, 43, 50, 52, 60];

/// Média dos logits dos últimos [window] frames com mão (a unidade "pausa").
class LogitSmoother {
  LogitSmoother({this.window = 6, this.classes = 61});

  final int window;
  final int classes;
  final List<Float32List> _frames = [];

  int get length => _frames.length;

  void add(Float32List logits) {
    _frames.add(Float32List.fromList(logits));
    if (_frames.length > window) _frames.removeAt(0);
  }

  void clear() => _frames.clear();

  Float32List? mean() {
    if (_frames.isEmpty) return null;
    final out = Float32List(classes);
    for (final f in _frames) {
      for (var i = 0; i < classes; i++) {
        out[i] += f[i];
      }
    }
    for (var i = 0; i < classes; i++) {
      out[i] /= _frames.length;
    }
    return out;
  }
}

/// EMA contínua dos logits dos frames com mão: o frame novo pesa [alpha]. Responde
/// já no primeiro frame e não tem o degrau da janela fixa (um frame saindo dela
/// mudava a média de uma vez). Limpa quando a mão some.
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
  /// Sem mão (ou janela de frames ainda curta).
  noHand,

  /// A mais provável não chega ao limiar: a mão está mudando de forma.
  belowThreshold,

  /// Passa do limiar, mas é uma CM fora do vocabulário do soletrador.
  outsideDemo,

  /// CM da demo (ou variante dela) acima do limiar: vai para o soletrador.
  accepted,
}

/// O que a tela mostra num instante.
class Decision {
  const Decision({
    required this.state,
    required this.top,
    this.accepted,
    this.recognized,
    this.group,
    this.probability = 0,
  });

  final DecisionState state;

  /// Código canônico aceito (o da demo), só no estado [DecisionState.accepted].
  final int? accepted;

  /// A CM que a rede de fato reconheceu (pode ser uma variante, ou uma de fora).
  final int? recognized;

  /// Grupo vencedor: o código canônico, ou a própria CM quando ela não tem grupo.
  final int? group;

  /// Probabilidade da vencedora (a do grupo, quando há variantes).
  final double probability;

  /// As três mais prováveis entre as 61.
  final List<Ranked> top;

  static const neutral = Decision(state: DecisionState.noHand, top: []);
}

/// Argmax nas 61 (o modelo não é forçado para dentro do vocabulário), com as
/// variantes de cada CM da demo somadas num grupo.
///
/// [variants] é {código canônico: [códigos das variantes]} (assets/handshapes.json).
/// Somar ajuda porque a rede espalha a confiança entre formas vizinhas (CM46 e
/// CM47 são o par mais confundido da base): nas pausas fora do fold do LOSO, os
/// grupos F, L e B levam o recall nas da demo de 79,4% a 87,4% com o falso aceite
/// em ~3%, no limiar 0,45. [vocabulary] são os códigos canônicos que o soletrador
/// aceita.
Decision decide(
  Float32List probabilities,
  double threshold, {
  Map<int, List<int>> variants = const {},
  Iterable<int> vocabulary = demoCodes,
}) {
  final order = List<int>.generate(probabilities.length, (i) => i)
    ..sort((a, b) => probabilities[b].compareTo(probabilities[a]));
  final top = [for (final i in order.take(3)) Ranked(i, probabilities[i])];

  final groupOf = <int, int>{
    for (final e in variants.entries)
      for (final c in e.value) c: e.key,
  };
  final grouped = <int, double>{};
  for (var c = 0; c < probabilities.length; c++) {
    final key = groupOf[c] ?? c;
    grouped[key] = (grouped[key] ?? 0) + probabilities[c];
  }
  var best = grouped.keys.first;
  for (final e in grouped.entries) {
    if (e.value > grouped[best]!) best = e.key;
  }
  final probability = grouped[best]!;
  // Dentro do grupo vencedor, a CM que a rede reconheceu de fato.
  final members = [best, ...?variants[best]];
  final recognized = members.reduce(
    (a, b) => probabilities[a] >= probabilities[b] ? a : b,
  );
  if (probability < threshold) {
    return Decision(
      state: DecisionState.belowThreshold,
      top: top,
      recognized: recognized,
      group: best,
      probability: probability,
    );
  }
  final inDemo = vocabulary.contains(best);
  return Decision(
    state: inDemo ? DecisionState.accepted : DecisionState.outsideDemo,
    top: top,
    accepted: inDemo ? best : null,
    recognized: recognized,
    group: best,
    probability: probability,
  );
}

/// Probabilidade do grupo [code] (ele e as variantes dele).
double groupProbability(
  Float32List probabilities,
  int code,
  Map<int, List<int>> variants,
) {
  var p = probabilities[code];
  for (final v in variants[code] ?? const <int>[]) {
    p += probabilities[v];
  }
  return p;
}

/// Histerese (Schmitt trigger) sobre a decisão crua de cada frame.
///
/// Entrar (vindo de "em transição" ou trocando de CM) exige a mesma top-1, acima
/// do limiar, por [enterFrames] frames seguidos. Uma vez ativa, a CM fica no card
/// enquanto o próprio grupo estiver ≥ limiar − [release]: a mão respirando não faz
/// o card piscar.
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

  Decision step(
    Decision raw,
    Float32List probabilities,
    double threshold, {
    Map<int, List<int>> variants = const {},
    Iterable<int> vocabulary = demoCodes,
  }) {
    // Candidata a entrar: top-1 acima do limiar, contada em frames seguidos.
    final strong =
        raw.state == DecisionState.accepted ||
        raw.state == DecisionState.outsideDemo;
    if (strong && raw.group != _active) {
      _streak = raw.group == _candidate ? _streak + 1 : 1;
      _candidate = raw.group;
      if (_streak >= enterFrames) {
        _active = raw.group;
        _candidate = null;
        _streak = 0;
      }
    } else if (!strong || raw.group == _active) {
      _candidate = null;
      _streak = 0;
    }

    final active = _active;
    if (active != null) {
      final p = groupProbability(probabilities, active, variants);
      if (p < threshold - release) {
        _active = null;
      } else {
        final members = [active, ...?variants[active]];
        final recognized = members.reduce(
          (a, b) => probabilities[a] >= probabilities[b] ? a : b,
        );
        final inDemo = vocabulary.contains(active);
        return Decision(
          state: inDemo ? DecisionState.accepted : DecisionState.outsideDemo,
          top: raw.top,
          accepted: inDemo ? active : null,
          recognized: recognized,
          group: active,
          probability: p,
        );
      }
    }
    // Nada ativo: "em transição", mesmo que a crua já passe do limiar (ainda sem
    // os frames seguidos).
    return Decision(
      state: raw.state == DecisionState.noHand
          ? DecisionState.noHand
          : DecisionState.belowThreshold,
      top: raw.top,
      recognized: raw.recognized,
      group: raw.group,
      probability: raw.probability,
    );
  }
}

/// Soletrador: a mesma configuração aceita por [hold] (0,75 s) vira um caractere.
///
/// Depois de digitar, só digita de novo quando a mão muda (ou passa por
/// "Transição"), para não repetir a letra enquanto ela fica parada.
class Speller {
  /// 0,75 s parado digita: 1,2 s ficava lento na demo.
  Speller({this.hold = const Duration(milliseconds: 750)});

  final Duration hold;
  final List<String> _tokens = [];
  int? _current;
  Duration? _since;
  bool _typed = false;

  String get text => _tokens.join();

  /// Progresso de 0 a 1 da configuração atual até virar caractere.
  double progress(Duration now) {
    if (_current == null || _since == null || _typed) return 0;
    return ((now - _since!).inMicroseconds / hold.inMicroseconds).clamp(
      0.0,
      1.0,
    );
  }

  /// Atualiza com a decisão do instante [now]; devolve o token digitado, se houver.
  String? update(int? accepted, Duration now, String Function(int code) token) {
    if (accepted != _current) {
      _current = accepted;
      _since = now;
      _typed = false;
      return null;
    }
    if (accepted == null || _typed || now - _since! < hold) return null;
    _typed = true;
    final value = token(accepted);
    _tokens.add(value);
    return value;
  }

  /// Apaga o último token (uma letra, ou um "[CM n]" inteiro).
  void backspace() {
    if (_tokens.isNotEmpty) _tokens.removeLast();
  }

  void clear() => _tokens.clear();

  void space() => _tokens.add(' ');
}
