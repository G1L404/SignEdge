/// Leitura aproximada das CMs (assets/handshapes.json): a descrição da forma e a
/// letra ou número da datilologia que ela lembra.
library;

import 'dart:convert';

import 'decision.dart';

class Handshape {
  const Handshape({
    required this.cm,
    required this.shape,
    required this.letter,
    required this.alt,
    required this.variants,
  });

  factory Handshape.fromJson(Map<String, dynamic> json) => Handshape(
    cm: json['cm'] as int,
    shape: json['shape'] as String,
    letter: json['letter'] as String?,
    alt: json['alt'] as String?,
    variants: [for (final v in json['variants'] as List? ?? []) v as int],
  );

  final int cm;
  final String shape;

  /// Letra ou número que a forma lembra, segundo `letters_source`.
  final String? letter;

  /// Leitura alternativa (ex.: CM 38 = L ou 2).
  final String? alt;

  /// CMs vizinhas (números CM) que herdam esta leitura.
  final List<int> variants;

  /// "L / 2", ou só a letra.
  String? get reading => letter == null
      ? null
      : alt == null
      ? letter
      : '$letter / $alt';
}

class HandshapeCatalog {
  HandshapeCatalog(Iterable<Handshape> shapes, {required this.reviewed})
    : _byCm = {for (final s in shapes) s.cm: s},
      _groupOf = {
        for (final s in shapes)
          for (final v in s.variants) v: s,
      };

  factory HandshapeCatalog.parse(String source) {
    final json = jsonDecode(source) as Map<String, dynamic>;
    return HandshapeCatalog([
      for (final s in json['shapes'] as List)
        Handshape.fromJson(s as Map<String, dynamic>),
    ], reviewed: json['review'] == 'revisado');
  }

  final Map<int, Handshape> _byCm;

  /// Variante → a entrada que lhe dá a leitura.
  final Map<int, Handshape> _groupOf;

  /// Se a leitura já passou por revisão de pessoa surda ou intérprete.
  final bool reviewed;

  /// Descrição da forma: a da própria CM, senão a do grupo dela.
  String? shapeOf(int cm) => (_byCm[cm] ?? _groupOf[cm])?.shape;

  /// Leitura da CM: a própria, senão a do grupo (CM 22 lembra o C da 26).
  String? readingOf(int cm) => _byCm[cm]?.reading ?? _groupOf[cm]?.reading;

  /// Leituras distintas das CMs com pelo menos [minimum] de probabilidade, na
  /// ordem de [ranked]: "C ou O" quando a rede divide a confiança entre formas
  /// de letras diferentes, em vez de afirmar uma só.
  ///
  /// A primeira de [ranked] é a CM do card: sem leitura própria, nada aparece
  /// (a letra de uma vizinha pareceria ser a dela).
  List<String> readingsAmong(List<Ranked> ranked, {double minimum = 0.15}) {
    if (ranked.isEmpty || readingOf(ranked.first.cm) == null) return const [];
    final out = <String>[];
    for (final r in ranked) {
      if (r.probability < minimum) continue;
      final reading = readingOf(r.cm);
      if (reading != null && !out.contains(reading)) out.add(reading);
    }
    return out;
  }
}
