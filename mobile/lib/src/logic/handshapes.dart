/// Metadados das CMs da demo (assets/handshapes.json).
library;

import 'dart:convert';

class Handshape {
  const Handshape({
    required this.cm,
    required this.shape,
    required this.letter,
    required this.alt,
    required this.variants,
    required this.examples,
  });

  factory Handshape.fromJson(Map<String, dynamic> json) => Handshape(
    cm: json['cm'] as int,
    shape: json['shape'] as String,
    letter: json['letter'] as String?,
    alt: json['alt'] as String?,
    variants: [for (final v in json['variants'] as List? ?? []) v as int],
    examples: [for (final e in json['examples'] as List) e as String],
  );

  final int cm;
  final String shape;

  /// Caractere do soletrador (datilologia ou dígito), segundo `letters_source`.
  final String? letter;

  /// Leitura alternativa (ex.: CM38 = L ou 2), mostrada no card.
  final String? alt;

  /// CMs vizinhas (números CM) que contam como esta no soletrador.
  final List<int> variants;

  /// Sinais que usam a CM; vazio enquanto a revisão linguística não sai.
  final List<String> examples;

  int get code => cm - 1;

  /// O que o soletrador digita.
  String get token => letter ?? '[CM$cm]';
}

class HandshapeCatalog {
  HandshapeCatalog(this.byCode, {required this.reviewed, this.lettersSource});

  factory HandshapeCatalog.parse(String source) {
    final json = jsonDecode(source) as Map<String, dynamic>;
    final shapes = [
      for (final s in json['shapes'] as List)
        Handshape.fromJson(s as Map<String, dynamic>),
    ];
    return HandshapeCatalog(
      {for (final s in shapes) s.code: s},
      reviewed: json['review'] == 'revisado',
      lettersSource: json['letters_source'] as String?,
    );
  }

  final Map<int, Handshape> byCode;

  /// Se os exemplos já passaram por revisão de pessoa surda ou intérprete.
  final bool reviewed;

  /// De onde vêm os caracteres do soletrador.
  final String? lettersSource;

  Handshape? operator [](int code) => byCode[code];

  /// Códigos das canônicas: entradas que não são variante de nenhuma outra (as que
  /// são, como 46 e 47, só emprestam a descrição própria ao card).
  Set<int> get canonicalCodes {
    final variants = {for (final s in byCode.values) ...s.variants};
    return {
      for (final s in byCode.values)
        if (!variants.contains(s.cm)) s.code,
    };
  }

  /// {código canônico: [códigos das variantes]}, para a decisão somar os grupos.
  Map<int, List<int>> get variantCodes => {
    for (final s in byCode.values)
      if (s.variants.isNotEmpty) s.code: [for (final cm in s.variants) cm - 1],
  };

  String token(int code) => byCode[code]?.token ?? '[CM${code + 1}]';
}
