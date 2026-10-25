import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:signedge_mobile/src/logic/decision.dart';
import 'package:signedge_mobile/src/logic/handshapes.dart';

void main() {
  final catalog = HandshapeCatalog.parse(
    File('assets/handshapes.json').readAsStringSync(),
  );

  test('leitura da própria CM, com a alternativa', () {
    expect(catalog.readingOf(26), 'C');
    expect(catalog.readingOf(38), 'L / 2');
  });

  test('variante herda a leitura e a descrição do grupo', () {
    expect(catalog.readingOf(22), 'C');
    expect(catalog.shapeOf(22), catalog.shapeOf(26));
    // Variante com entrada própria fica com a sua descrição.
    expect(catalog.shapeOf(47), isNot(catalog.shapeOf(44)));
  });

  test('CM sem letra nem grupo não tem leitura', () {
    expect(catalog.readingOf(39), isNull);
    expect(catalog.readingOf(30), isNull);
  });

  test('confiança dividida entre letras diferentes vira "C ou O"', () {
    // Códigos = CM − 1: CM 22 (C), CM 21 (O), CM 17 abaixo do mínimo.
    final ranked = [Ranked(21, 0.6), Ranked(20, 0.3), Ranked(16, 0.05)];
    expect(catalog.readingsAmong(ranked), ['C', 'O / 0']);
    // A mesma letra duas vezes (CM 26 e a variante 22) aparece uma vez só.
    expect(catalog.readingsAmong([Ranked(25, 0.5), Ranked(21, 0.4)]), ['C']);
  });

  test('leitura ainda sem revisão', () {
    expect(catalog.reviewed, isFalse);
  });
}
