import 'dart:isolate';

import 'package:flutter_test/flutter_test.dart';
import 'package:signedge_mobile/src/vision/inference_isolate.dart';

Handler _echo(Object _) =>
    (request) => request;

Handler _dies(Object _) =>
    (request) => Isolate.exit();

void main() {
  test('worker responde cada pedido', () async {
    final worker = await Worker.spawn(_echo, 0);
    expect(await worker.call('frame'), 'frame');
    worker.dispose();
  });

  test('close libera e encerra, mesmo com o isolate já morto', () async {
    final alive = await Worker.spawn(_echo, 0);
    await alive.close();
    final dead = await Worker.spawn(_dies, 0);
    await dead.close();
  });

  test('isolate que morre falha o pedido pendente e os seguintes', () async {
    final worker = await Worker.spawn(_dies, 0);
    await expectLater(worker.call('frame'), throwsStateError);
    await expectLater(worker.call('frame'), throwsStateError);
    worker.dispose();
  });
}
