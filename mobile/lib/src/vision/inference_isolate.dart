/// Os dois isolates de inferência: a mão (GPU) e o SignEdge (CPU).
///
/// O frame chega como os planos YUV do sensor (TransferableTypedData, sem cópia
/// extra). Cada pedido leva um número, e a resposta volta com ele: um "reset" no
/// meio do caminho nunca é confundido com o resultado de um frame.
library;

import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'frame.dart';
import 'pipeline.dart';

class ModelBytes {
  const ModelBytes({
    required this.palm,
    required this.landmarks,
    required this.classifier,
  });
  final Uint8List palm, landmarks, classifier;
}

/// Esquece a mão rastreada (troca de câmera).
class ResetTracking {
  const ResetTracking();
}

typedef Handler = Object? Function(Object? request);

/// Um isolate que responde a cada pedido com o tratador montado por `setup`.
class Worker {
  Worker._(this._isolate, this._requests, Stream<dynamic> replies) {
    replies.listen((message) {
      final (id, payload) = message as (int, Object?);
      final completer = _pending.remove(id);
      if (payload is RemoteError) {
        completer?.completeError(payload);
      } else {
        completer?.complete(payload);
      }
    });
  }

  /// [setup] roda dentro do isolate com [init] e devolve o tratador dos pedidos.
  static Future<Worker> spawn(
    Handler Function(Object init) setup,
    Object init,
  ) async {
    final replies = ReceivePort();
    final stream = replies.asBroadcastStream();
    final isolate = await Isolate.spawn(_main, (replies.sendPort, setup, init));
    final first = await stream.first;
    if (first is RemoteError) throw first;
    return Worker._(isolate, first as SendPort, stream);
  }

  final Isolate _isolate;
  final SendPort _requests;
  final Map<int, Completer<Object?>> _pending = {};
  int _next = 0;

  Future<Object?> call(Object request) {
    final id = _next++;
    final completer = _pending[id] = Completer<Object?>();
    _requests.send((id, request));
    return completer.future;
  }

  void dispose() => _isolate.kill(priority: Isolate.immediate);

  static void _main((SendPort, Handler Function(Object), Object) args) {
    final (replies, setup, init) = args;
    final Handler handle;
    try {
      handle = setup(init);
    } catch (error, stack) {
      replies.send(RemoteError('$error', '$stack'));
      return;
    }
    final inbox = ReceivePort();
    replies.send(inbox.sendPort);
    inbox.listen((message) {
      final (id, request) = message as (int, Object?);
      try {
        replies.send((id, handle(request)));
      } catch (error, stack) {
        replies.send((id, RemoteError('$error', '$stack')));
      }
    });
  }
}

/// Isolate da mão: palma, pontos e o recorte do SignEdge.
Handler handWorker(Object init) {
  final (palm, landmarks) = init as (Uint8List, Uint8List);
  final stage = HandStage(palm, landmarks);
  return (request) {
    if (request is ResetTracking) {
      stage.reset();
      return null;
    }
    final clock = Stopwatch()..start();
    final frame = (request as YuvPlanes).materialize();
    final planes = clock.elapsedMicroseconds / 1000;
    final result = stage.process(frame);
    result.stages['isolate.planos'] = planes;
    return result;
  };
}

/// Isolate do SignEdge: só o classificador Int8.
Handler classifierWorker(Object init) {
  final stage = ClassifierStage(init as Uint8List);
  return (request) => stage.process(request as ClassifierRequest);
}
