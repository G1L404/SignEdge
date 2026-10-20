"""Leitura do .oni pelo OpenNI2: profundidade uint16 em mm e cor RGB uint8.

Só I/O. Nenhuma decisão de frame ou de caixa mora aqui.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from research.ingestion.errors import SampleRejected

_OPENNI_READY = False
# Devices de `read_all_frames(close=False)`; a referência impede o coletor de fechá-los.
_LEFT_OPEN: list = []


def require_playable_oni(oni_path: Path) -> None:
    if not oni_path.is_file() or oni_path.stat().st_size == 0:
        raise SampleRejected(f"oni vazio ou ausente: {oni_path.name}")


def _register_depth_to_color(device, openni2) -> None:
    """Projeta a profundidade na câmera colorida, com os streams ainda parados.

    O .oni nem sempre traz a calibração. Sem ela o mapa continua no espaço
    do sensor e a máscara não acompanha a mão na imagem RGB.
    """
    from openni.utils import OpenNIError

    try:
        device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    except OpenNIError:
        return


def _ensure_openni(openni2) -> None:
    global _OPENNI_READY
    if not _OPENNI_READY:
        openni2.initialize()
        _OPENNI_READY = True


def _open_playback(oni_path: Path):
    """Device e streams de profundidade e cor, já registrados e ligados.

    A velocidade fica a do arquivo: com -1 (manual) a leitura de dois streams trava.
    """
    from openni import openni2

    _ensure_openni(openni2)
    device = openni2.Device.open_file(str(oni_path).encode("utf-8"))
    device.playback.set_repeat_enabled(False)
    depth_stream = device.create_depth_stream()
    color_stream = device.create_color_stream()
    _register_depth_to_color(device, openni2)
    depth_stream.start()
    color_stream.start()
    return device, depth_stream, color_stream


def _close_playback(device, depth_stream, color_stream) -> None:
    """Fecha os streams antes do device.

    Fechar o device com os streams ainda abertos dá segfault no libOniFile:
    C2_1_7 caiu em 6 de 12 rodadas; fechando os streams antes, em 0 de 36.
    """
    for stream in (depth_stream, color_stream):
        stream.stop()
        stream.close()
    device.close()


def _grab_pair(depth_stream, color_stream) -> tuple[np.ndarray, np.ndarray]:
    """Próximo par de frames, copiado para fora dos buffers do OpenNI."""
    depth_frame = depth_stream.read_frame()
    color_frame = color_stream.read_frame()
    depth = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).copy()
    depth = depth.reshape((depth_frame.height, depth_frame.width))
    color = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8).copy()
    color = color.reshape((color_frame.height, color_frame.width, 3))
    return depth, color


def _same_grid(depth: np.ndarray, color: np.ndarray) -> np.ndarray:
    import cv2

    if depth.shape[:2] == color.shape[:2]:
        return depth
    return cv2.resize(depth, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)


def read_all_frames(
    oni_path, limit: int | None = None, close: bool = True
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Todos os pares (profundidade uint16 em mm, cor RGB uint8), na mesma grade.

    A reprodução anda em tempo real. Processar entre leituras faz o driver
    descartar frames, e o índice deixa de ser o número do frame; no fim do
    arquivo, sem repetição, a leitura trava. Por isso lemos tudo primeiro.
    O par i é o frame i + 1 do .oni.

    Com `close=False` o device fica aberto até o processo morrer: serve ao
    subprocesso de um clipe só, que sai com `os._exit` sem passar pelo
    fechamento do libOniFile.
    """
    import cv2

    cv2.setNumThreads(0)
    oni_path = Path(oni_path)
    require_playable_oni(oni_path)
    device, depth_stream, color_stream = _open_playback(oni_path)
    try:
        n_frames = int(depth_stream.get_number_of_frames())
        if limit is not None:
            n_frames = min(n_frames, int(limit))
        pairs = [_grab_pair(depth_stream, color_stream) for _ in range(n_frames)]
    except BaseException:
        _close_playback(device, depth_stream, color_stream)
        raise
    if close:
        _close_playback(device, depth_stream, color_stream)
    else:
        _LEFT_OPEN.append((device, depth_stream, color_stream))
    return [(_same_grid(depth, color), color) for depth, color in pairs]
