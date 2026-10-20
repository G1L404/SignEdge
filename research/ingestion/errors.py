"""Recusa de amostra: o clipe não vira tensor, e o motivo vai para o manifesto."""


class SampleRejected(Exception):
    """Arquivo vazio, junta fora da foto, ou nenhuma mão para recortar."""
