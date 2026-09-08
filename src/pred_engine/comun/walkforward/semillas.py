"""Derivacion determinista de semillas para ejecuciones asincronas (ASHA)."""

from __future__ import annotations

import hashlib


def derivar_semilla(seed: int, identificador: str, indice_ventana: int) -> int:
    payload = f"{seed}:{identificador}:{indice_ventana}".encode()
    digesto = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digesto, "big") % (2**31 - 1)
