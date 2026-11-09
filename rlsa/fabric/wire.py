"""Length-prefixed pickle framing over a stream socket.

Wire format: 4-byte big-endian unsigned length, then that many bytes of a
pickled Python object. Transport only -- no domain knowledge.

Security note: pickle.loads runs arbitrary code from the payload. All peers
must be trusted (same FABRIC slice / localhost). Do not expose these sockets
to an untrusted network.
"""
from __future__ import annotations

import pickle
import socket
import struct
from typing import Any, Optional

_HEADER = struct.Struct(">I")


def send_msg(sock: socket.socket, obj: Any) -> None:
    body = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(body)) + body)


def _recv_exactly(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket) -> Optional[Any]:
    """Returns the next object, or None if the peer closed cleanly before a
    message started. Raises EOFError if the peer closed mid-message."""
    header = _recv_exactly(sock, _HEADER.size)
    if header is None:
        return None
    (length,) = _HEADER.unpack(header)
    body = _recv_exactly(sock, length)
    if body is None:
        raise EOFError(f"connection closed mid-message (expected {length} body bytes)")
    return pickle.loads(body)
