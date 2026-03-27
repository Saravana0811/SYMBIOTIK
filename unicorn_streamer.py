# demo/CDR/eeg/unicorn_streamer.py
from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional


@dataclass(frozen=True)
class StreamSample:
    t_sec: float
    values: Dict[str, float]  # channel -> value


class UnicornTCPStreamer:
    """
    Minimal deployment-ready TCP streamer for Unicorn-like socket streams.

    Assumptions:
      - TCP payload contains float32 values for ALL channels in fixed order.
      - Each float32 is 4 bytes, little-endian.
      - One "sample" = len(channels) float32 values.
      - Device sends continuous byte stream (may split across recv calls).

    This yields StreamSample(t_sec, {ch: value}).

    NOTE: If your Unicorn server uses a different framing/protocol (headers, packets),
          adapt _extract_frames() accordingly.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 51234,
        channels: Optional[List[str]] = None,
        recv_bytes: int = 8192,
        socket_timeout_s: float = 5.0,
    ):
        self.host = host
        self.port = int(port)
        self.channels = channels or ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
        self.recv_bytes = int(recv_bytes)
        self.socket_timeout_s = float(socket_timeout_s)

        self._sock: Optional[socket.socket] = None
        self._buf = bytearray()

        self._frame_bytes = 4 * len(self.channels)
        self._struct = struct.Struct("<" + "f" * len(self.channels))  # little-endian float32

    def connect(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.socket_timeout_s)
        s.connect((self.host, self.port))
        self._sock = s

    def close(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        finally:
            self._sock = None
            self._buf = bytearray()

    def __enter__(self) -> "UnicornTCPStreamer":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _extract_frames(self) -> Iterator[Dict[str, float]]:
        while len(self._buf) >= self._frame_bytes:
            frame = self._buf[: self._frame_bytes]
            del self._buf[: self._frame_bytes]
            vals = self._struct.unpack(frame)
            yield {self.channels[i]: float(vals[i]) for i in range(len(self.channels))}

    def samples(self) -> Iterator[StreamSample]:
        if self._sock is None:
            raise RuntimeError("Not connected. Call connect() first.")

        while True:
            chunk = self._sock.recv(self.recv_bytes)
            if not chunk:
                break
            self._buf.extend(chunk)
            for frame in self._extract_frames():
                yield StreamSample(t_sec=time.time(), values=frame)