from __future__ import annotations

import math
import re
import socket
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class EyeReaderConfig:
    host: str = "127.0.0.1"
    port: int = 4242
    socket_timeout_s: float = 1.0

    # priming (wait until first <REC> arrives)
    prime_timeout_s: float = 2.0
    recv_size_prime: int = 4096
    recv_size_stream: int = 8192

    # metadata column
    keep_time_unit_detected_col: bool = True


class EyeReader:
    """
    Gazepoint TCP reader that:
      - enables pupil + gaze streams
      - incrementally parses complete <REC .../> tags (no duplicates)
      - parses pupil (LPD/RPD preferred, fallback LPMM/RPMM)
      - parses fixation-related fields if present (FPOGX/FPOGY/FPOGS/FPOGV)
      - parses device TIME if present, else uses host time
      - AUTO-detects TIME unit (sec/ms/us) and normalizes to seconds in the output DF

    Important:
      - duration_seconds must be passed by the caller (app/service)
    """

    def __init__(self, cfg: EyeReaderConfig = EyeReaderConfig()) -> None:
        self.cfg = cfg

        num = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        self._rec_pat = re.compile(r"<REC\b[^>]*?/?>")
        self._time_pat = re.compile(rf'\bTIME="({num})"')

        # Prefer PUPILMM (LPD/RPD). Some streams use LPMM/RPMM.
        self._pupil_pat = re.compile(
            rf'\b(?:LPD|LPMM)="({num})".*?\b(?:RPD|RPMM)="({num})"'
        )

        # Fixation point of gaze (normalized). Optional FPOGS, optional FPOGV.
        self._fix_pat = re.compile(
            rf'\bFPOGX="({num})".*?\bFPOGY="({num})"'
            rf'(?:.*?\bFPOGS="({num})")?'
            rf'(?:.*?\bFPOGV="([01])")?'
        )

    def _enable_streams(self, s: socket.socket) -> None:
        cmds = [
            '<SET ID="ENABLE_SEND_PUPILMM" STATE="1" />\r\n',
            '<SET ID="ENABLE_SEND_EYE_LEFT" STATE="1" />\r\n',
            '<SET ID="ENABLE_SEND_EYE_RIGHT" STATE="1" />\r\n',
            '<SET ID="ENABLE_SEND_POG_FIX" STATE="1" />\r\n',
            '<SET ID="ENABLE_SEND_VALIDITY" STATE="1" />\r\n',
            '<SET ID="ENABLE_SEND_CONFIDENCE" STATE="1" />\r\n',
            '<SET ID="ENABLE_SEND_TIME" STATE="1" />\r\n',
            '<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n',
        ]
        for c in cmds:
            s.sendall(c.encode())

    def _stop_streams(self, s: socket.socket) -> None:
        try:
            s.sendall('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'.encode())
        except Exception:
            pass

    def _parse_one_rec(self, rec: str) -> dict[str, float | int]:
        # TIME from device if present; fallback host seconds
        tm = self._time_pat.search(rec)
        ts = float(tm.group(1)) if tm else time.time()

        lp = math.nan
        rp = math.nan
        fx = math.nan
        fy = math.nan
        fpogs = math.nan
        fpogv = math.nan

        pm = self._pupil_pat.search(rec)
        if pm:
            lp = float(pm.group(1))
            rp = float(pm.group(2))

        fm = self._fix_pat.search(rec)
        if fm:
            fx = float(fm.group(1))
            fy = float(fm.group(2))
            if fm.group(3) is not None:
                fpogs = float(fm.group(3))
            if fm.group(4) is not None:
                fpogv = int(fm.group(4))

        return {
            "TIME": ts,
            "LPD": lp,  # pupil diameter (or LPMM fallback)
            "RPD": rp,  # pupil diameter (or RPMM fallback)
            "FPOGX": fx,  # normalized [0..1]
            "FPOGY": fy,  # normalized [0..1]
            "FPOGS": fpogs,
            "FPOGV": fpogv,  # 0/1 if present
        }

    @staticmethod
    def _normalize_time_to_seconds(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """
        Detect whether TIME is in seconds, milliseconds, or microseconds.
        Uses median diff to infer unit. Returns (df, detected_unit).
        """
        if "TIME" not in df.columns or len(df) < 2:
            return df, "unknown"

        dt = pd.to_numeric(df["TIME"], errors="coerce").diff().dropna()
        if dt.empty:
            return df, "unknown"

        med = float(dt.median())

        unit = "seconds"
        scale = 1.0
        if med > 1000.0:
            unit = "microseconds"
            scale = 1_000_000.0
        elif med > 1.0:
            unit = "milliseconds"
            scale = 1000.0

        out = df.copy()
        out["TIME"] = pd.to_numeric(out["TIME"], errors="coerce") / scale
        return out, unit

    def collect_eye_dataframe(self, duration_seconds: float) -> pd.DataFrame:
        """
        Collect records for duration_seconds and return a DataFrame.
        Output TIME is normalized to seconds automatically.

        Caller must pass duration_seconds (do not read config inside the reader).
        """
        gaze_socket: Optional[socket.socket] = None
        rows: list[dict[str, float | int]] = []
        buf = ""

        try:
            gaze_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            gaze_socket.connect((self.cfg.host, self.cfg.port))
            gaze_socket.settimeout(self.cfg.socket_timeout_s)

            self._enable_streams(gaze_socket)

            # Prime until first REC arrives
            deadline = time.time() + self.cfg.prime_timeout_s
            while time.time() < deadline:
                try:
                    chunk = gaze_socket.recv(self.cfg.recv_size_prime)
                    if not chunk:
                        continue
                    buf += chunk.decode(errors="ignore")
                    if "<REC" in buf:
                        break
                except socket.timeout:
                    pass

            start = time.time()
            while (time.time() - start) < float(duration_seconds):
                try:
                    chunk = gaze_socket.recv(self.cfg.recv_size_stream)
                    if not chunk:
                        continue
                    buf += chunk.decode(errors="ignore")

                    recs = self._rec_pat.findall(buf)
                    if not recs:
                        continue

                    # Keep remainder after last complete REC
                    last = recs[-1]
                    last_end = buf.rfind(last) + len(last)
                    buf = buf[last_end:]

                    for rec in recs:
                        row = self._parse_one_rec(rec)

                        # Keep if anything besides TIME is present
                        has_any = any(
                            not math.isnan(row[k])  # type: ignore[arg-type]
                            for k in ("LPD", "RPD", "FPOGX", "FPOGY", "FPOGS")
                        )
                        if has_any:
                            rows.append(row)

                except socket.timeout:
                    continue

            df = pd.DataFrame(rows)
            df, detected_unit = self._normalize_time_to_seconds(df)

            if self.cfg.keep_time_unit_detected_col:
                df["TIME_UNIT_DETECTED"] = detected_unit

            return df

        except Exception as e:
            print(f"[EyeReader] Gazepoint connection error: {e}")
            return pd.DataFrame([])

        finally:
            if gaze_socket is not None:
                self._stop_streams(gaze_socket)
                try:
                    gaze_socket.close()
                except Exception:
                    pass
