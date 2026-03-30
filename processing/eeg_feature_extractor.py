# demo/CDR/eeg/eeg_feature_extractor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, detrend, filtfilt, welch


@dataclass(frozen=True)
class EEGFeatureWindow:
    start_t_sec: float
    end_t_sec: float
    features: Dict[str, float]
    bad_window: int
    qc_reason: str


def _butter_filter(
    fs: float,
    hp_hz: Optional[float] = None,
    lp_hz: Optional[float] = None,
    order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    if hp_hz is not None and lp_hz is not None:
        return butter(order, [hp_hz / nyq, lp_hz / nyq], btype="bandpass")
    if hp_hz is not None:
        return butter(order, hp_hz / nyq, btype="highpass")
    if lp_hz is not None:
        return butter(order, lp_hz / nyq, btype="lowpass")
    raise ValueError("hp_hz and/or lp_hz must be provided")


def _winsorize(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    if x.size == 0:
        return x
    a = np.nanpercentile(x, lo)
    b = np.nanpercentile(x, hi)
    if not np.isfinite(a) or not np.isfinite(b) or a >= b:
        return x
    return np.clip(x, a, b)


def _preprocess(
    x: np.ndarray,
    fs: float,
    hp_hz: float = 0.5,
    lp_hz: Optional[float] = 45.0,
    order: int = 4,
    do_linear_detrend: bool = True,
    winsorize: bool = True,
    win_lo: float = 1.0,
    win_hi: float = 99.0,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 64:
        return x

    # Robust against baseline drift / DC shifts
    x = detrend(x, type="constant")
    if do_linear_detrend:
        x = detrend(x, type="linear")

    # Robust against slow drift / muscle noise: bandpass
    b, a = _butter_filter(fs=fs, hp_hz=hp_hz, lp_hz=lp_hz, order=order)
    if x.size > 3 * max(len(a), len(b)):
        x = filtfilt(b, a, x)

    # Robust against spikes (outliers)
    if winsorize:
        x = _winsorize(x, lo=win_lo, hi=win_hi)

    return x


def _bandpower_welch(x: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 64:
        return float("nan")

    nperseg = min(1024, x.size)
    f, pxx = welch(
        x,
        fs=fs,
        nperseg=nperseg,
        detrend=False,
        scaling="density",
        window="hann",
    )
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m):
        return float("nan")
    return float(np.trapz(pxx[m], f[m]))


class EEGFeatureExtractor5s:
    """
    Accumulates stream samples into fixed windows (default 5s) and outputs features.

    Drift/global-effect handling:
      - constant + linear detrend
      - bandpass (0.5–45 Hz default)
      - winsorization to suppress spikes
    """

    def __init__(
        self,
        fs: float = 250.0,
        window_s: float = 5.0,
        theta_band: Tuple[float, float] = (4.0, 7.0),
        alpha_band: Tuple[float, float] = (8.0, 12.0),
        frontal_channels: Optional[List[str]] = None,
        parietal_channels: Optional[List[str]] = None,
        hp_hz: float = 0.5,
        lp_hz: Optional[float] = 45.0,
        filter_order: int = 4,
        do_linear_detrend: bool = True,
        winsorize: bool = True,
        winsor_lo: float = 1.0,
        winsor_hi: float = 99.0,
        min_samples: int = 800,
        min_var: float = 1e-12,
        max_ptp: float = 1e9,
    ):
        self.fs = float(fs)
        self.window_s = float(window_s)

        self.theta_band = theta_band
        self.alpha_band = alpha_band

        self.frontal_channels = frontal_channels or ["Fz", "C3", "Cz", "C4"]
        self.parietal_channels = parietal_channels or ["Pz", "PO7", "Oz", "PO8"]

        self.hp_hz = float(hp_hz)
        self.lp_hz = float(lp_hz) if lp_hz is not None else None
        self.filter_order = int(filter_order)
        self.do_linear_detrend = bool(do_linear_detrend)

        self.winsorize = bool(winsorize)
        self.winsor_lo = float(winsor_lo)
        self.winsor_hi = float(winsor_hi)

        self.min_samples = int(min_samples)
        self.min_var = float(min_var)
        self.max_ptp = float(max_ptp)

        self._t0: Optional[float] = None
        self._t_last: Optional[float] = None
        self._buf: Dict[str, List[float]] = {}

    def reset(self) -> None:
        self._t0 = None
        self._t_last = None
        self._buf = {}

    def push(self, t_sec: float, values: Dict[str, float]) -> Optional[EEGFeatureWindow]:
        if self._t0 is None:
            self._t0 = float(t_sec)
        self._t_last = float(t_sec)

        for ch, v in values.items():
            self._buf.setdefault(ch, []).append(float(v))

        if (float(t_sec) - float(self._t0)) < self.window_s:
            return None

        win = self._compute_window()
        self.reset()
        return win

    def _compute_window(self) -> EEGFeatureWindow:
        assert self._t0 is not None and self._t_last is not None

        feats: Dict[str, float] = {}
        bad = 0
        reason = ""

        any_ch = next(iter(self._buf.keys()), None)
        n = len(self._buf.get(any_ch, [])) if any_ch else 0
        feats["n_samples"] = float(n)

        if n < self.min_samples:
            return EEGFeatureWindow(self._t0, self._t_last, feats, 1, "too_few_samples")

        for ch, series in self._buf.items():
            x_raw = np.asarray(series, dtype=float)
            x = _preprocess(
                x_raw,
                fs=self.fs,
                hp_hz=self.hp_hz,
                lp_hz=self.lp_hz,
                order=self.filter_order,
                do_linear_detrend=self.do_linear_detrend,
                winsorize=self.winsorize,
                win_lo=self.winsor_lo,
                win_hi=self.winsor_hi,
            )

            if x.size < 64:
                feats[f"{ch}_theta_power"] = float("nan")
                feats[f"{ch}_alpha_power"] = float("nan")
                continue

            v = float(np.var(x))
            ptp = float(np.ptp(x))
            feats[f"{ch}_var"] = v
            feats[f"{ch}_ptp"] = ptp

            if bad == 0 and v < self.min_var:
                bad = 1
                reason = "flatline"
            if bad == 0 and ptp > self.max_ptp:
                bad = 1
                reason = "artifact"

            feats[f"{ch}_theta_power"] = _bandpower_welch(x, self.fs, *self.theta_band)
            feats[f"{ch}_alpha_power"] = _bandpower_welch(x, self.fs, *self.alpha_band)

        frontal_theta = [
            feats.get(f"{ch}_theta_power", np.nan) for ch in self.frontal_channels
        ]
        parietal_alpha = [
            feats.get(f"{ch}_alpha_power", np.nan) for ch in self.parietal_channels
        ]

        frontal_theta = [v for v in frontal_theta if np.isfinite(v)]
        parietal_alpha = [v for v in parietal_alpha if np.isfinite(v)]

        feats["frontal_theta_mean"] = float(np.mean(frontal_theta)) if frontal_theta else float("nan")
        feats["parietal_alpha_mean"] = float(np.mean(parietal_alpha)) if parietal_alpha else float("nan")

        return EEGFeatureWindow(
            start_t_sec=float(self._t0),
            end_t_sec=float(self._t_last),
            features=feats,
            bad_window=int(bad),
            qc_reason=reason,
        )