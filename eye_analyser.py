import numpy as np
import pandas as pd
from typing import Any, List
from scipy.ndimage import uniform_filter1d
from scipy.spatial import ConvexHull
import math

try:
    import pywt  # needed for IPA/LHIPA
except Exception:  # pragma: no cover
    pywt = None


class EyeAnalyser:
    # -------------------------
    # Helpers (Fixations)
    # -------------------------
    @staticmethod
    def _scanpath_length(points: np.ndarray) -> float:
        if points.shape[0] < 2:
            return np.nan
        return float(np.nansum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    @staticmethod
    def _spatial_entropy(points: np.ndarray, bins: int = 10) -> float:
        """
        Simple 2D occupancy entropy of fixation points.
        """
        if points.shape[0] < 5:
            return np.nan
        x = points[:, 0]
        y = points[:, 1]
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < 5:
            return np.nan

        # avoid degenerate ranges
        xr = (np.nanmin(x), np.nanmax(x))
        yr = (np.nanmin(y), np.nanmax(y))
        if not np.isfinite(xr[0]) or not np.isfinite(xr[1]) or xr[0] == xr[1]:
            return np.nan
        if not np.isfinite(yr[0]) or not np.isfinite(yr[1]) or yr[0] == yr[1]:
            return np.nan

        H, _, _ = np.histogram2d(x, y, bins=bins, range=[xr, yr])
        p = H.ravel().astype(float)
        s = p.sum()
        if s <= 0:
            return np.nan
        p /= s
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    @staticmethod
    def _saccade_metrics_from_fixations(
        t_start: np.ndarray,
        dur: np.ndarray,
        xy: np.ndarray,
    ) -> dict[str, Any]:
        """
        Mirrors your compute_saccade_metrics(...) logic but from arrays.
        Assumption: `t_start[i]` is fixation start time of event i.
        """
        if len(t_start) < 2:
            return {
                "saccade_count": 0,
                "saccade_amp_mean": np.nan,
                "saccade_amp_sum": np.nan,
                "saccade_vel_mean": np.nan,
                "se_saccade_vel_mean": np.nan,
                "saccade_vel_std": np.nan,
                "saccade_vel_n": 0,
            }

        pts = np.asarray(xy, dtype=float)
        amp = np.linalg.norm(np.diff(pts, axis=0), axis=1)

        t0 = np.asarray(t_start, dtype=float)
        d = np.asarray(dur, dtype=float)

        ifi = np.diff(t0) - d[:-1]
        ifi = np.clip(ifi, 0.05, None)  # 50ms min, same as your code

        vel = amp / ifi
        # clip huge outliers (99th percentile)
        if np.any(np.isfinite(vel)):
            vel = np.clip(vel, None, np.nanpercentile(vel, 99))

        vel_f = vel[np.isfinite(vel)]
        return {
            "saccade_count": int(len(amp)),
            "saccade_amp_mean": float(np.nanmean(amp)) if len(amp) else np.nan,
            "saccade_amp_sum": float(np.nansum(amp)) if len(amp) else np.nan,
            "saccade_vel_mean": float(np.nanmean(vel)) if vel_f.size else np.nan,
            "se_saccade_vel_mean": (
                (float(np.nanstd(vel_f, ddof=1)) / math.sqrt(vel_f.size))
                if vel_f.size > 2
                else np.nan
            ),
            "saccade_vel_std": float(np.nanstd(vel_f, ddof=1))
            if vel_f.size > 2
            else np.nan,
            "saccade_vel_n": int(vel_f.size),
        }

    # -------------------------
    # Helpers (Pupil)
    # -------------------------
    @staticmethod
    def _estimate_fs(t: np.ndarray) -> float:
        if len(t) < 3:
            return np.nan
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) == 0:
            return np.nan
        return float(1.0 / np.median(dt))

    @staticmethod
    def _interp_short_gaps(v: pd.Series, max_gap_n: int) -> pd.Series:
        if max_gap_n < 1:
            return v
        return v.interpolate(
            method="linear",
            limit=max_gap_n,
            limit_direction="both",
            limit_area="inside",
        )

    @staticmethod
    def _rolling_smooth(v: pd.Series, smooth_n: int) -> pd.Series:
        if smooth_n < 3:
            return v
        return v.rolling(
            smooth_n,
            min_periods=max(2, smooth_n // 3),
            center=True,
        ).mean()

    @staticmethod
    def _safe_wavelet_level(
        n: int, wavelet: str = "db4", max_level_cap: int = 3
    ) -> int:
        """
        Return a safe wavelet level for pywt.wavedec.
        Important: do NOT force level >= 2; for short signals max_allowed can be 0/1.
        """
        if pywt is None:
            return 0
        w = pywt.Wavelet(wavelet)
        max_allowed = pywt.dwt_max_level(data_len=n, filter_len=w.dec_len)
        if max_allowed < 1:
            return 0
        return int(min(max_level_cap, max_allowed))

    @staticmethod
    def _ipa_wavelet(x: np.ndarray) -> float:
        if pywt is None:
            return np.nan
        x = x[np.isfinite(x)]
        if len(x) < 64:
            return np.nan
        level = EyeAnalyser._safe_wavelet_level(len(x), wavelet="db4", max_level_cap=3)
        if level < 1:
            return np.nan
        coeffs = pywt.wavedec(x, "db4", level=level)
        details = np.concatenate([c for c in coeffs[1:] if len(c) > 0])
        if len(details) < 10:
            return np.nan
        sigma = np.median(np.abs(details)) / 0.6745 if np.any(details) else 0.0
        thr = sigma * math.sqrt(2.0 * math.log(len(details))) if sigma > 0 else 0.0
        return float(np.mean(np.abs(details) > thr))

    @staticmethod
    def _lhipa_wavelet(x: np.ndarray) -> float:
        if pywt is None:
            return np.nan
        x = x[np.isfinite(x)]
        if len(x) < 64:
            return np.nan
        level = EyeAnalyser._safe_wavelet_level(len(x), wavelet="db4", max_level_cap=3)
        if level < 1:
            return np.nan
        coeffs = pywt.wavedec(x, "db4", level=level)
        details = [c for c in coeffs[1:] if len(c) > 0]
        if len(details) < 2:
            return np.nan
        split = max(1, len(details) // 2)
        high = np.concatenate(details[:split])
        low = np.concatenate(details[split:]) if split < len(details) else np.array([])
        eh = float(np.mean(high**2)) if len(high) else np.nan
        el = float(np.mean(low**2)) if len(low) else np.nan
        if not np.isfinite(eh) or not np.isfinite(el) or el <= 0:
            return np.nan
        return float(eh / el)

    @staticmethod
    def _split_half_se(x: np.ndarray, fn) -> float:
        """
        History-free uncertainty proxy (same idea as your split-half).
        Returns abs diff / 2.
        """
        x = x[np.isfinite(x)]
        if len(x) < 128:
            return np.nan
        mid = len(x) // 2
        a = x[:mid]
        b = x[mid:]
        va = fn(a)
        vb = fn(b)
        if not np.isfinite(va) or not np.isfinite(vb):
            return np.nan
        return float(abs(vb - va) / 2.0)

    # ============================================================
    # Fixation metrics
    # ============================================================
    @staticmethod
    def _fixation_features_by_time_percentiles(
        times: np.ndarray,
        fixation_durations: np.ndarray,
        fixation_xy: np.ndarray,
        boundaries: List[float] = [
            0.0,
            0.33,
            0.66,
            1.0,
        ],  # unused (kept for compatibility)
        max_trial_duration: float = 2.0,
    ) -> dict:
        # sort by time
        idx = np.argsort(times)
        t = np.asarray(times, dtype=float)[idx]
        d = np.asarray(fixation_durations, dtype=float)[idx]
        xy = np.asarray(fixation_xy, dtype=float)[idx]

        valid = np.isfinite(t) & np.isfinite(d) & np.isfinite(xy).all(axis=1)
        t, d, xy = t[valid], d[valid], xy[valid]

        if t.size == 0:
            return {
                "fixation_count": 0,
                "fixation_rate": np.nan,
                "mean_fixation_duration": np.nan,
                "std_fixation_duration": np.nan,
                "dwell_time": np.nan,
                "coord_dispersion": np.nan,
                "hull_area": np.nan,
                "mean_dist_from_centroid": np.nan,
                "scanpath_length": np.nan,
                "spatial_entropy": np.nan,
                "saccade_count": 0,
                "saccade_amp_mean": np.nan,
                "saccade_amp_sum": np.nan,
                "saccade_vel_mean": np.nan,
                "se_saccade_vel_mean": np.nan,
                "saccade_vel_std": np.nan,
                "saccade_vel_n": 0,
                "win_dur_s": np.nan,
                "n_fix_events": 0,
                "se_fixation_rate": np.nan,
                "se_mean_fixation_duration": np.nan,
            }

        t0 = float(t[0])
        t1 = float(min(t[-1], t0 + max_trial_duration))
        win_dur = max(1e-6, t1 - t0)

        # restrict to window
        m = (t >= t0) & (t <= t1)
        t = t[m]
        d = d[m]
        xy = xy[m]

        n_fix = int(len(t))
        fixation_rate = n_fix / win_dur if win_dur > 0 else np.nan

        mean_fix_dur = float(np.nanmean(d)) if n_fix else np.nan
        std_fix_dur = float(np.nanstd(d, ddof=1)) if n_fix > 2 else np.nan
        se_mean_fix_dur = (
            (std_fix_dur / math.sqrt(n_fix))
            if (n_fix > 2 and np.isfinite(std_fix_dur))
            else np.nan
        )

        dwell_time = float(np.nansum(d)) if n_fix else np.nan

        if n_fix:
            disp = float(np.nanstd(xy, axis=0).mean())
            centroid = np.nanmean(xy, axis=0)
            mean_dist = float(np.nanmean(np.linalg.norm(xy - centroid, axis=1)))
        else:
            disp = np.nan
            mean_dist = np.nan

        if n_fix >= 3:
            try:
                hull_area = float(ConvexHull(xy).volume)
            except Exception:
                hull_area = np.nan
        else:
            hull_area = np.nan

        sp_length = EyeAnalyser._scanpath_length(xy) if n_fix else np.nan
        sp_entropy = EyeAnalyser._spatial_entropy(xy, bins=10) if n_fix else np.nan

        sac = EyeAnalyser._saccade_metrics_from_fixations(t_start=t, dur=d, xy=xy)

        se_fixation_rate = (
            math.sqrt(n_fix + 1.0) / win_dur
            if np.isfinite(win_dur) and win_dur > 0
            else np.nan
        )

        return {
            "fixation_count": n_fix,
            "fixation_rate": float(fixation_rate)
            if np.isfinite(fixation_rate)
            else np.nan,
            "mean_fixation_duration": mean_fix_dur,
            "std_fixation_duration": std_fix_dur,
            "dwell_time": dwell_time,
            "coord_dispersion": disp,
            "hull_area": hull_area,
            "mean_dist_from_centroid": mean_dist,
            "scanpath_length": sp_length,
            "spatial_entropy": sp_entropy,
            **sac,
            "win_dur_s": float(win_dur),
            "n_fix_events": n_fix,
            "se_fixation_rate": float(se_fixation_rate)
            if np.isfinite(se_fixation_rate)
            else np.nan,
            "se_mean_fixation_duration": float(se_mean_fix_dur)
            if np.isfinite(se_mean_fix_dur)
            else np.nan,
        }

    # =========================================================
    # Pupil metrics
    # =========================================================
    @staticmethod
    def _pupil_features_by_time_percentiles(
        times: np.ndarray,
        values: np.ndarray,
        boundaries: List[float] = [
            0.0,
            0.33,
            0.66,
            1.0,
        ],  # unused (kept for compatibility)
        peak_prominence: float = 0.1,  # kept (unused now)
        peak_height: float = 0.1,  # kept (unused now)
        peak_distance: int = 5,  # kept (unused now)
        sampling_period: float = 0.01,  # kept (used only as fallback)
        max_trial_duration: float = 2.0,
        baseline_duration: float = 0.2,  # kept (unused in this simplified window version)
        smoothing_window: int = 5,  # kept (fallback smoothing)
    ) -> dict:
        idx = np.argsort(times)
        t = np.asarray(times, dtype=float)[idx]
        v = np.asarray(values, dtype=float)[idx]

        v = np.where(v == 0, np.nan, v)

        if t.size == 0:
            return {
                "mean_pupil": np.nan,
                "pupil_std": np.nan,
                "pupil_cv": np.nan,
                "ipa": np.nan,
                "lhipa": np.nan,
                "n_samples": 0,
                "fs_est": np.nan,
                "win_dur_s": np.nan,
                "se_mean_pupil": np.nan,
                "se_ipa": np.nan,
                "se_lhipa": np.nan,
            }

        t0 = float(t[0])
        t1 = float(min(t[-1], t0 + max_trial_duration))
        win_dur = max(1e-6, t1 - t0)

        m = (t >= t0) & (t <= t1)
        t = t[m]
        v = v[m]

        t, uq_idx = np.unique(t, return_index=True)
        v = v[uq_idx]

        n = int(len(t))
        if n < 2:
            return {
                "mean_pupil": float(np.nanmean(v))
                if np.any(np.isfinite(v))
                else np.nan,
                "pupil_std": np.nan,
                "pupil_cv": np.nan,
                "ipa": np.nan,
                "lhipa": np.nan,
                "n_samples": n,
                "fs_est": np.nan,
                "win_dur_s": float(win_dur),
                "se_mean_pupil": np.nan,
                "se_ipa": np.nan,
                "se_lhipa": np.nan,
            }

        fs = EyeAnalyser._estimate_fs(t)
        pupil_raw = pd.Series(v)

        pupil_clean = pupil_raw.copy()
        if np.isfinite(fs) and fs > 0:
            max_interp_gap_s = 0.25
            smooth_s = 0.10  # TODO: tune if needed

            max_gap_n = int(round(max_interp_gap_s * fs))
            if max_gap_n >= 1:
                pupil_clean = EyeAnalyser._interp_short_gaps(
                    pupil_clean, max_gap_n=max_gap_n
                )

            smooth_n = int(round(smooth_s * fs))
            pupil_clean = EyeAnalyser._rolling_smooth(pupil_clean, smooth_n=smooth_n)
        else:
            pupil_clean = pd.Series(
                uniform_filter1d(
                    pupil_raw.to_numpy(dtype=float), size=max(1, smoothing_window)
                )
            )

        x = pupil_clean.to_numpy(dtype=float)

        mean_pupil = float(np.nanmean(x)) if np.any(np.isfinite(x)) else np.nan
        pupil_std = float(np.nanstd(x, ddof=1)) if np.isfinite(x).sum() > 2 else np.nan
        pupil_cv = (
            (pupil_std / mean_pupil)
            if (np.isfinite(pupil_std) and np.isfinite(mean_pupil) and mean_pupil > 0)
            else np.nan
        )

        ipa = EyeAnalyser._ipa_wavelet(x)
        lhipa = EyeAnalyser._lhipa_wavelet(x)

        n_eff = max(2, int(round(n / 2)))
        se_mean_pupil = (
            (pupil_std / math.sqrt(n_eff))
            if (np.isfinite(pupil_std) and n_eff > 1)
            else np.nan
        )

        se_ipa = EyeAnalyser._split_half_se(x, EyeAnalyser._ipa_wavelet)
        se_lhipa = EyeAnalyser._split_half_se(x, EyeAnalyser._lhipa_wavelet)

        return {
            "mean_pupil": mean_pupil,
            "pupil_std": pupil_std,
            "pupil_cv": pupil_cv,
            "ipa": ipa,
            "lhipa": lhipa,
            "n_samples": n,
            "fs_est": float(fs) if np.isfinite(fs) else np.nan,
            "win_dur_s": float(win_dur),
            "se_mean_pupil": float(se_mean_pupil)
            if np.isfinite(se_mean_pupil)
            else np.nan,
            "se_ipa": float(se_ipa) if np.isfinite(se_ipa) else np.nan,
            "se_lhipa": float(se_lhipa) if np.isfinite(se_lhipa) else np.nan,
        }
