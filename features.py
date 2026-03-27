from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from CDR.eye.models.eye_analyser import EyeAnalyser
from CDR.eye.models.fixation_detection_2d import FixationDetector2D
from CDR.eye.models.eye_reader import EyeReader


@dataclass(frozen=True)
class EyeFeatureParams:
    velocity_threshold: float = 20.0
    min_fixation_duration_ms: float = 50.0
    use_validity_filter: bool = True  # if FPOGV exists, keep only == 1


def compute_eye_features_from_df(
    raw_data: pd.DataFrame,
    window_s: float,
    params: EyeFeatureParams = EyeFeatureParams(),
) -> dict[str, Any]:
    """
    Pure feature extraction: window dataframe -> feature dict.
    No file I/O, no config, no RL calls.
    """
    if raw_data is None or raw_data.empty or "TIME" not in raw_data.columns:
        return {}

    df = raw_data.copy()

    if params.use_validity_filter and "FPOGV" in df.columns:
        df = df[df["FPOGV"] == 1]

    raw_fix = FixationDetector2D(
        velocity_threshold=params.velocity_threshold,
        min_fixation_duration_ms=params.min_fixation_duration_ms,
    ).filter_fixations(df)

    analyser = EyeAnalyser()

    # Fixation features
    if raw_fix.empty:
        fix_features = analyser._fixation_features_by_time_percentiles(
            times=np.array([]),
            fixation_durations=np.array([]),
            fixation_xy=np.empty((0, 2)),
            max_trial_duration=float(window_s),
        )
    else:
        relative_fix_times = raw_fix["TIME"].to_numpy(dtype=float) - float(
            raw_fix["TIME"].iloc[0]
        )
        fix_features = analyser._fixation_features_by_time_percentiles(
            times=relative_fix_times,
            fixation_durations=raw_fix["FPOGD"].to_numpy(dtype=float) / 1000.0,
            fixation_xy=raw_fix[["FPOGX", "FPOGY"]].to_numpy(dtype=float),
            max_trial_duration=float(window_s),
        )

    # Pupil features (LPD/RPD preferred)
    if "LPD" in raw_data.columns and "RPD" in raw_data.columns:
        vals = (raw_data["LPD"] + raw_data["RPD"]).to_numpy(dtype=float) / 2.0
    elif "LPMM" in raw_data.columns and "RPMM" in raw_data.columns:
        vals = (raw_data["LPMM"] + raw_data["RPMM"]).to_numpy(dtype=float) / 2.0
    else:
        vals = np.full(len(raw_data), np.nan, dtype=float)

    vals[vals == 0] = np.nan

    relative_times = raw_data["TIME"].to_numpy(dtype=float) - float(
        raw_data["TIME"].iloc[0]
    )

    pup_features = analyser._pupil_features_by_time_percentiles(
        times=relative_times,
        values=vals,
        max_trial_duration=float(window_s),
    )

    return {**fix_features, **pup_features}


def compute_eye_features_from_reader(
    reader: EyeReader,
    window_s: float,
    params: EyeFeatureParams = EyeFeatureParams(),
) -> dict[str, Any]:
    """
    Convenience wrapper: reader -> dataframe -> features.
    """
    raw = reader.collect_eye_dataframe(duration_seconds=float(window_s))
    return compute_eye_features_from_df(raw, window_s=float(window_s), params=params)
