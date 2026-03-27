import pandas as pd
import numpy as np


class FixationDetector2D:
    """
    Simple velocity-threshold (I-VT style) fixation detector.

    Assumptions:
    - x,y are normalized [0..1] (or consistent units)
    - TIME is in seconds and should be strictly increasing
    """

    def __init__(
        self,
        velocity_threshold: float = 0.05,
        min_fixation_duration_ms: float = 100.0,
    ):
        self.velocity_threshold = float(velocity_threshold)
        self.min_fixation_duration_ms = float(min_fixation_duration_ms)

    def filter_fixations(
        self,
        df: pd.DataFrame,
        x_col: str = "FPOGX",
        y_col: str = "FPOGY",
        timestamp_col: str = "TIME",
    ) -> pd.DataFrame:
        cols = [timestamp_col, x_col, y_col]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        d = df[cols].copy()
        d = d.dropna(subset=cols)
        if len(d) < 2:
            return pd.DataFrame(columns=["FPOGX", "FPOGY", "FPOGD", "TIME"])

        # sort and enforce strictly increasing time
        d = d.sort_values(timestamp_col).reset_index(drop=True)
        dt = d[timestamp_col].diff()

        # drop non-increasing timestamps to avoid dt<=0 artifacts
        keep = dt.isna() | (dt > 0)
        d = d[keep].reset_index(drop=True)
        if len(d) < 2:
            return pd.DataFrame(columns=["FPOGX", "FPOGY", "FPOGD", "TIME"])

        # recompute diffs after cleaning
        dt = d[timestamp_col].diff()
        dx = d[x_col].diff()
        dy = d[y_col].diff()

        dist = np.sqrt(dx**2 + dy**2)
        velocity = dist / dt

        # classify: first sample treated as fixation by convention
        movement_type = np.where(
            velocity >= self.velocity_threshold, "saccade", "fixation"
        )
        movement_type[0] = "fixation"
        d["movement_type"] = movement_type

        # group consecutive types
        d["group"] = (d["movement_type"] != d["movement_type"].shift()).cumsum()

        fix = d[d["movement_type"] == "fixation"].copy()
        if fix.empty:
            return pd.DataFrame(columns=["FPOGX", "FPOGY", "FPOGD", "TIME"])

        g = fix.groupby("group", as_index=False)
        summary = g.agg(
            FPOGX=(x_col, "mean"),
            FPOGY=(y_col, "mean"),
            TIME=(timestamp_col, "first"),
            end_time=(timestamp_col, "last"),
        )
        summary["FPOGD"] = (summary["end_time"] - summary["TIME"]) * 1000.0
        summary = summary.drop(columns=["end_time"])

        # filter short fixations
        summary = summary[
            summary["FPOGD"] >= self.min_fixation_duration_ms
        ].reset_index(drop=True)
        return summary[["FPOGX", "FPOGY", "FPOGD", "TIME"]]
