from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Optional

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

# ------------------------------------------------------------
# Local imports from your existing codebase
# Keep this file in the same folder as:
#   features.py
#   fixation_detection_2d.py
#   eye_analyser.py
#   eeg_feature_extractor.py
# ------------------------------------------------------------
from eeg_feature_extractor import EEGFeatureExtractor5s
from fixation_detection_2d import FixationDetector2D
from eye_analyser import EyeAnalyser


# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "Symbiotic System"
KEYCLOAK_FILE = "user_keycloak.json"

WINDOW_S = 5.0

EEG_FS = 250.0
EEG_CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

# A practical synthetic rate for the eye stream
EYE_FS = 60.0


# ============================================================
# HELPERS
# ============================================================
def clean_numeric_dict(data: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            cleaned[key] = clean_numeric_dict(value)
        elif isinstance(value, float):
            if not (math.isnan(value) or math.isinf(value)):
                cleaned[key] = value
        else:
            cleaned[key] = value
    return cleaned


def load_json_file(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json_file(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ============================================================
# EYE FEATURE COMPUTATION
# Uses the same logic as your eye pipeline but avoids the
# package import problems in features.py.
# ============================================================
class EyeFeatureParams:
    def __init__(
        self,
        velocity_threshold: float = 20.0,
        min_fixation_duration_ms: float = 50.0,
        use_validity_filter: bool = True,
    ) -> None:
        self.velocity_threshold = velocity_threshold
        self.min_fixation_duration_ms = min_fixation_duration_ms
        self.use_validity_filter = use_validity_filter


def compute_eye_features_from_df(
    raw_data: pd.DataFrame,
    window_s: float,
    params: EyeFeatureParams | None = None,
) -> dict[str, Any]:
    if params is None:
        params = EyeFeatureParams()

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


# ============================================================
# SIMULATED EEG
# 250 Hz, 8 channels, mild rhythms + noise
# ============================================================
class SimulatedEEGStreamer:
    def __init__(self, fs: float = EEG_FS, channels: list[str] | None = None) -> None:
        self.fs = float(fs)
        self.channels = channels or EEG_CHANNELS
        self.dt = 1.0 / self.fs

        self._sample_index = 0
        self._t0 = time.time()

        # fixed per-channel characteristics
        self._amp = {
            "Fz": 18.0,
            "C3": 16.0,
            "Cz": 15.0,
            "C4": 16.0,
            "Pz": 14.0,
            "PO7": 13.0,
            "Oz": 14.0,
            "PO8": 13.0,
        }
        self._phase = {
            ch: np.random.uniform(0, 2 * np.pi) for ch in self.channels
        }

    def next_sample(self) -> tuple[float, dict[str, float]]:
        t_sec = self._sample_index * self.dt
        values: dict[str, float] = {}

        for ch in self.channels:
            # theta + alpha + slow drift + white noise
            theta = 7.0 * np.sin(2 * np.pi * 5.5 * t_sec + self._phase[ch])
            alpha = 10.0 * np.sin(2 * np.pi * 10.0 * t_sec + 0.5 * self._phase[ch])

            # stronger posterior alpha in parietal/occipital channels
            if ch in {"Pz", "PO7", "Oz", "PO8"}:
                alpha *= 1.6

            # slightly stronger frontal theta
            if ch in {"Fz", "C3", "Cz", "C4"}:
                theta *= 1.3

            drift = 2.0 * np.sin(2 * np.pi * 0.2 * t_sec + 0.2 * self._phase[ch])
            noise = np.random.normal(0.0, self._amp[ch] * 0.20)

            values[ch] = float(theta + alpha + drift + noise)

        self._sample_index += 1
        return t_sec, values


# ============================================================
# SIMULATED EYE / GAZEPOINT-LIKE WINDOW
# Generates TIME, LPD, RPD, FPOGX, FPOGY, FPOGS, FPOGV
# compatible with your feature pipeline assumptions.
# ============================================================
class SimulatedEyeReader:
    def __init__(self, fs: float = EYE_FS) -> None:
        self.fs = float(fs)
        self.dt = 1.0 / self.fs

        self.screen_center = np.array([0.5, 0.5], dtype=float)
        self.current_fix = self.screen_center.copy()
        self.global_time = 0.0

    def _new_fixation_target(self) -> np.ndarray:
        x = np.random.uniform(0.15, 0.85)
        y = np.random.uniform(0.15, 0.85)
        return np.array([x, y], dtype=float)

    def collect_eye_dataframe(self, duration_seconds: float) -> pd.DataFrame:
        n = max(2, int(round(self.fs * duration_seconds)))
        rows: list[dict[str, float | int]] = []

        # Build fixation-style gaze with occasional saccades
        i = 0
        fix_start_t = self.global_time

        while i < n:
            # fixation duration between 120 ms and 450 ms
            fix_len = int(np.random.randint(
                max(3, int(0.12 * self.fs)),
                max(4, int(0.45 * self.fs))
            ))
            fix_len = min(fix_len, n - i)

            next_target = self._new_fixation_target()

            # occasional short saccade transition
            sacc_len = int(np.random.randint(
                max(1, int(0.02 * self.fs)),
                max(2, int(0.05 * self.fs))
            ))
            sacc_len = min(sacc_len, n - i)

            # fixation samples clustered around current fixation point
            for _ in range(fix_len):
                t = self.global_time
                gaze_xy = self.current_fix + np.random.normal(0.0, 0.008, size=2)

                # keep normalized range
                gaze_xy = np.clip(gaze_xy, 0.0, 1.0)

                # pupil around ~3.2 mm with slow variation
                pupil_base = 3.2 + 0.15 * np.sin(2 * np.pi * 0.15 * t)
                left = pupil_base + np.random.normal(0.0, 0.05)
                right = pupil_base + np.random.normal(0.0, 0.05)

                valid = 1 if np.random.rand() > 0.03 else 0

                rows.append(
                    {
                        "TIME": float(t),
                        "LPD": float(max(0.1, left)),
                        "RPD": float(max(0.1, right)),
                        "FPOGX": float(gaze_xy[0]),
                        "FPOGY": float(gaze_xy[1]),
                        "FPOGS": float(fix_start_t),
                        "FPOGV": int(valid),
                    }
                )

                self.global_time += self.dt
                i += 1
                if i >= n:
                    break

            if i >= n:
                break

            # saccade transition samples between current fixation and next
            transition = np.linspace(self.current_fix, next_target, num=max(2, sacc_len))
            for p in transition:
                if i >= n:
                    break

                t = self.global_time
                gaze_xy = p + np.random.normal(0.0, 0.003, size=2)
                gaze_xy = np.clip(gaze_xy, 0.0, 1.0)

                pupil_base = 3.2 + 0.15 * np.sin(2 * np.pi * 0.15 * t)
                left = pupil_base + np.random.normal(0.0, 0.05)
                right = pupil_base + np.random.normal(0.0, 0.05)

                valid = 1 if np.random.rand() > 0.03 else 0

                rows.append(
                    {
                        "TIME": float(t),
                        "LPD": float(max(0.1, left)),
                        "RPD": float(max(0.1, right)),
                        "FPOGX": float(gaze_xy[0]),
                        "FPOGY": float(gaze_xy[1]),
                        "FPOGS": float(fix_start_t),
                        "FPOGV": int(valid),
                    }
                )

                self.global_time += self.dt
                i += 1

            self.current_fix = next_target.copy()
            fix_start_t = self.global_time

        df = pd.DataFrame(rows)
        return df


# ============================================================
# EEG WORKER THREAD
# ============================================================
class EEGWorker(threading.Thread):
    def __init__(self, out_queue: Queue, stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self.out_queue = out_queue
        self.stop_event = stop_event

    def run(self) -> None:
        streamer = SimulatedEEGStreamer(fs=EEG_FS, channels=EEG_CHANNELS)
        extractor = EEGFeatureExtractor5s(fs=EEG_FS, window_s=WINDOW_S)

        sample_sleep = 1.0 / EEG_FS

        while not self.stop_event.is_set():
            t_sec, values = streamer.next_sample()
            win = extractor.push(t_sec, values)

            if win is not None:
                eeg_payload = {
                    "start_t_sec": win.start_t_sec,
                    "end_t_sec": win.end_t_sec,
                    "bad_window": win.bad_window,
                    "qc_reason": win.qc_reason,
                    **win.features,
                }
                self.out_queue.put(clean_numeric_dict(eeg_payload))

            time.sleep(sample_sleep)


# ============================================================
# MAIN APP
# ============================================================
class SymbioticSystemApp:
    def __init__(self, root: tk.Tk, project_dir: Path) -> None:
        self.root = root
        self.project_dir = project_dir
        self.root.title(APP_TITLE)
        self.root.geometry("860x700")
        self.root.minsize(760, 620)

        self.keycloak_path = self.project_dir / KEYCLOAK_FILE
        self.user_keycloak_id: Optional[str] = None

        self.eye_reader = SimulatedEyeReader(fs=EYE_FS)

        self.eeg_queue: Queue = Queue()
        self.stop_event = threading.Event()

        self.eeg_worker: Optional[EEGWorker] = None
        self.main_worker: Optional[threading.Thread] = None
        self.system_running = False

        self.latest_eye_features: dict[str, Any] = {}
        self.latest_eeg_features: dict[str, Any] = {}

        self.setup_styles()
        self.build_ui()
        self.restore_saved_keycloak_id()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- UI ----------------
    def setup_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Arial", 20, "bold"))
        style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        style.configure("Body.TLabel", font=("Arial", 11))
        style.configure("Custom.TButton", font=("Arial", 11, "bold"), padding=8)
        style.configure("Custom.TEntry", padding=6)

    def build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=20)
        main.pack(fill="both", expand=True)

        ttk.Label(
            main,
            text="Welcome to the Symbiotic System",
            style="Title.TLabel",
            anchor="center",
        ).pack(pady=(0, 10))

        ttk.Label(
            main,
            text=(
                "This is a simulated prototype version.\n"
                "It uses synthetic EEG and eye-tracking data and prints computed features locally."
            ),
            style="Body.TLabel",
            justify="center",
        ).pack(pady=(0, 15))

        instructions = ttk.LabelFrame(main, text="Instruction Guide", padding=15)
        instructions.pack(fill="x", pady=10)

        instruction_lines = [
            "1. Create your Keycloak account.",
            "2. Enter your Keycloak ID below.",
            "3. Click Save ID.",
            "4. Click Start Simulation.",
            "5. Every 5 seconds, combined features will be printed in the terminal.",
        ]
        for line in instruction_lines:
            ttk.Label(instructions, text=line, style="Body.TLabel").pack(anchor="w", pady=1)

        user_frame = ttk.LabelFrame(main, text="User ID Entry", padding=15)
        user_frame.pack(fill="x", pady=15)

        ttk.Label(
            user_frame,
            text="Enter your Keycloak ID:",
            style="Header.TLabel",
        ).pack(anchor="w", pady=(0, 8))

        self.keycloak_entry = ttk.Entry(user_frame, width=42, style="Custom.TEntry")
        self.keycloak_entry.pack(anchor="w", pady=(0, 10))

        row = ttk.Frame(user_frame)
        row.pack(anchor="w")

        ttk.Button(row, text="Save ID", style="Custom.TButton", command=self.save_keycloak_id).pack(side="left", padx=(0, 10))
        ttk.Button(row, text="Show Stored ID", style="Custom.TButton", command=self.show_stored_id).pack(side="left", padx=(0, 10))
        ttk.Button(row, text="Load Saved ID", style="Custom.TButton", command=self.restore_saved_keycloak_id).pack(side="left")

        control_frame = ttk.LabelFrame(main, text="Simulation Control", padding=15)
        control_frame.pack(fill="x", pady=10)

        ctrl_row = ttk.Frame(control_frame)
        ctrl_row.pack(anchor="w", pady=(0, 10))

        self.start_button = ttk.Button(
            ctrl_row,
            text="Start Simulation",
            style="Custom.TButton",
            command=self.start_system,
        )
        self.start_button.pack(side="left", padx=(0, 10))

        self.stop_button = ttk.Button(
            ctrl_row,
            text="Stop Simulation",
            style="Custom.TButton",
            command=self.stop_system,
        )
        self.stop_button.pack(side="left")

        self.status_label = ttk.Label(control_frame, text="Status: Idle", style="Header.TLabel")
        self.status_label.pack(anchor="w", pady=(5, 5))

        self.id_label = ttk.Label(control_frame, text="Stored Keycloak ID: None", style="Body.TLabel")
        self.id_label.pack(anchor="w")

        output_frame = ttk.LabelFrame(main, text="Application Log", padding=15)
        output_frame.pack(fill="both", expand=True, pady=10)

        self.output_text = tk.Text(output_frame, wrap="word", height=18)
        self.output_text.pack(fill="both", expand=True)

        self.log("Simulation app ready.")

    # ---------------- Logging ----------------
    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        self.output_text.insert("end", line)
        self.output_text.see("end")
        print(message)

    # ---------------- Keycloak ----------------
    def save_keycloak_id(self) -> None:
        entered_id = self.keycloak_entry.get().strip()
        if not entered_id:
            messagebox.showwarning("Input Error", "Please enter a valid Keycloak ID.")
            return

        self.user_keycloak_id = entered_id
        save_json_file(self.keycloak_path, {"user_keycloak_id": entered_id})
        self.id_label.config(text=f"Stored Keycloak ID: {entered_id}")
        self.log(f"Keycloak ID saved locally to {self.keycloak_path.name}")
        messagebox.showinfo("Success", "Keycloak ID saved successfully.")

    def restore_saved_keycloak_id(self) -> None:
        data = load_json_file(self.keycloak_path, {})
        saved_id = str(data.get("user_keycloak_id", "")).strip()

        if not saved_id:
            self.log("No saved Keycloak ID found.")
            return

        self.user_keycloak_id = saved_id
        self.keycloak_entry.delete(0, "end")
        self.keycloak_entry.insert(0, saved_id)
        self.id_label.config(text=f"Stored Keycloak ID: {saved_id}")
        self.log("Saved Keycloak ID loaded.")

    def show_stored_id(self) -> None:
        if not self.user_keycloak_id:
            messagebox.showinfo("Stored ID", "No Keycloak ID stored yet.")
            return
        messagebox.showinfo("Stored ID", f"Current stored Keycloak ID: {self.user_keycloak_id}")

    # ---------------- Start / Stop ----------------
    def start_system(self) -> None:
        if self.system_running:
            messagebox.showinfo("Running", "Simulation is already running.")
            return

        if not self.user_keycloak_id:
            messagebox.showwarning("Missing Keycloak ID", "Please save the Keycloak ID before starting.")
            return

        self.stop_event.clear()
        self.latest_eye_features = {}
        self.latest_eeg_features = {}

        self.eeg_worker = EEGWorker(out_queue=self.eeg_queue, stop_event=self.stop_event)
        self.eeg_worker.start()

        self.main_worker = threading.Thread(target=self.run_combined_loop, daemon=True)
        self.main_worker.start()

        self.system_running = True
        self.status_label.config(text="Status: Running (Simulated)")
        self.log("Simulation started.")

    def stop_system(self) -> None:
        if not self.system_running:
            self.log("Simulation already stopped.")
            return

        self.stop_event.set()
        self.system_running = False
        self.status_label.config(text="Status: Stopped")
        self.log("Simulation stopped.")

    def on_close(self) -> None:
        self.stop_system()
        self.root.destroy()

    # ---------------- Main Loop ----------------
    def run_combined_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                # Eye window
                eye_df = self.eye_reader.collect_eye_dataframe(WINDOW_S)
                eye_features = compute_eye_features_from_df(eye_df, window_s=WINDOW_S)
                self.latest_eye_features = clean_numeric_dict(eye_features)

                # Drain queue and keep latest EEG window
                try:
                    while True:
                        self.latest_eeg_features = self.eeg_queue.get_nowait()
                except Empty:
                    pass

                if not self.latest_eeg_features:
                    self.log("[WARN] Waiting for first EEG window...")
                    time.sleep(0.2)
                    continue

                combined = {
                    "user_keycloak_id": self.user_keycloak_id,
                    "eye_features": self.latest_eye_features,
                    "eeg_features": self.latest_eeg_features,
                }

                print("\n" + "=" * 80)
                print("COMBINED FEATURE WINDOW")
                print(json.dumps(combined, indent=2))
                print("=" * 80 + "\n")

                self.log(
                    f"Printed combined feature window "
                    f"(eye={len(self.latest_eye_features)} keys, "
                    f"eeg={len(self.latest_eeg_features)} keys)"
                )

                # keep the loop close to 5-second cadence
                time.sleep(WINDOW_S)

            except Exception as exc:
                self.log(f"[ERROR] {exc}")
                time.sleep(1.0)


# ============================================================
# ENTRY POINT
# ============================================================
def main() -> None:
    project_dir = Path(__file__).resolve().parent
    root = tk.Tk()
    app = SymbioticSystemApp(root, project_dir=project_dir)
    root.mainloop()


if __name__ == "__main__":
    main()