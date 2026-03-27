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

from eeg_feature_extractor import EEGFeatureExtractor5s
from fixation_detection_2d import FixationDetector2D
from eye_analyser import EyeAnalyser


APP_TITLE = "Symbiotic System"
KEYCLOAK_FILE = "user_keycloak.json"

WINDOW_S = 5.0
BASELINE_S = 15.0

EEG_FS = 250.0
EEG_CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
EYE_FS = 60.0


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
        relative_fix_times = raw_fix["TIME"].to_numpy(dtype=float) - float(raw_fix["TIME"].iloc[0])
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

    relative_times = raw_data["TIME"].to_numpy(dtype=float) - float(raw_data["TIME"].iloc[0])

    pup_features = analyser._pupil_features_by_time_percentiles(
        times=relative_times,
        values=vals,
        max_trial_duration=float(window_s),
    )

    return {**fix_features, **pup_features}


class SimulatedEEGStreamer:
    def __init__(self, fs: float = EEG_FS, channels: list[str] | None = None) -> None:
        self.fs = float(fs)
        self.channels = channels or EEG_CHANNELS
        self.dt = 1.0 / self.fs
        self._sample_index = 0
        self._phase = {ch: np.random.uniform(0, 2 * np.pi) for ch in self.channels}
        self._lock = threading.Lock()

    def next_sample(self) -> tuple[float, dict[str, float]]:
        with self._lock:
            t_sec = self._sample_index * self.dt
            values: dict[str, float] = {}

            for ch in self.channels:
                theta = 7.0 * np.sin(2 * np.pi * 5.5 * t_sec + self._phase[ch])
                alpha = 10.0 * np.sin(2 * np.pi * 10.0 * t_sec + 0.5 * self._phase[ch])

                if ch in {"Pz", "PO7", "Oz", "PO8"}:
                    alpha *= 1.6
                if ch in {"Fz", "C3", "Cz", "C4"}:
                    theta *= 1.3

                drift = 2.0 * np.sin(2 * np.pi * 0.2 * t_sec + 0.2 * self._phase[ch])
                noise = np.random.normal(0.0, 3.0)

                values[ch] = float(theta + alpha + drift + noise)

            self._sample_index += 1
            return t_sec, values


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

    def collect_eye_dataframe(self, duration_seconds: float, baseline_mode: bool = False) -> pd.DataFrame:
        n = max(2, int(round(self.fs * duration_seconds)))
        rows: list[dict[str, float | int]] = []

        i = 0
        fix_start_t = self.global_time

        while i < n:
            if baseline_mode:
                next_target = np.array([0.5, 0.5], dtype=float)
                fix_len = min(n - i, int(self.fs * 1.0))
                sacc_len = 0
            else:
                fix_len = int(np.random.randint(max(3, int(0.12 * self.fs)), max(4, int(0.45 * self.fs))))
                fix_len = min(fix_len, n - i)
                next_target = self._new_fixation_target()
                sacc_len = int(np.random.randint(max(1, int(0.02 * self.fs)), max(2, int(0.05 * self.fs))))
                sacc_len = min(sacc_len, n - i)

            for _ in range(fix_len):
                t = self.global_time

                if baseline_mode:
                    gaze_xy = np.array([0.5, 0.5]) + np.random.normal(0.0, 0.003, size=2)
                else:
                    gaze_xy = self.current_fix + np.random.normal(0.0, 0.008, size=2)

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
                if i >= n:
                    break

            if i >= n:
                break

            if not baseline_mode:
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

        return pd.DataFrame(rows)


class EEGWorker(threading.Thread):
    def __init__(self, out_queue: Queue, stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.streamer = SimulatedEEGStreamer(fs=EEG_FS, channels=EEG_CHANNELS)
        self.extractor = EEGFeatureExtractor5s(fs=EEG_FS, window_s=WINDOW_S)
        self.sample_sleep = 1.0 / EEG_FS

        self.baseline_collecting = False
        self.baseline_lock = threading.Lock()
        self.baseline_samples: list[tuple[float, dict[str, float]]] = []

    def start_baseline_buffer(self) -> None:
        with self.baseline_lock:
            self.baseline_samples = []
            self.baseline_collecting = True

    def stop_baseline_buffer(self) -> list[tuple[float, dict[str, float]]]:
        with self.baseline_lock:
            self.baseline_collecting = False
            samples = list(self.baseline_samples)
            self.baseline_samples = []
            return samples

    def run(self) -> None:
        while not self.stop_event.is_set():
            t_sec, values = self.streamer.next_sample()

            with self.baseline_lock:
                if self.baseline_collecting:
                    self.baseline_samples.append((t_sec, dict(values)))

            win = self.extractor.push(t_sec, values)
            if win is not None:
                eeg_payload = {
                    "start_t_sec": win.start_t_sec,
                    "end_t_sec": win.end_t_sec,
                    "bad_window": win.bad_window,
                    "qc_reason": win.qc_reason,
                    **win.features,
                }
                self.out_queue.put(clean_numeric_dict(eeg_payload))

            time.sleep(self.sample_sleep)


def compute_eeg_features_from_samples(
    samples: list[tuple[float, dict[str, float]]],
    fs: float = EEG_FS,
    window_s: float = WINDOW_S,
) -> dict[str, Any]:
    extractor = EEGFeatureExtractor5s(fs=fs, window_s=window_s)

    for t_sec, values in samples:
        extractor.push(t_sec, values)

    data = pd.DataFrame([v for _, v in samples])
    if data.empty:
        return {
            "bad_window": True,
            "qc_reason": "no_samples",
        }

    result: dict[str, Any] = {
        "bad_window": False,
        "qc_reason": "",
        "n_samples": len(data),
    }

    frontal = ["Fz", "C3", "Cz", "C4"]
    posterior = ["Pz", "PO7", "Oz", "PO8"]

    for ch in data.columns:
        x = data[ch].to_numpy(dtype=float)
        result[f"{ch}_var"] = float(np.var(x))
        result[f"{ch}_ptp"] = float(np.ptp(x))

    def band_proxy(x: np.ndarray, low: float, high: float, fs: float) -> float:
        fft_vals = np.fft.rfft(x - np.mean(x))
        freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return 0.0
        power = np.sum(np.abs(fft_vals[mask]) ** 2)
        return float(power / max(1, np.sum(mask)))

    theta_vals = []
    alpha_vals = []

    for ch in data.columns:
        x = data[ch].to_numpy(dtype=float)
        theta = band_proxy(x, 4.0, 7.0, fs)
        alpha = band_proxy(x, 8.0, 12.0, fs)
        result[f"{ch}_theta_power"] = theta
        result[f"{ch}_alpha_power"] = alpha

        if ch in frontal:
            theta_vals.append(theta)
        if ch in posterior:
            alpha_vals.append(alpha)

    result["frontal_theta_mean"] = float(np.mean(theta_vals)) if theta_vals else 0.0
    result["parietal_alpha_mean"] = float(np.mean(alpha_vals)) if alpha_vals else 0.0

    return clean_numeric_dict(result)


class SymbioticSystemApp:
    def __init__(self, root: tk.Tk, project_dir: Path) -> None:
        self.root = root
        self.project_dir = project_dir
        self.root.title(APP_TITLE)
        self.root.geometry("900x760")
        self.root.minsize(780, 650)

        self.keycloak_path = self.project_dir / KEYCLOAK_FILE
        self.user_keycloak_id: Optional[str] = None

        self.eye_reader = SimulatedEyeReader(fs=EYE_FS)
        self.eeg_queue: Queue = Queue()
        self.stop_event = threading.Event()

        self.eeg_worker: Optional[EEGWorker] = None
        self.main_worker: Optional[threading.Thread] = None

        self.system_running = False
        self.baseline_done = False
        self.baseline_running = False
        self.normal_loop_enabled = False

        self.latest_eye_features: dict[str, Any] = {}
        self.latest_eeg_features: dict[str, Any] = {}

        self.setup_styles()
        self.build_ui()
        self.restore_saved_keycloak_id()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

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
        ).pack(pady=(0, 10))

        ttk.Label(
            main,
            text=(
                "Experimental simulated prototype\n"
                "Flow: Enter Keycloak ID -> Start Session -> Start Baseline -> Normal 5-second output"
            ),
            style="Body.TLabel",
            justify="center",
        ).pack(pady=(0, 15))

        instructions = ttk.LabelFrame(main, text="Instruction Guide", padding=15)
        instructions.pack(fill="x", pady=10)

        lines = [
            "1. Enter your Keycloak ID and save it.",
            "2. Click Start Session.",
            "3. Click Start Baseline.",
            "4. A fixation cross '+' will appear for 15 seconds.",
            "5. Baseline features will be printed once.",
            "6. After that, normal 5-second output continues automatically.",
        ]
        for line in lines:
            ttk.Label(instructions, text=line, style="Body.TLabel").pack(anchor="w", pady=1)

        user_frame = ttk.LabelFrame(main, text="User ID Entry", padding=15)
        user_frame.pack(fill="x", pady=15)

        ttk.Label(user_frame, text="Enter your Keycloak ID:", style="Header.TLabel").pack(anchor="w", pady=(0, 8))

        self.keycloak_entry = ttk.Entry(user_frame, width=42, style="Custom.TEntry")
        self.keycloak_entry.pack(anchor="w", pady=(0, 10))

        row = ttk.Frame(user_frame)
        row.pack(anchor="w")

        ttk.Button(row, text="Save ID", style="Custom.TButton", command=self.save_keycloak_id).pack(side="left", padx=(0, 10))
        ttk.Button(row, text="Show Stored ID", style="Custom.TButton", command=self.show_stored_id).pack(side="left", padx=(0, 10))
        ttk.Button(row, text="Load Saved ID", style="Custom.TButton", command=self.restore_saved_keycloak_id).pack(side="left")

        control_frame = ttk.LabelFrame(main, text="Experiment Control", padding=15)
        control_frame.pack(fill="x", pady=10)

        ctrl_row = ttk.Frame(control_frame)
        ctrl_row.pack(anchor="w", pady=(0, 10))

        self.start_session_button = ttk.Button(
            ctrl_row,
            text="Start Session",
            style="Custom.TButton",
            command=self.start_session,
        )
        self.start_session_button.pack(side="left", padx=(0, 10))

        self.start_baseline_button = ttk.Button(
            ctrl_row,
            text="Start Baseline",
            style="Custom.TButton",
            command=self.start_baseline,
        )
        self.start_baseline_button.pack(side="left", padx=(0, 10))

        self.stop_button = ttk.Button(
            ctrl_row,
            text="Stop Session",
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

        self.log("Application ready.")

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        self.output_text.insert("end", line)
        self.output_text.see("end")
        print(message)

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

    def start_session(self) -> None:
        if self.system_running:
            messagebox.showinfo("Running", "Session is already running.")
            return

        if not self.user_keycloak_id:
            messagebox.showwarning("Missing Keycloak ID", "Please save the Keycloak ID before starting.")
            return

        self.stop_event.clear()
        self.baseline_done = False
        self.baseline_running = False
        self.normal_loop_enabled = False
        self.latest_eye_features = {}
        self.latest_eeg_features = {}

        self.eeg_worker = EEGWorker(out_queue=self.eeg_queue, stop_event=self.stop_event)
        self.eeg_worker.start()

        self.main_worker = threading.Thread(target=self.run_normal_loop, daemon=True)
        self.main_worker.start()

        self.system_running = True
        self.status_label.config(text="Status: Session started, waiting for baseline")
        self.log("Session started. Simulated EEG and eye data are running in background.")

    def start_baseline(self) -> None:
        if not self.system_running:
            messagebox.showwarning("Session Not Started", "Please click Start Session first.")
            return

        if self.baseline_running:
            messagebox.showinfo("Baseline", "Baseline is already running.")
            return

        if self.baseline_done:
            messagebox.showinfo("Baseline", "Baseline is already completed.")
            return

        self.baseline_running = True
        self.normal_loop_enabled = False
        self.status_label.config(text="Status: Baseline running")
        self.log("Baseline started.")

        if self.eeg_worker is not None:
            self.eeg_worker.start_baseline_buffer()

        baseline_window = tk.Toplevel(self.root)
        baseline_window.title("Baseline")
        baseline_window.geometry("600x400")
        baseline_window.configure(bg="white")
        baseline_window.attributes("-topmost", True)

        label = tk.Label(
            baseline_window,
            text="+",
            font=("Arial", 48, "bold"),
            bg="white",
            fg="black",
        )
        label.place(relx=0.5, rely=0.5, anchor="center")

        countdown_label = tk.Label(
            baseline_window,
            text=f"Baseline: {int(BASELINE_S)} s",
            font=("Arial", 14),
            bg="white",
            fg="black",
        )
        countdown_label.place(relx=0.5, rely=0.65, anchor="center")

        start_time = time.time()

        def update_countdown() -> None:
            elapsed = time.time() - start_time
            remaining = max(0, int(math.ceil(BASELINE_S - elapsed)))
            countdown_label.config(text=f"Baseline: {remaining} s")

            if elapsed >= BASELINE_S:
                baseline_window.destroy()
                self.finish_baseline()
            else:
                baseline_window.after(200, update_countdown)

        update_countdown()

    def finish_baseline(self) -> None:
        self.log("Baseline window closed. Computing 15-second baseline features...")

        eye_df = self.eye_reader.collect_eye_dataframe(BASELINE_S, baseline_mode=True)
        eye_features = compute_eye_features_from_df(eye_df, window_s=BASELINE_S)

        eeg_samples: list[tuple[float, dict[str, float]]] = []
        if self.eeg_worker is not None:
            eeg_samples = self.eeg_worker.stop_baseline_buffer()

        eeg_features = compute_eeg_features_from_samples(
            eeg_samples,
            fs=EEG_FS,
            window_s=BASELINE_S,
        )

        baseline_dict = {
            "user_keycloak_id": self.user_keycloak_id,
            "baseline": True,
            "eye_features": clean_numeric_dict(eye_features),
            "eeg_features": clean_numeric_dict(eeg_features),
        }

        print("\n" + "=" * 80)
        print("BASELINE FEATURE WINDOW")
        print(json.dumps(baseline_dict, indent=2))
        print("=" * 80 + "\n")

        self.log("Baseline features printed.")
        self.baseline_done = True
        self.baseline_running = False
        self.normal_loop_enabled = True
        self.status_label.config(text="Status: Baseline finished, normal 5-second output running")

    def stop_system(self) -> None:
        if not self.system_running:
            self.log("Session already stopped.")
            return

        self.stop_event.set()
        self.system_running = False
        self.baseline_running = False
        self.normal_loop_enabled = False
        self.status_label.config(text="Status: Stopped")
        self.log("Session stopped.")

    def on_close(self) -> None:
        self.stop_system()
        self.root.destroy()

    def run_normal_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                if not self.normal_loop_enabled:
                    time.sleep(0.2)
                    continue

                eye_df = self.eye_reader.collect_eye_dataframe(WINDOW_S, baseline_mode=False)
                eye_features = compute_eye_features_from_df(eye_df, window_s=WINDOW_S)
                self.latest_eye_features = clean_numeric_dict(eye_features)

                try:
                    while True:
                        self.latest_eeg_features = self.eeg_queue.get_nowait()
                except Empty:
                    pass

                if not self.latest_eeg_features:
                    self.log("[WARN] Waiting for EEG 5-second feature window...")
                    time.sleep(0.2)
                    continue

                combined = {
                    "user_keycloak_id": self.user_keycloak_id,
                    "baseline": False,
                    "eye_features": self.latest_eye_features,
                    "eeg_features": self.latest_eeg_features,
                }

                print("\n" + "=" * 80)
                print("NORMAL FEATURE WINDOW")
                print(json.dumps(combined, indent=2))
                print("=" * 80 + "\n")

                self.log(
                    f"Printed normal feature window "
                    f"(eye={len(self.latest_eye_features)} keys, "
                    f"eeg={len(self.latest_eeg_features)} keys)"
                )

                time.sleep(WINDOW_S)

            except Exception as exc:
                self.log(f"[ERROR] {exc}")
                time.sleep(1.0)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    root = tk.Tk()
    app = SymbioticSystemApp(root, project_dir=project_dir)
    root.mainloop()


if __name__ == "__main__":
    main()