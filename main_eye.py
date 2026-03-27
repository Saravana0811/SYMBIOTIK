from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from utils.api import userAPI

from CDR.common.baseline.manager import BaselineConfig, BaselineManager
from CDR.common.baseline.storage import BaselinePaths, BaselineStorage
from CDR.eye.features import compute_eye_features_from_reader
from CDR.eye.models.eye_reader import EyeReader


# -----------------------------
# Paths (no os.getcwd)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # demo/
BASELINES_DIR = PROJECT_ROOT / "CDR" / "baselines"
BASELINES_DIR.mkdir(parents=True, exist_ok=True)


_last_no_data_log = 0.0


def warn_no_data(msg: str, every_s: float = 5.0) -> None:
    global _last_no_data_log
    now = time.time()
    if now - _last_no_data_log >= every_s:
        print(f"[EYE][WARN] {msg}")
        _last_no_data_log = now


def load_config(project_root: Path) -> dict[str, Any]:
    cfg_path = project_root / "config.json"
    return json.loads(cfg_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    config = load_config(PROJECT_ROOT)

    api = userAPI(server=config["rl_server"]["host"], port=config["rl_server"]["port"])
    window_s = float(config["frecuency_execution"]["eye"])

    # Create ONE reader and reuse it
    reader = EyeReader()

    # Shared baseline manager (CDR-wide)
    storage = BaselineStorage(BaselinePaths(baselines_dir=BASELINES_DIR))
    baseline_mgr = BaselineManager(
        storage=storage,
        cfg=BaselineConfig(window_s=5.0, scope="current_user_on_this_machine"),
    )

    # Register devices (for now, only eye)
    baseline_devices = {
        "eye": lambda ws: api.clean_cognitive_features(
            compute_eye_features_from_reader(reader, ws)
        ),
        # later:
        # "mouse": lambda ws: ...
        # "eeg": lambda ws: ...
    }

    baseline_payload = baseline_mgr.load()
    print("Eye client started. Baseline loaded:", bool(baseline_payload))
    print("Baseline file:", storage.paths.baseline_json)
    print("Baseline request flag:", storage.paths.request_flag)

    while True:
        # Baseline mode (triggered by UI creating CDR/baselines/baseline.request)
        if baseline_mgr.requested():
            print("Baseline requested. Computing baseline...")

            payload = baseline_mgr.compute_and_save(baseline_devices)
            eye_base = payload.get("devices", {}).get("eye", {}).get("features", {})

            if isinstance(eye_base, dict) and len(eye_base) > 0:
                baseline_mgr.clear_request()
                baseline_payload = baseline_mgr.load()
                print("Baseline saved:", storage.paths.baseline_json)
            else:
                print(
                    "[WARN] Baseline failed: eye features empty. Keeping request file."
                )

        # Normal mode
        eye_features = compute_eye_features_from_reader(reader, window_s)
        eye_features = api.clean_cognitive_features(eye_features)

        if not eye_features:
            warn_no_data("No valid eye features computed (skipping RL send).")
            continue

        res = api.send_cognitive_features_to_symbiotik(
            cognitive_features=eye_features,
            device="eye",
        )
        print("Eye features -> RL:", res)
