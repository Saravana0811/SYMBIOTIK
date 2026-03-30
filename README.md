# Compact Data Representation (CDR)

The CDR module is part of the [SYMBIOTIK](https://symbiotik.2025.2i2.eu/) project. Its purpose is to collect data from physiological sensors at the user's premises, extract features, package them into compact representations, and send them to the SYMBIOTIK system for processing.

## Prerequisites

- Python 3.10+
- [Unicorn Hybrid Black](https://www.unicorn-bi.com/) 8-channel EEG device
- [GazePoint GP3](https://www.gazept.com/product/gp3hd/) eye-tracking device
- Unicorn Suite (drivers and streaming software for the EEG device)
- GazePoint Analysis software (drivers and streaming software for the eye tracker)
- A SYMBIOTIK Dashboard account (see [Usage](#usage) below)

## Project Structure

```
app.py                  # Main entry point (GUI + experiment flow)
config.json             # User configuration (Keycloak ID)
processing/             # Feature extraction modules
  eeg_feature_extractor.py
  eye_analyser.py
  fixation_detection_2d.py
streamers/              # Hardware interfaces
  eye_reader.py         # GazePoint eye tracker
  unicorn_streamer.py   # Unicorn EEG TCP streamer
```

## Installation

```bash
git clone <repository-url>
cd SYMBIOTIK
pip install -r requirements.txt
```

## Usage

1. Make sure the EEG and eye-tracking devices are connected and their respective software is running.
2. Start the CDR application:
   ```bash
   python app.py
   ```
3. A GUI window will appear with further instructions.
4. Log in to the SYMBIOTIK Dashboard at `https://xxxx.eu/fvt` and create an account if you don't have one.
5. Copy your **User ID** from the top-right corner of the Dashboard.
6. Paste the User ID into the GUI and click **Save ID**.
7. Click **Start Session**.
8. Click **Start Baseline**. A popup window with a fixation cross (`+`) will appear. Look at the cross for 15 seconds as indicated by the countdown.
9. After the baseline completes, the CDR will continuously collect features from both sensors and send them to the SYMBIOTIK system.
10. Go back to the Dashboard, select one of the 6 predefined view rooms, and interact with the widgets to answer the presented questions.
11. If everything is set up correctly, the widgets' UI will adapt in real time based on the cognitive load measured from the sensors.
