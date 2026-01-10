# Audio to MIDI Converter

This project converts piano audio recordings into MIDI files using state-of-the-art deep learning models.

## Libraries Used

1.  **ByteDance Piano Transcription** (`piano_transcription_inference`):
    *   **Best for:** Solo piano recordings.
    *   **Features:** High-accuracy note detection, velocity estimation, and pedal transcription.
    *   **Model:** Uses a CRNN model (~165MB) which is downloaded automatically on the first run.

2.  **Spotify Basic Pitch** (`basic-pitch`) [Optional]:
    *   **Best for:** General polyphonic instrument transcription.
    *   **Status:** Optional in this script. Skips if not installed.

## Setup

### Prerequisites
*   Python 3.10 or newer (ByteDance works on newer versions; Basic Pitch needs Python 3.11).
*   FFmpeg (optional). On macOS, the script can fall back to `afconvert` for MP3/M4A.

### Installation

1.  Clone this repository.
2.  Install the required dependencies for ByteDance:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you only need ByteDance, you can skip Basic Pitch.*

3.  (Optional) Install Basic Pitch in a Python 3.11 venv:
    ```bash
    brew install python@3.11
    /opt/homebrew/bin/python3.11 -m venv .venv-basicpitch
    .venv-basicpitch/bin/pip install --upgrade pip
    .venv-basicpitch/bin/pip install basic-pitch torch piano_transcription_inference
    ```

## Usage

1.  **Place Audio Files:**
    Put your `.wav`, `.mp3`, `.flac`, or `.m4a` files into the `Audio/` folder.

2.  **Run the Script (ByteDance only):**
    ```bash
    python main.py
    ```

3.  **Run the Script (ByteDance + Basic Pitch in venv):**
    ```bash
    TMPDIR="$(pwd)/.tmp" MPLCONFIGDIR="$(pwd)/.tmp" .venv-basicpitch/bin/python main.py
    ```

4.  **Find MIDI Output:**
    *   ByteDance results will be in `Midi/ByteDance/`.
    *   Basic Pitch results will be in `Midi/BasicPitch/`.
    *   Two-hand splits are in `Midi/Splits/ByteDance/` and `Midi/Splits/BasicPitch/`.
        *   `_simple.mid` uses a middle-C split (MIDI 60).
        *   `_smart.mid` uses a 2-cluster pitch split.

## MuseScore Notes (Playback vs. MIDI)

MuseScore imports MIDI by converting it into notation (quantizing timing and
guessing voices/hands). That means MuseScore playback can sound different from
a raw MIDI player.

Tips to minimize differences:
*   Use a very short minimum note value (e.g., 1/64 or 1/128) in the MIDI
    import dialog.
*   Disable any "simplify" or "reduce ties" options.
*   Avoid auto hand-splitting on import; split hands manually later.
*   Keep the original MIDI for playback verification in a DAW or media player.

## Moving to Another Machine (e.g., Mac Mini)

1.  Copy this entire folder to your Mac.
2.  Run the installation and usage steps above.
3.  **Note on Model Weights:** The ByteDance model (~165MB) is downloaded to your user home directory (`~/piano_transcription_inference_data`) upon the first run. 
    *   If you want to manually move the model to avoid re-downloading, copy `note_F1=0.9677_pedal_F1=0.9186.pth` to that location on your Mac.
