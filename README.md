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
*   Python 3.10 or newer (Python 3.13 is supported for ByteDance, but Basic Pitch may have compatibility issues).
*   FFmpeg (optional, but recommended for processing various audio formats).

### Installation

1.  Clone this repository.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you have issues installing `basic-pitch` on Python 3.13, the script will still work with just the ByteDance model.*

## Usage

1.  **Place Audio Files:**
    Put your `.wav`, `.mp3`, `.flac`, or `.m4a` files into the `Audio/` folder.

2.  **Run the Script:**
    ```bash
    python main.py
    ```

3.  **Find MIDI Output:**
    *   ByteDance results will be in `Midi/ByteDance/`.
    *   Basic Pitch results will be in `Midi/BasicPitch/`.

## Moving to Another Machine (e.g., Mac Mini)

1.  Copy this entire folder to your Mac.
2.  Run the installation and usage steps above.
3.  **Note on Model Weights:** The ByteDance model (~165MB) is downloaded to your user home directory (`~/piano_transcription_inference_data`) upon the first run. 
    *   If you want to manually move the model to avoid re-downloading, copy `note_F1=0.9677_pedal_F1=0.9186.pth` to that location on your Mac.
