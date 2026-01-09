import os
import glob
import torch
import warnings
import subprocess
from pathlib import Path

# Filter warnings to keep output clean
warnings.filterwarnings("ignore")

def setup_directories():
    base_output = Path("Midi")
    bytedance_dir = base_output / "ByteDance"
    basicpitch_dir = base_output / "BasicPitch"
    
    bytedance_dir.mkdir(parents=True, exist_ok=True)
    basicpitch_dir.mkdir(parents=True, exist_ok=True)
    
    return bytedance_dir, basicpitch_dir

def transcribe_bytedance(audio_path, output_dir):
    try:
        from piano_transcription_inference import PianoTranscription, sample_rate, load_audio

        def load_audio_fallback(path, sr):
            try:
                (audio, _) = load_audio(path, sr=sr, mono=True)
                return audio
            except Exception:
                pass

            candidate_path = Path(path)
            if candidate_path.suffix.lower() in (".mp3", ".m4a"):
                converted_dir = Path("Audio") / "_converted"
                converted_dir.mkdir(parents=True, exist_ok=True)
                converted_path = converted_dir / f"{candidate_path.stem}.wav"
                if (not converted_path.exists() or
                        converted_path.stat().st_mtime < candidate_path.stat().st_mtime):
                    subprocess.run(
                        [
                            "afconvert",
                            "-f", "WAVE",
                            "-d", "LEI16",
                            str(candidate_path),
                            str(converted_path),
                        ],
                        check=True,
                    )
                candidate_path = converted_path

            import librosa
            audio, _ = librosa.load(str(candidate_path), sr=sr, mono=True)
            return audio
        
        print(f"  [ByteDance] Processing: {os.path.basename(audio_path)}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize transcriptor
        transcriptor = PianoTranscription(device=device, checkpoint_path=None)
        
        # Load audio with a fallback that uses afconvert on macOS for MP3/M4A.
        audio = load_audio_fallback(audio_path, sample_rate)
        
        # Output path
        midi_filename = os.path.splitext(os.path.basename(audio_path))[0] + ".mid"
        midi_path = output_dir / midi_filename
        
        # Transcribe
        transcriptor.transcribe(audio, str(midi_path))
        print(f"  [ByteDance] Saved to: {midi_path}")
        
    except Exception as e:
        print(f"  [ByteDance] Error: {e}")

def transcribe_basic_pitch(audio_path, output_dir):
    try:
        try:
            from basic_pitch.inference import predict_and_save
        except ImportError:
            print("  [BasicPitch] Library not installed. Skipping.")
            return

        print(f"  [BasicPitch] Processing: {os.path.basename(audio_path)}")
        
        # Output path - basic_pitch automatically adds _basic_pitch.mid, so we just give the dir
        # However, to keep names clean, we might need to rename, but let's stick to default for now
        
        # predict_and_save takes a list of input paths
        predict_and_save(
            [str(audio_path)],
            str(output_dir),
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False
        )
        print(f"  [BasicPitch] Saved to: {output_dir}")
        
    except Exception as e:
        print(f"  [BasicPitch] Error: {e}")

def main():
    print("=== Audio to MIDI Converter ===")
    print("Libraries: ByteDance (piano_transcription_inference) & Spotify (basic-pitch)")
    
    bytedance_dir, basicpitch_dir = setup_directories()
    
    # Extensions to look for
    extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
    audio_files = []
    
    for ext in extensions:
        audio_files.extend(glob.glob(os.path.join("Audio", ext)))
    
    if not audio_files:
        print("\nNo audio files found in 'Audio/' folder.")
        print("Please place your piano recordings (.wav, .mp3) there and run this script again.")
        return

    print(f"\nFound {len(audio_files)} audio file(s). Starting transcription...\n")

    for audio_file in audio_files:
        print(f"--- Converting: {audio_file} ---")
        transcribe_bytedance(audio_file, bytedance_dir)
        transcribe_basic_pitch(audio_file, basicpitch_dir)
        print("")

    print("All tasks completed.")

if __name__ == "__main__":
    main()
