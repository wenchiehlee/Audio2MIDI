import os
import glob
import torch
import warnings
import subprocess
import shutil
from pathlib import Path
from mido import MidiFile, MidiTrack, MetaMessage

# Filter warnings to keep output clean
warnings.filterwarnings("ignore")

def setup_directories():
    base_output = Path("Midi")
    bytedance_dir = base_output / "ByteDance"
    basicpitch_dir = base_output / "BasicPitch"
    splits_bytedance_dir = base_output / "Splits" / "ByteDance"
    splits_basicpitch_dir = base_output / "Splits" / "BasicPitch"
    
    bytedance_dir.mkdir(parents=True, exist_ok=True)
    basicpitch_dir.mkdir(parents=True, exist_ok=True)
    splits_bytedance_dir.mkdir(parents=True, exist_ok=True)
    splits_basicpitch_dir.mkdir(parents=True, exist_ok=True)
    
    return bytedance_dir, basicpitch_dir, splits_bytedance_dir, splits_basicpitch_dir

def ensure_wav_for_audio(path):
    candidate_path = Path(path)
    if candidate_path.suffix.lower() not in (".mp3", ".m4a"):
        return candidate_path

    converted_dir = Path("Audio") / "_converted"
    converted_dir.mkdir(parents=True, exist_ok=True)
    converted_path = converted_dir / f"{candidate_path.stem}.wav"

    if (not converted_path.exists() or
            converted_path.stat().st_mtime < candidate_path.stat().st_mtime):
        try:
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
        except Exception:
            if shutil.which("ffmpeg") is None:
                raise
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(candidate_path),
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "44100",
                    "-ac",
                    "1",
                    str(converted_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    return converted_path

def split_midi_two_hands(input_path, simple_output_path, smart_output_path, split_note=60):
    midi = MidiFile(input_path)
    events = []
    abs_time = 0
    idx = 0
    for track in midi.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            events.append((abs_time, idx, msg))
            idx += 1

    note_pitches = [
        msg.note for (_, _, msg) in events
        if msg.type == "note_on" and msg.velocity > 0
    ]

    def kmeans_1d(values, max_iter=20):
        if not values:
            return None
        c1 = min(values)
        c2 = max(values)
        if c1 == c2:
            return (c1, c2)
        for _ in range(max_iter):
            cluster1 = []
            cluster2 = []
            for v in values:
                if abs(v - c1) <= abs(v - c2):
                    cluster1.append(v)
                else:
                    cluster2.append(v)
            new_c1 = sum(cluster1) / len(cluster1) if cluster1 else c1
            new_c2 = sum(cluster2) / len(cluster2) if cluster2 else c2
            if abs(new_c1 - c1) < 1e-6 and abs(new_c2 - c2) < 1e-6:
                break
            c1, c2 = new_c1, new_c2
        return tuple(sorted((c1, c2)))

    smart_centroids = kmeans_1d(note_pitches)

    def assign_hand(note, method):
        if method == "simple" or smart_centroids is None:
            return "RH" if note >= split_note else "LH"
        c1, c2 = smart_centroids
        return "RH" if abs(note - c2) < abs(note - c1) else "LH"

    def build_tracks(method, output_path):
        meta_events = []
        rh_events = []
        lh_events = []
        active = {}

        for time, order, msg in sorted(events, key=lambda item: (item[0], item[1])):
            if msg.type in ("note_on", "note_off"):
                is_on = msg.type == "note_on" and msg.velocity > 0
                key = (msg.channel, msg.note)
                if is_on:
                    hand = assign_hand(msg.note, method)
                    active[key] = hand
                else:
                    hand = active.pop(key, assign_hand(msg.note, method))
                target = rh_events if hand == "RH" else lh_events
                target.append((time, order, msg))
            else:
                meta_events.append((time, order, msg))

        def build_track(event_list):
            track = MidiTrack()
            last_time = 0
            for time, _, msg in sorted(event_list, key=lambda item: (item[0], item[1])):
                delta = time - last_time
                last_time = time
                track.append(msg.copy(time=delta))
            if not track or track[-1].type != "end_of_track":
                track.append(MetaMessage("end_of_track", time=0))
            return track

        out_midi = MidiFile(ticks_per_beat=midi.ticks_per_beat)
        out_midi.tracks.append(build_track(meta_events))
        out_midi.tracks.append(build_track(rh_events))
        out_midi.tracks.append(build_track(lh_events))
        out_midi.save(output_path)

    build_tracks("simple", simple_output_path)
    build_tracks("smart", smart_output_path)

def transcribe_bytedance(audio_path, output_dir):
    try:
        from piano_transcription_inference import PianoTranscription, sample_rate, load_audio

        def load_audio_fallback(path, sr):
            try:
                (audio, _) = load_audio(path, sr=sr, mono=True)
                return audio
            except Exception:
                pass

            candidate_path = ensure_wav_for_audio(path)

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

        splits_base = Path("Midi") / "Splits" / "ByteDance"
        splits_base.mkdir(parents=True, exist_ok=True)
        split_simple = splits_base / f"{Path(midi_filename).stem}_simple.mid"
        split_smart = splits_base / f"{Path(midi_filename).stem}_smart.mid"
        split_midi_two_hands(midi_path, split_simple, split_smart)
        print(f"  [Split] Saved to: {split_simple}")
        print(f"  [Split] Saved to: {split_smart}")
        
    except Exception as e:
        print(f"  [ByteDance] Error: {e}")

def transcribe_basic_pitch(audio_path, output_dir):
    try:
        try:
            from basic_pitch.inference import predict_and_save, ICASSP_2022_MODEL_PATH
        except ImportError:
            print("  [BasicPitch] Library not installed. Skipping.")
            return

        print(f"  [BasicPitch] Processing: {os.path.basename(audio_path)}")
        input_path = ensure_wav_for_audio(audio_path)
        
        # Output path - basic_pitch automatically adds _basic_pitch.mid, so we just give the dir
        # However, to keep names clean, we might need to rename, but let's stick to default for now
        
        # predict_and_save takes a list of input paths
        predict_and_save(
            [str(input_path)],
            str(output_dir),
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )
        print(f"  [BasicPitch] Saved to: {output_dir}")

        midi_filename = os.path.splitext(os.path.basename(audio_path))[0] + "_basic_pitch.mid"
        midi_path = output_dir / midi_filename
        if midi_path.exists():
            splits_base = Path("Midi") / "Splits" / "BasicPitch"
            splits_base.mkdir(parents=True, exist_ok=True)
            split_simple = splits_base / f"{Path(midi_filename).stem}_simple.mid"
            split_smart = splits_base / f"{Path(midi_filename).stem}_smart.mid"
            split_midi_two_hands(midi_path, split_simple, split_smart)
            print(f"  [Split] Saved to: {split_simple}")
            print(f"  [Split] Saved to: {split_smart}")
        
    except Exception as e:
        print(f"  [BasicPitch] Error: {e}")

def main():
    print("=== Audio to MIDI Converter ===")
    print("Libraries: ByteDance (piano_transcription_inference) & Spotify (basic-pitch)")
    
    bytedance_dir, basicpitch_dir, _, _ = setup_directories()
    
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
