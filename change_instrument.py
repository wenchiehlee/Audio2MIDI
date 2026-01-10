import argparse
from pathlib import Path

from mido import MidiFile, MidiTrack, Message


def parse_args():
    parser = argparse.ArgumentParser(
        description="Change MIDI instrument program (default: Alto Sax)."
    )
    parser.add_argument("input", help="Input MIDI file path")
    parser.add_argument(
        "-o",
        "--output",
        help="Output MIDI file path (default: <input>_sax.mid)",
    )
    parser.add_argument(
        "--program",
        type=int,
        default=65,
        help="MIDI program number 0-127 (default: 65 = Alto Sax)",
    )
    return parser.parse_args()


def change_program(input_path, output_path, program):
    midi = MidiFile(input_path)
    out = MidiFile(ticks_per_beat=midi.ticks_per_beat)

    for track in midi.tracks:
        channels_used = {
            msg.channel
            for msg in track
            if msg.type in ("note_on", "note_off") and hasattr(msg, "channel")
        }
        new_track = MidiTrack()

        if channels_used:
            for channel in sorted(channels_used):
                new_track.append(
                    Message("program_change", program=program, channel=channel, time=0)
                )

        for msg in track:
            if msg.type == "program_change":
                new_track.append(msg.copy(program=program))
            else:
                new_track.append(msg)

        out.tracks.append(new_track)

    out.save(output_path)


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_sax.mid")
    )
    change_program(input_path, output_path, args.program)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
