"""One-off: transcribe the sample earnings call so evals run on a fixed transcript.

Usage: uv run python -m evals.transcribe EarningsCall.wav
"""
import json
import sys
from pathlib import Path

from faster_whisper import WhisperModel

OUT_DIR = Path(__file__).parent / "data"


def main(audio_path: str) -> None:
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, language="en", beam_size=5, vad_filter=True)
    segs = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "earnings_call_segments.json").write_text(json.dumps(segs, indent=1))
    (OUT_DIR / "earnings_call_transcript.txt").write_text(" ".join(s["text"] for s in segs))
    print(f"{len(segs)} segments, {segs[-1]['end'] / 60:.1f} min")


if __name__ == "__main__":
    main(sys.argv[1])
