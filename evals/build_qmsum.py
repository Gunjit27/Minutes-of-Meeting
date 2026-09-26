"""Build the eval set from QMSum (https://github.com/Yale-LILY/QMSum, MIT license).

QMSum pairs real hour-long meetings (here: AMI product-design meetings) with
human-written questions, answers, and the transcript turns that support each
answer. Those turns become gold character spans, so retrieval can be scored
without an LLM.

Usage (once, output is committed):
    git clone --depth 1 https://github.com/Yale-LILY/QMSum.git /tmp/QMSum
    uv run python -m evals.build_qmsum /tmp/QMSum
"""
import json
import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
MEETINGS_DIR = DATA_DIR / "meetings"

N_MEETINGS = 10
QUESTIONS_PER_MEETING = 3
MAX_SPAN_TURNS = 60  # skip "summarize the whole hour" questions; top-3 chunks can't cover them

NOISE = re.compile(r"\{\w+\}")  # {vocalsound}, {disfmarker}, {gap} ...


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", NOISE.sub(" ", text)).strip()


def build_meeting(turns: list[dict]) -> tuple[str, list[tuple[int, int] | None]]:
    """Join turns as 'Speaker: text' lines; return text and each turn's char range."""
    lines, offsets, pos = [], [], 0
    for turn in turns:
        content = clean(turn["content"])
        if not content:
            offsets.append(None)
            continue
        line = f"{turn['speaker']}: {content}"
        offsets.append((pos, pos + len(line)))
        lines.append(line)
        pos += len(line) + 1
    return "\n".join(lines), offsets


def gold_spans(spans: list[list[str]], offsets) -> list[list[int]]:
    out = []
    for start, end in spans:
        kept = [o for o in offsets[int(start): int(end) + 1] if o]
        if kept:
            out.append([kept[0][0], kept[-1][1]])
    return out


def main(qmsum_dir: str) -> None:
    src = Path(qmsum_dir) / "data" / "Product" / "test"
    MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
    golden = []

    for path in sorted(src.glob("*.json"))[:N_MEETINGS]:
        meeting = json.loads(path.read_text(encoding="utf-8"))
        text, offsets = build_meeting(meeting["meeting_transcripts"])
        (MEETINGS_DIR / f"{path.stem}.txt").write_text(text, encoding="utf-8")

        def span_turns(q):
            return sum(int(b) - int(a) + 1 for a, b in q["relevant_text_span"])

        focused = [q for q in meeting["specific_query_list"] if span_turns(q) <= MAX_SPAN_TURNS]
        for i, q in enumerate(focused[:QUESTIONS_PER_MEETING]):
            golden.append({
                "id": f"{path.stem}-{i}",
                "meeting": path.stem,
                "question": q["query"],
                "expected_answer": q["answer"],
                "gold_spans": gold_spans(q["relevant_text_span"], offsets),
            })

    (DATA_DIR / "golden.json").write_text(json.dumps(golden, indent=1), encoding="utf-8")
    print(f"{len(golden)} questions over {N_MEETINGS} meetings")


if __name__ == "__main__":
    main(sys.argv[1])
