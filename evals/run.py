"""Evaluate Debrief's RAG pipeline on 10 QMSum meetings, once per retrieval mode.

Usage:
    uv run python -m evals.run --no-judge     # retrieval metrics only: fast, free
    uv run python -m evals.run                # plus DeepEval scores
    uv run python -m evals.run --limit 5      # quick smoke run

For every question it retrieves chunks from that question's meeting, answers
with the app's own prompt and LLM, then scores:
- retrieval, no LLM, against QMSum's human-marked evidence turns:
  hit@3 (a top-3 chunk overlaps the evidence), MRR, and evidence recall
  (share of the evidence text the top 3 chunks cover)
- DeepEval with an LLM judge: faithfulness, answer relevancy and contextual
  relevancy by default (--metrics all adds contextual precision and recall)

Retrieval metrics run for every mode. The judge runs only on --judge-modes
(default: the old vector setup and the final hybrid_rerank), because judge
calls are what eat a free-tier API quota.

Rows are saved to evals/results/<mode>.jsonl after every question, so an
interrupted run (e.g. a daily rate limit) resumes where it stopped. Delete a
mode's .jsonl to re-run it from scratch.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase  # noqa: E402

from evals.judge import DEFAULT_JUDGE, get_judge  # noqa: E402
from logic import llm as llm_config  # noqa: E402
from logic.rag.chunk_embed import SPLITTER  # noqa: E402
from logic.rag.query import llm, prompt  # noqa: E402
from logic.rag.retrieval import MODES, TOP_K, build_index, retrieve  # noqa: E402

EVAL_DIR = Path(__file__).parent
DATA_DIR = EVAL_DIR / "data"
RESULTS_DIR = EVAL_DIR / "results"

ALL_METRICS = {
    "faithfulness": FaithfulnessMetric,
    "answer_relevancy": AnswerRelevancyMetric,
    "contextual_relevancy": ContextualRelevancyMetric,
    "contextual_precision": ContextualPrecisionMetric,
    "contextual_recall": ContextualRecallMetric,
}
DEFAULT_METRICS = ["faithfulness", "answer_relevancy", "contextual_relevancy"]
RETRIEVAL_COLUMNS = ["hit@3", "mrr", "evidence_recall"]


class Meeting:
    """One meeting's chunks, with each chunk's character range in the transcript."""

    def __init__(self, text: str):
        docs = SPLITTER.create_documents([text])
        self.chunks = [d.page_content for d in docs]
        self.ranges = {d.page_content: (d.metadata["start_index"], d.metadata["start_index"] + len(d.page_content)) for d in docs}
        self.index = None

    def get_index(self):
        if self.index is None:
            self.index = build_index(self.chunks)
        return self.index


def overlap(a: tuple[int, int], b: list[int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def retrieval_scores(contexts: list[str], ranges: dict, gold: list[list[int]]) -> dict:
    hits = [any(overlap(ranges[c], g) for g in gold) for c in contexts]
    first = hits.index(True) + 1 if any(hits) else None

    gold_chars = sum(g[1] - g[0] for g in gold)
    covered = set()
    for c in contexts:
        start, end = ranges[c]
        for g in gold:
            covered.update(range(max(start, g[0]), min(end, g[1])))

    return {
        "hit@3": 1.0 if first else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "evidence_recall": round(len(covered) / gold_chars, 3) if gold_chars else 0.0,
    }


def retrieve_case(case: dict, meeting: Meeting, mode: str) -> dict:
    start = time.perf_counter()
    contexts = retrieve(meeting.get_index(), case["question"], mode=mode)
    return {
        "id": case["id"],
        "question": case["question"],
        "retrieved": contexts,
        **retrieval_scores(contexts, meeting.ranges, case["gold_spans"]),
        "retrieval_s": round(time.perf_counter() - start, 3),
    }


def answer(row: dict) -> None:
    """Generate the answer only when it will be judged, so --no-judge needs no LLM."""
    if row.get("answer") is None:
        start = time.perf_counter()
        row["answer"] = llm.invoke(prompt.format(context="\n\n".join(row["retrieved"]), query=row["question"])).content
        row["answer_s"] = round(time.perf_counter() - start, 2)


class DailyLimitReached(Exception):
    """The provider's daily token quota is spent; stop instead of recording errors."""


def judge_case(row: dict, case: dict, metrics: list[str], judge) -> None:
    test_case = LLMTestCase(
        input=case["question"],
        actual_output=row["answer"],
        expected_output=case["expected_answer"],
        retrieval_context=row["retrieved"],
    )
    # Scores from a different judge aren't comparable, so switching judges re-scores the row.
    judge_name = judge.get_model_name()
    if row.get("judge") != judge_name:
        for name in ALL_METRICS:
            row.pop(name, None)
            row.pop(f"{name}_reason", None)
        row["judge"] = judge_name

    for name in metrics:
        if row.get(name) is not None:
            continue
        metric = ALL_METRICS[name](model=judge, async_mode=False, include_reason=True)
        try:
            metric.measure(test_case)
            row[name], row[f"{name}_reason"] = metric.score, metric.reason
        except Exception as e:
            if "tokens per day" in str(e):
                raise DailyLimitReached(str(e)) from e
            # one bad judge response shouldn't sink the run; it's retried next time
            row[name], row[f"{name}_reason"] = None, f"error: {e}"


def run_mode(mode: str, cases: list[dict], meetings: dict, metrics: list[str], judge) -> list[dict]:
    out = RESULTS_DIR / f"{mode}.jsonl"
    rows = {}
    if out.exists():
        rows = {r["id"]: r for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}

    print(f"\n[{mode}] {sum(c['id'] in rows for c in cases)}/{len(cases)} cached")
    for i, case in enumerate(cases, 1):
        row = rows.get(case["id"]) or retrieve_case(case, meetings[case["meeting"]], mode)
        if judge:
            answer(row)
            try:
                judge_case(row, case, metrics, judge)
            except DailyLimitReached:
                rows[case["id"]] = row
                out.write_text("".join(json.dumps(r) + "\n" for r in rows.values()), encoding="utf-8")
                raise SystemExit(f"\nDaily token limit reached at {case['id']}. Progress is saved; run the same command again later.")
        rows[case["id"]] = row
        out.write_text("".join(json.dumps(r) + "\n" for r in rows.values()), encoding="utf-8")
        scores = " ".join(f"{k}={row.get(k)}" for k in ["hit@3", *metrics] if k in row)
        print(f"  {i}/{len(cases)} {case['id']}: {scores}")

    return [rows[c["id"]] for c in cases]


def summarize(rows: list[dict], metrics: list[str]) -> dict:
    summary = {"n": len(rows)}
    for col in RETRIEVAL_COLUMNS + metrics:
        values = [r[col] for r in rows if r.get(col) is not None]
        summary[col] = round(statistics.mean(values), 3) if len(values) == len(rows) else None
    summary["p50_retrieval_ms"] = round(1000 * statistics.median(r["retrieval_s"] for r in rows))
    return summary


def write_summary(summaries: dict, metrics: list[str], n_questions: int, n_meetings: int, judge_name: str) -> str:
    cols = RETRIEVAL_COLUMNS + metrics + ["p50_retrieval_ms"]
    lines = ["| Retrieval | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for mode, s in summaries.items():
        cells = ["–" if s[c] is None else (str(s[c]) if c == "p50_retrieval_ms" else f"{s[c]:.2f}") for c in cols]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")

    table = "\n".join(lines)
    config = (
        f"{n_questions} QMSum questions over {n_meetings} meetings · top-{TOP_K} chunks · "
        f"answer LLM: {llm_config.LLM_PROVIDER} · embeddings: {llm_config.EMBED_PROVIDER} · judge: {judge_name} · "
        "– = not judged"
    )
    (RESULTS_DIR / "summary.md").write_text(f"{table}\n\n{config}\n", encoding="utf-8")
    (RESULTS_DIR / "summary.json").write_text(json.dumps({"config": config, "modes": summaries}, indent=2), encoding="utf-8")
    return f"{table}\n\n{config}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--judge-modes", nargs="+", default=["vector", "hybrid_rerank"], choices=MODES)
    parser.add_argument("--no-judge", action="store_true", help="retrieval metrics only, no judge calls")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, choices=[*ALL_METRICS, "all"])
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help="provider:model, e.g. groq:openai/gpt-oss-120b")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    args = parser.parse_args()

    metrics = list(ALL_METRICS) if "all" in args.metrics else args.metrics
    cases = json.loads((DATA_DIR / "golden.json").read_text(encoding="utf-8"))[: args.limit]
    meetings = {
        m: Meeting((DATA_DIR / "meetings" / f"{m}.txt").read_text(encoding="utf-8"))
        for m in sorted({c["meeting"] for c in cases})
    }
    judge = None if args.no_judge else get_judge(args.judge)

    RESULTS_DIR.mkdir(exist_ok=True)
    summaries = {}
    for mode in args.modes:
        mode_judge = judge if mode in args.judge_modes else None
        summaries[mode] = summarize(run_mode(mode, cases, meetings, metrics, mode_judge), metrics)

    judge_name = judge.get_model_name() if judge else "none"
    print("\n" + write_summary(summaries, metrics, len(cases), len(meetings), judge_name))


if __name__ == "__main__":
    main()
