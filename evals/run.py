"""Evaluate Debrief's RAG pipeline on the sample earnings call, once per retrieval mode.

Usage:
    uv run python -m evals.run                      # all modes, all questions
    uv run python -m evals.run --modes vector       # one mode
    uv run python -m evals.run --limit 5            # quick smoke run

For every question it retrieves chunks, answers with the app's own prompt and
LLM, then scores:
- retrieval (no LLM): hit@3 and MRR, using the evidence quote in golden.json
- DeepEval (Groq judge): faithfulness, answer relevancy, contextual
  precision, contextual recall, contextual relevancy

Per-question rows are appended to evals/results/<mode>.jsonl as they finish,
so a run that hits Groq's free-tier daily limit resumes where it stopped.
Delete a mode's .jsonl to re-run it from scratch.
"""
import argparse
import json
import os
import re
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

from evals.judge import DEFAULT_JUDGE, GroqJudge  # noqa: E402
from logic import llm as llm_config  # noqa: E402
from logic.rag.chunk_embed import chunk_transcript  # noqa: E402
from logic.rag.query import llm, prompt  # noqa: E402
from logic.rag.retrieval import MODES, TOP_K, build_index, retrieve  # noqa: E402

EVAL_DIR = Path(__file__).parent
DATA_DIR = EVAL_DIR / "data"
RESULTS_DIR = EVAL_DIR / "results"

METRICS = {
    "faithfulness": FaithfulnessMetric,
    "answer_relevancy": AnswerRelevancyMetric,
    "contextual_precision": ContextualPrecisionMetric,
    "contextual_recall": ContextualRecallMetric,
    "contextual_relevancy": ContextualRelevancyMetric,
}
COLUMNS = ["hit@3", "mrr", *METRICS, "latency_s"]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9$%.\s]", " ", text.lower())).strip()


def evidence_rank(evidence: str, chunks: list[str]) -> int | None:
    """1-based rank of the first retrieved chunk containing the evidence quote."""
    needle = normalize(evidence)
    for rank, chunk in enumerate(chunks, start=1):
        if needle in normalize(chunk):
            return rank
    return None


def score_case(case: dict, index, mode: str, judge) -> dict:
    start = time.perf_counter()
    contexts = retrieve(index, case["question"], mode=mode)
    answer = llm.invoke(prompt.format(context="\n\n".join(contexts), query=case["question"])).content
    latency = time.perf_counter() - start

    rank = evidence_rank(case["evidence"], contexts)
    row = {
        "id": case["id"],
        "question": case["question"],
        "answer": answer,
        "retrieved": contexts,
        "hit@3": 1.0 if rank else 0.0,
        "mrr": 1.0 / rank if rank else 0.0,
        "latency_s": round(latency, 2),
    }

    test_case = LLMTestCase(
        input=case["question"],
        actual_output=answer,
        expected_output=case["expected_answer"],
        retrieval_context=contexts,
    )
    for name, metric_cls in METRICS.items():
        metric = metric_cls(model=judge, async_mode=False, include_reason=True)
        try:
            metric.measure(test_case)
            row[name], row[f"{name}_reason"] = metric.score, metric.reason
        except Exception as e:  # one bad judge response shouldn't sink the whole run
            row[name], row[f"{name}_reason"] = None, f"error: {e}"
    return row


def run_mode(mode: str, cases: list[dict], chunks: list[str], judge) -> list[dict]:
    out = RESULTS_DIR / f"{mode}.jsonl"
    done = {}
    if out.exists():
        done = {r["id"]: r for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}

    todo = [c for c in cases if c["id"] not in done]
    print(f"\n[{mode}] {len(done)} cached, {len(todo)} to run")
    if todo:
        index = build_index(chunks)
        with out.open("a", encoding="utf-8") as f:
            for i, case in enumerate(todo, 1):
                row = score_case(case, index, mode, judge)
                done[row["id"]] = row
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"  {i}/{len(todo)} {case['id']}: faithfulness={row['faithfulness']} hit@3={row['hit@3']}")

    ids = {c["id"] for c in cases}
    return [r for r in done.values() if r["id"] in ids]


def summarize(rows: list[dict]) -> dict:
    summary = {"n": len(rows)}
    for col in COLUMNS:
        values = [r[col] for r in rows if r.get(col) is not None]
        if col == "latency_s":
            summary[col] = round(statistics.median(values), 2) if values else None
        else:
            summary[col] = round(statistics.mean(values), 3) if values else None
    return summary


def write_summary(summaries: dict, judge_name: str) -> str:
    header = "| Retrieval | " + " | ".join(COLUMNS[:-1]) + " | p50 latency (s) |"
    lines = [header, "|" + "---|" * (len(COLUMNS) + 1)]
    for mode, s in summaries.items():
        cells = [f"{s[c]:.2f}" if s[c] is not None else "n/a" for c in COLUMNS]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")

    table = "\n".join(lines)
    config = (
        f"Questions: {next(iter(summaries.values()))['n']} · top-k: {TOP_K} · "
        f"answer LLM: {llm_config.LLM_PROVIDER} · embeddings: {llm_config.EMBED_PROVIDER} · "
        f"judge: {judge_name}"
    )
    (RESULTS_DIR / "summary.md").write_text(f"{table}\n\n{config}\n", encoding="utf-8")
    (RESULTS_DIR / "summary.json").write_text(json.dumps({"config": config, "modes": summaries}, indent=2), encoding="utf-8")
    return f"{table}\n\n{config}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    args = parser.parse_args()

    transcript = (DATA_DIR / "earnings_call_transcript.txt").read_text(encoding="utf-8")
    cases = json.loads((DATA_DIR / "golden.json").read_text(encoding="utf-8"))[: args.limit]
    chunks = chunk_transcript(transcript)
    judge = GroqJudge(args.judge_model)

    RESULTS_DIR.mkdir(exist_ok=True)
    summaries = {mode: summarize(run_mode(mode, cases, chunks, judge)) for mode in args.modes}
    print("\n" + write_summary(summaries, judge.get_model_name()))


if __name__ == "__main__":
    main()
