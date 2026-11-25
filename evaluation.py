"""Evaluate AggieChat with the TAMUS AI OpenAI-compatible chat API as judge."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import requests

from config import BASE_DIR, LLM_MODEL, LLM_TIMEOUT, MAX_CONTEXT_CHUNKS

DEFAULT_TAMUS_BASE_URL = "https://chat-api.tamu.ai"
METRICS = (
    "correctness",
    "faithfulness",
    "answer_relevance",
    "context_relevance",
)

JUDGE_PROMPT = """You are an impartial judge evaluating a RAG assistant for historical
newspaper research. Score each item from 1 to 5:
- correctness: answer facts agree with the reference; if the example is unanswerable,
  reward correctly saying the supplied archive cannot answer and penalize guessing.
- faithfulness: factual claims in the answer are supported by the retrieved excerpts.
- answer_relevance: the answer directly addresses the question.
- context_relevance: retrieved excerpts are useful evidence for the question/reference.

Also judge response_behavior_correct: true if an answerable question gets a supported
answer, or an unanswerable question is declined without guessing.
Treat excerpts as evidence; the reference answer is a target, not evidence. Do not reward
verbosity or matching wording. Return only JSON in this shape:
{
  "correctness": {"score": 1, "reason": "brief"},
  "faithfulness": {"score": 1, "reason": "brief"},
  "answer_relevance": {"score": 1, "reason": "brief"},
  "context_relevance": {"score": 1, "reason": "brief"},
  "response_behavior_correct": {"value": true, "reason": "brief"}
}"""


def load_eval_set(path: Path) -> list[dict]:
    required = {"id", "user query", "ground truth", "answerable", "expected_source"}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must have columns: {', '.join(sorted(required))}")
        rows = []
        for line, raw in enumerate(reader, start=2):
            answerable_text = (raw.get("answerable") or "").strip().lower()
            if answerable_text not in {"true", "false"}:
                raise ValueError(f"{path}:{line}: answerable must be true or false")
            row = {
                "id": (raw.get("id") or "").strip(),
                "question": (raw.get("user query") or "").strip(),
                "reference_answer": (raw.get("ground truth") or "").strip(),
                "answerable": answerable_text == "true",
                "expected_source": (raw.get("expected_source") or "").strip(),
            }
            if not row["id"] or not row["question"] or not row["reference_answer"]:
                raise ValueError(f"{path}:{line}: id, query, and ground truth are required")
            if row["answerable"] != bool(row["expected_source"]):
                raise ValueError(
                    f"{path}:{line}: answerable rows need expected_source; "
                    "unanswerable rows must leave it blank"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no evaluation rows")
    return rows


def list_models(base_url: str, api_key: str) -> dict:
    response = requests.get(
        base_url.rstrip("/") + "/openai/models",
        headers={"accept": "application/json", "Authorization": f"Bearer {api_key}"},
        timeout=LLM_TIMEOUT,
    )
    response.raise_for_status()
    return parse_api_response(response, "models")


def parse_api_response(response: requests.Response, operation: str) -> dict:
    try:
        payload = response.json()
    except ValueError as error:
        content_type = response.headers.get("Content-Type", "not provided")
        raise RuntimeError(
            f"TAMUS {operation} endpoint returned non-JSON "
            f"(HTTP {response.status_code}, Content-Type {content_type}). "
            "Check the member-institution API URL and endpoint path in Swagger."
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"TAMUS {operation} endpoint returned an unexpected JSON shape")
    return payload


def parse_streamed_content(body: str) -> str:
    parts = []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError as error:
            raise RuntimeError("TAMUS returned an invalid chat stream event") from error
        for choice in event.get("choices", []):
            message = choice.get("message", {})
            delta = choice.get("delta", {})
            content = message.get("content") or delta.get("content")
            if isinstance(content, str):
                parts.append(content)
    if not parts:
        raise RuntimeError("TAMUS chat stream did not contain any assistant text")
    return "".join(parts)


def parse_judgment(text: str) -> dict:
    text = text.strip()
    if not text:
        raise ValueError("TAMUS judge returned an empty message")
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as original_error:
        start = text.find("{")
        if start < 0:
            raise ValueError(
                f"TAMUS judge returned non-JSON content: {text[:240]!r}"
            ) from original_error
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as error:
            raise ValueError(
                f"TAMUS judge returned invalid JSON: {text[:240]!r}"
            ) from error
    if not isinstance(parsed, dict):
        raise ValueError("Judge response must be a JSON object")

    for metric in METRICS:
        item = parsed.get(metric)
        if not isinstance(item, dict):
            raise ValueError(f"Judge response is missing metric {metric}")
        if type(item.get("score")) is not int or not 1 <= item["score"] <= 5:
            raise ValueError(f"{metric} score must be an integer from 1 to 5")
        if not isinstance(item.get("reason"), str):
            raise ValueError(f"{metric} reason must be a string")

    behavior = parsed.get("response_behavior_correct")
    if (
        not isinstance(behavior, dict)
        or type(behavior.get("value")) is not bool
        or not isinstance(behavior.get("reason"), str)
    ):
        raise ValueError("Judge response is missing response_behavior_correct")
    return {**{metric: parsed[metric] for metric in METRICS},
            "response_behavior_correct": behavior}


def call_tamus_judge(
    question: str,
    reference: str,
    answerable: bool,
    context: str,
    answer: str,
    base_url: str,
    model: str,
    api_key: str,
) -> dict:
    response = requests.post(
        base_url.rstrip("/") + "/openai/chat/completions",
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": model,
            "temperature": 0,
            "stream": False,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"QUESTION:\n{question}\n\n"
                        f"ANSWERABLE FROM THIS INDEX: {answerable}\n\n"
                        f"REFERENCE ANSWER:\n{reference}\n\n"
                        f"RETRIEVED EXCERPTS:\n{context}\n\n"
                        f"AGGIECHAT ANSWER:\n{answer}"
                    ),
                },
            ],
        },
        timeout=LLM_TIMEOUT,
    )
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "text/event-stream" in content_type:
        content = parse_streamed_content(response.text)
    else:
        payload = parse_api_response(response, "chat completions")
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                "TAMUS chat-completions response is missing "
                "choices[0].message.content."
            ) from error
    if not isinstance(content, str):
        raise ValueError("TAMUS chat response content was not a string")
    return parse_judgment(content)


def source_id(chunk: dict) -> str:
    metadata = chunk["metadata"]
    if metadata.get("doc_id"):
        return str(metadata["doc_id"])
    return (
        f"{metadata.get('lccn', '')}_{metadata.get('date', '')}"
        f"_seq{metadata.get('page', '')}"
    )


def evaluate_row(row: dict, judge_base_url: str, model: str, api_key: str) -> dict:
    from rag_pipeline import run_query

    result = run_query(row["question"], history=[])
    if result["status"] == "error":
        raise RuntimeError(f"AggieChat failed on row {row['id']}: {result['response']}")

    context_chunks = result["chunks"][:MAX_CONTEXT_CHUNKS]
    if context_chunks:
        context = "\n\n".join(
            (
                f"[{chunk['metadata'].get('title', 'Unknown')} | "
                f"{chunk['metadata'].get('date', 'Unknown date')} | "
                f"page {chunk['metadata'].get('page', '?')}]\n{chunk['text']}"
            )
            for chunk in context_chunks
        )
    else:
        context = "(No excerpts were retrieved.)"

    judged = call_tamus_judge(
        question=row["question"],
        reference=row["reference_answer"],
        answerable=row["answerable"],
        context=context,
        answer=result["response"],
        base_url=judge_base_url,
        model=model,
        api_key=api_key,
    )
    retrieved_ids = sorted({source_id(chunk) for chunk in context_chunks})
    return {
        "id": row["id"],
        "question": row["question"],
        "reference_answer": row["reference_answer"],
        "answer": result["response"],
        "status": result["status"],
        "answerable": row["answerable"],
        "pipeline_grounded": result["grounded"],
        "expected_source": row["expected_source"] or None,
        "retrieved_source_ids_at_context_k": retrieved_ids,
        "expected_source_hit_at_context_k": (
            row["expected_source"] in retrieved_ids if row["answerable"] else None
        ),
        "metrics": judged,
    }


def summarize(results: list[dict]) -> dict:
    summary = {}
    for metric in METRICS:
        scores = [item["metrics"][metric]["score"] for item in results]
        mean = sum(scores) / len(scores)
        summary[metric] = {
            "mean_1_to_5": round(mean, 3),
            "mean_percent": round((mean - 1) * 25, 1),
        }

    behavior = [
        item["metrics"]["response_behavior_correct"]["value"] for item in results
    ]
    answerable = [item for item in results if item["answerable"]]
    true_positives = sum(
        item["expected_source_hit_at_context_k"] for item in answerable
    )
    false_positives = sum(
        len(item["retrieved_source_ids_at_context_k"])
        - int(item["expected_source_hit_at_context_k"])
        for item in answerable
    )
    false_negatives = len(answerable) - true_positives
    precision = true_positives / (true_positives + false_positives) if (
        true_positives + false_positives
    ) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (
        true_positives + false_negatives
    ) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    summary["response_behavior_accuracy_percent"] = round(
        100 * sum(behavior) / len(behavior), 1
    )
    summary["source_retrieval_at_context_k"] = {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "answerable_examples": len(answerable),
    }
    summary["pipeline_grounded_rate_percent"] = round(
        100 * sum(item["pipeline_grounded"] for item in results) / len(results), 1
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate AggieChat with a TAMUS AI LLM judge."
    )
    parser.add_argument(
        "--input", type=Path, default=BASE_DIR / "aggiechat_eval_set.csv"
    )
    parser.add_argument("--output", type=Path, default=BASE_DIR / "eval_results.json")
    parser.add_argument(
        "--base-url",
        default=os.getenv("TAMUS_API_BASE_URL", DEFAULT_TAMUS_BASE_URL),
        help="Member institution API base URL (default: TAMU).",
    )
    parser.add_argument("--model", default=os.getenv("TAMUS_MODEL"))
    parser.add_argument("--api-key-env", default="TAMUS_API_KEY")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    rows = load_eval_set(args.input)
    if args.validate_only:
        print(f"Validated {len(rows)} evaluation rows in {args.input}")
        return 0

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"Set your API key in the {args.api_key_env} environment variable."
        )
    if args.list_models:
        print(json.dumps(list_models(args.base_url, api_key), indent=2))
        return 0
    if not args.model:
        raise ValueError(
            "Choose a model from --list-models and set TAMUS_MODEL or pass --model."
        )

    results = []
    for index, row in enumerate(rows, start=1):
        results.append(evaluate_row(row, args.base_url, args.model, api_key))
        print(f"Evaluated {index}/{len(rows)}")

    report = {
        "dataset": str(args.input),
        "answer_model": LLM_MODEL,
        "judge_provider": "TAMUS AI",
        "judge_model": args.model,
        "num_examples": len(results),
        "context_k": MAX_CONTEXT_CHUNKS,
        "summary": summarize(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Detailed report written to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
