
# ============================================================
# PROJECT 01 — AI DOCUMENT INTELLIGENCE
# STEP 08 — TESTING & EVALUATION
# PART 04 — RAG EVALUATION & PERFORMANCE
#
# FILE: evaluation/evaluate_rag.py
#
# PURPOSE:
# 1. Evaluate retrieval quality using Hit@K.
# 2. Check expected information in RAG answers.
# 3. Measure end-to-end response latency.
# 4. Save evaluation results to JSON.
#
# REQUIREMENTS:
# - FastAPI server must be running.
# - Ollama must be available.
# - Test documents must already be indexed.
# ============================================================

import json
import time
from pathlib import Path

import requests


# ============================================================
# STEP 1 — Configuration
# ============================================================

API_URL = "http://127.0.0.1:8000"

TOP_K = 3

TIMEOUT_SECONDS = 120

OUTPUT_FILE = (
    Path(__file__).resolve().parent
    / "evaluation_results.json"
)


# ============================================================
# STEP 2 — Evaluation dataset
# ============================================================

EVALUATION_DATASET = [
    {
        "id": "Q01",
        "question": (
            "How many days of annual leave "
            "do employees receive?"
        ),
        "expected_answer": "21",
        "expected_source": "employee_handbook.docx",
    },
    {
        "id": "Q02",
        "question": (
            "Who is the manager of Project Falcon?"
        ),
        "expected_answer": "Sarah Johnson",
        "expected_source": "project_test.txt",
    },
    {
        "id": "Q03",
        "question": (
            "Who is the manager of Project Phoenix?"
        ),
        "expected_answer": "Michael Brown",
        "expected_source": "streamlit_test.txt",
    },
]


# ============================================================
# STEP 3 — Evaluate one question
# ============================================================

def evaluate_question(test_case):

    question = test_case["question"]

    start_time = time.perf_counter()

    response = requests.post(
        f"{API_URL}/ask",
        json={
            "question": question,
            "top_k": TOP_K,
        },
        timeout=TIMEOUT_SECONDS,
    )

    latency = time.perf_counter() - start_time

    response.raise_for_status()

    result = response.json()

    answer = result.get("answer", "")

    sources = result.get("sources", [])

    # Check whether expected source was retrieved.
    retrieved_source_names = [
        str(source.get("source", ""))
        for source in sources
    ]

    retrieval_hit = any(
        source_name == test_case["expected_source"]
        for source_name in retrieved_source_names
    )

    # Simple exact-information check.
    answer_correct = (
        test_case["expected_answer"].casefold()
        in answer.casefold()
    )

    return {
        "id": test_case["id"],
        "question": question,
        "expected_answer": test_case["expected_answer"],
        "actual_answer": answer,
        "expected_source": test_case["expected_source"],
        "retrieved_sources": retrieved_source_names,
        "retrieval_hit": retrieval_hit,
        "answer_correct": answer_correct,
        "latency_seconds": round(latency, 3),
    }


# ============================================================
# STEP 4 — Evaluate complete dataset
# ============================================================

def run_evaluation():

    print("=" * 60)
    print("AI DOCUMENT INTELLIGENCE — RAG EVALUATION")
    print("=" * 60)

    results = []

    for test_case in EVALUATION_DATASET:

        print(f"\nEvaluating: {test_case['id']}")
        print(f"Question: {test_case['question']}")

        try:
            result = evaluate_question(test_case)

            results.append(result)

            print(
                "Retrieval Hit:",
                "PASS" if result["retrieval_hit"] else "FAIL",
            )

            print(
                "Answer Check:",
                "PASS" if result["answer_correct"] else "FAIL",
            )

            print(
                "Latency:",
                result["latency_seconds"],
                "seconds",
            )

            print("Answer:", result["actual_answer"])

        except requests.RequestException as error:

            print("REQUEST FAILED:", error)

            results.append({
                "id": test_case["id"],
                "question": test_case["question"],
                "error": str(error),
                "retrieval_hit": False,
                "answer_correct": False,
                "latency_seconds": None,
            })

    # ========================================================
    # STEP 5 — Calculate evaluation metrics
    # ========================================================

    total_questions = len(results)

    retrieval_hits = sum(
        result["retrieval_hit"]
        for result in results
    )

    correct_answers = sum(
        result["answer_correct"]
        for result in results
    )

    successful_latencies = [
        result["latency_seconds"]
        for result in results
        if result["latency_seconds"] is not None
    ]

    retrieval_hit_rate = (
        retrieval_hits / total_questions
        if total_questions else 0
    )

    answer_accuracy = (
        correct_answers / total_questions
        if total_questions else 0
    )

    average_latency = (
        sum(successful_latencies)
        / len(successful_latencies)
        if successful_latencies else None
    )

    summary = {
        "total_questions": total_questions,
        "retrieval_hit_at_k": round(
            retrieval_hit_rate, 4
        ),
        "answer_accuracy": round(
            answer_accuracy, 4
        ),
        "average_latency_seconds": (
            round(average_latency, 3)
            if average_latency is not None
            else None
        ),
        "top_k": TOP_K,
        "results": results,
    }

    # ========================================================
    # STEP 6 — Save results
    # ========================================================

    OUTPUT_FILE.write_text(
        json.dumps(
            summary,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)

    print("Total Questions:", total_questions)

    print(
        f"Retrieval Hit@{TOP_K}:",
        f"{retrieval_hit_rate * 100:.2f}%",
    )

    print(
        "Answer Accuracy:",
        f"{answer_accuracy * 100:.2f}%",
    )

    if average_latency is not None:
        print(
            "Average Latency:",
            round(average_latency, 3),
            "seconds",
        )

    print("\nResults saved to:", OUTPUT_FILE)


# ============================================================
# STEP 7 — Run evaluation
# ============================================================

if __name__ == "__main__":
    run_evaluation()
