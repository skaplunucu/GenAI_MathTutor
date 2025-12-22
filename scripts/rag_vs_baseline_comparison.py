#!/usr/bin/env python3
"""
RAG vs Baseline Comparison Script

Systematically compares RAG answers against baseline (no retrieval)
to measure the improvement from retrieval-augmented generation.
"""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np

# Test questions for comparison
COMPARISON_QUESTIONS = [
    "Що таке об'єм кулі?",
    "Як знайти площу круга?",
    "Що таке призма?",
    "Як обчислити площу трикутника?",
    "Поясни властивості правильної піраміди"
]


def calculate_factual_accuracy(answer: str, ground_truth_keywords: List[str]) -> float:
    """
    Check if answer contains expected factual keywords.

    Args:
        answer: Generated answer text
        ground_truth_keywords: Expected keywords/phrases

    Returns:
        Ratio of found keywords (0-1)
    """
    answer_lower = answer.lower()
    found = sum(1 for kw in ground_truth_keywords if kw.lower() in answer_lower)
    return found / len(ground_truth_keywords) if ground_truth_keywords else 0.0


def check_hallucination(answer: str, retrieved_chunks: List[str]) -> float:
    """
    Estimate hallucination by checking if answer contains info not in chunks.

    Simple heuristic: Check if key mathematical terms in answer appear in chunks.

    Returns:
        Grounding score (0-1): 1 = fully grounded, 0 = likely hallucinated
    """
    # Extract mathematical formulas and key terms from answer
    # This is a simplified version - in production, use more sophisticated methods

    if not retrieved_chunks:
        return 0.0  # No context = can't be grounded

    combined_context = " ".join(retrieved_chunks).lower()

    # Count how many sentences in answer have support in context
    sentences = answer.split('.')
    grounded = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check if sentence keywords appear in context
        words = sentence.lower().split()
        # Filter out common words
        content_words = [w for w in words if len(w) > 3 and w.isalpha()]

        if content_words:
            overlap = sum(1 for w in content_words if w in combined_context)
            if overlap / len(content_words) > 0.3:  # 30% overlap threshold
                grounded += 1

    return grounded / len(sentences) if sentences else 0.0


def calculate_citation_quality(answer: str, num_citations: int) -> float:
    """
    Evaluate citation quality in answer.

    Returns:
        Citation score (0-1)
    """
    # Check if answer references sources
    has_source_mentions = any(marker in answer for marker in ['Джерело', '[Джерело', 'стор.', 'с.'])

    if not has_source_mentions:
        return 0.0

    # Count actual source references
    source_count = answer.count('Джерело')

    # Normalize by expected citations
    return min(source_count / max(num_citations, 1), 1.0)


def compare_rag_vs_baseline(
    baseline_answer: str,
    rag_answer: str,
    retrieved_chunks: List[str],
    citations: List[str],
    ground_truth_keywords: List[str] = None
) -> Dict[str, float]:
    """
    Compare RAG answer quality against baseline.

    Returns:
        Dictionary of comparison metrics
    """

    # 1. Citation presence
    baseline_has_citations = calculate_citation_quality(baseline_answer, 0)
    rag_has_citations = calculate_citation_quality(rag_answer, len(citations))

    # 2. Grounding (anti-hallucination)
    baseline_grounding = 0.0  # No retrieval = no grounding
    rag_grounding = check_hallucination(rag_answer, retrieved_chunks)

    # 3. Factual accuracy (if ground truth provided)
    if ground_truth_keywords:
        baseline_accuracy = calculate_factual_accuracy(baseline_answer, ground_truth_keywords)
        rag_accuracy = calculate_factual_accuracy(rag_answer, ground_truth_keywords)
    else:
        baseline_accuracy = None
        rag_accuracy = None

    # 4. Answer length (completeness proxy)
    baseline_length = len(baseline_answer)
    rag_length = len(rag_answer)

    # 5. Calculate improvement
    improvements = {
        'citation_improvement': rag_has_citations - baseline_has_citations,
        'grounding_improvement': rag_grounding - baseline_grounding,
        'length_ratio': rag_length / baseline_length if baseline_length > 0 else 0,
    }

    if baseline_accuracy is not None:
        improvements['accuracy_improvement'] = rag_accuracy - baseline_accuracy

    return {
        'baseline': {
            'citations': baseline_has_citations,
            'grounding': baseline_grounding,
            'accuracy': baseline_accuracy,
            'length': baseline_length
        },
        'rag': {
            'citations': rag_has_citations,
            'grounding': rag_grounding,
            'accuracy': rag_accuracy,
            'length': rag_length
        },
        'improvements': improvements
    }


# Example ground truth keywords for each question
GROUND_TRUTH = {
    "Що таке об'єм кулі?": [
        "об'єм", "куля", "формула", "4/3", "πr³", "радіус"
    ],
    "Як знайти площу круга?": [
        "площа", "круг", "формула", "πr²", "радіус"
    ],
    "Що таке призма?": [
        "призма", "основа", "бічні ребра", "грань", "багатокутник"
    ],
    "Як обчислити площу трикутника?": [
        "площа", "трикутник", "основа", "висота", "формула"
    ],
    "Поясни властивості правильної піраміди": [
        "піраміда", "правильна", "основа", "бічні ребра", "висота"
    ]
}


if __name__ == "__main__":
    print("RAG vs Baseline Comparison Framework")
    print("=" * 80)
    print("\nThis script provides functions to measure RAG improvement.")
    print("\nKey Metrics:")
    print("  1. Citation Quality - Are sources referenced?")
    print("  2. Grounding Score - Is answer based on retrieved content?")
    print("  3. Factual Accuracy - Does it contain expected information?")
    print("  4. Length/Completeness - How comprehensive is the answer?")
    print("\n" + "=" * 80)
    print("\nUsage:")
    print("  from rag_vs_baseline_comparison import compare_rag_vs_baseline")
    print("\n  results = compare_rag_vs_baseline(")
    print("      baseline_answer=baseline_response,")
    print("      rag_answer=rag_response.answer,")
    print("      retrieved_chunks=[c.text for c in rag_response.retrieved_chunks],")
    print("      citations=rag_response.citations,")
    print("      ground_truth_keywords=GROUND_TRUTH[question]")
    print("  )")
    print("\n  print(f'Citation Improvement: {results[\"improvements\"][\"citation_improvement\"]:.2%}')")
    print("  print(f'Grounding Improvement: {results[\"improvements\"][\"grounding_improvement\"]:.2%}')")
