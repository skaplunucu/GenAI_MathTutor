"""
Unified evaluation functions for Math Tutor experiments.

Provides consistent metric calculations across all experiments:
- Base score (70%): language, completeness, structure
- Experiment bonus (30%): retrieval, tools, or agent-specific metrics
"""

from typing import Dict, Any, List
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# Load test questions from evaluation dataset
def load_evaluation_dataset(dataset_path: str = '../evaluation/evaluation_dataset.jsonl'):
    """Load questions from evaluation dataset."""
    import json
    from pathlib import Path

    path = Path(dataset_path)
    if not path.exists():
        # Try alternative path
        path = Path(__file__).parent.parent / 'evaluation' / 'evaluation_dataset.jsonl'

    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")

    questions = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if entry['type'] == 'task':  # Only tasks, not quizzes
                questions.append({
                    'input': entry['input'],
                    'difficulty': entry['difficulty'],
                    'expected_answer': entry['expected_answer'],
                    'generated_answer': entry.get('generated_answer', ''),
                })

    return questions


# Load standard and tool questions from dataset
try:
    _all_questions = load_evaluation_dataset()
    # Use all 30 tasks for comprehensive evaluation (was 15)
    STANDARD_TEST_QUESTIONS = [q['input'] for q in _all_questions]  # All tasks
    TOOL_TEST_QUESTIONS = [q['input'] for q in _all_questions if 'об\'єм' in q['input'].lower() or 'площа' in q['input'].lower()]  # All geometry tasks
    EVALUATION_DATASET = _all_questions
except:
    # Fallback to hardcoded questions if dataset not available
    STANDARD_TEST_QUESTIONS = [
        "Згенеруй задачу про об'єм кулі з розв'язанням",
        "Створи задачу про площу трикутника з рішенням",
        "Придумай задачу про логарифми з детальним розв'язанням",
        "Згенеруй задачу про показникові рівняння",
        "Створи задачу про площу циліндра з розв'язком",
        "Придумай задачу про похідну функції",
        "Згенеруй задачу про теорему Піфагора",
        "Створи задачу про об'єм піраміди з рішенням",
    ]
    TOOL_TEST_QUESTIONS = [
        "Згенеруй задачу про об'єм кулі з конкретними числами та розв'язанням",
        "Створи задачу про площу трикутника з перевіркою обчислень",
        "Придумай задачу про логарифми з числовим прикладом",
        "Згенеруй задачу про показникове рівняння з розв'язком",
        "Створи задачу про площу циліндра з конкретними вимірами",
    ]
    EVALUATION_DATASET = []


def calculate_ukrainian_ratio(text: str) -> float:
    """Calculate proportion of Ukrainian (Cyrillic) characters."""
    if not text:
        return 0.0
    cyrillic = sum(1 for c in text if 0x0400 <= ord(c) <= 0x052F)
    alpha = sum(1 for c in text if c.isalpha())
    return cyrillic / alpha if alpha > 0 else 0.0


def has_structure(text: str) -> bool:
    """Check if answer has task/solution/answer structure."""
    text_lower = text.lower()
    has_task = any(word in text_lower for word in ['задача', 'умова'])
    has_solution = any(word in text_lower for word in ['розв\'язання', 'розв\'язок', 'рішення'])
    has_answer = 'відповідь' in text_lower
    return has_task and has_solution and has_answer


def has_citations(text: str) -> bool:
    """Check if answer references textbook sources."""
    return any(word in text.lower() for word in ['джерело', 'підручник', 'за формулою'])


def calculate_completeness(answer_length: int, target_length: int = 400) -> float:
    """Evaluate answer completeness based on length."""
    return min(answer_length / target_length, 1.0)


def calculate_answer_correctness(generated_answer: str, expected_answer: str) -> float:
    """
    Calculate correctness by comparing generated answer with expected answer.

    Uses fuzzy matching to handle formatting differences:
    - Exact match: 1.0
    - Expected value in generated text: 1.0
    - Numerical proximity (within 10%): 0.8
    - Partial match: 0.5
    - No match: 0.0
    """
    import re

    if not generated_answer or not expected_answer:
        return 0.0

    # Normalize strings
    gen_lower = generated_answer.lower().strip()
    exp_lower = expected_answer.lower().strip()

    # Exact match
    if gen_lower == exp_lower:
        return 1.0

    # Expected answer appears in generated text
    if exp_lower in gen_lower:
        return 1.0

    # Try to extract and compare numbers
    try:
        # Extract numbers from both
        gen_numbers = re.findall(r'\d+[.,]?\d*', generated_answer)
        exp_numbers = re.findall(r'\d+[.,]?\d*', expected_answer)

        if gen_numbers and exp_numbers:
            # Convert to float for comparison
            gen_float = float(gen_numbers[0].replace(',', '.'))
            exp_float = float(exp_numbers[0].replace(',', '.'))

            # Check if within 10% tolerance
            if abs(gen_float - exp_float) / max(abs(exp_float), 1) < 0.1:
                return 0.8
    except:
        pass

    # Partial match (expected words in generated)
    exp_words = set(exp_lower.split())
    gen_words = set(gen_lower.split())
    if exp_words & gen_words:
        overlap = len(exp_words & gen_words) / len(exp_words)
        if overlap > 0.5:
            return 0.5

    return 0.0


WEIGHTS = {
    'base': {
        'ukrainian': 0.25,
        'completeness': 0.20,
        'structure': 0.10,
        'correctness': 0.15,  # NEW: Answer correctness
    },
    'bonus': {
        'retrieval': 0.15,
        'rerank': 0.15,
        'tool_usage': 0.10,
        'verification': 0.05,
        'quality_score': 0.15,
    }
}


def calculate_base_score(ukrainian_ratio: float, completeness: float, structure_score: float, correctness: float = 0.0) -> float:
    """Calculate base quality score common to all experiments."""
    return (
        WEIGHTS['base']['ukrainian'] * ukrainian_ratio +
        WEIGHTS['base']['completeness'] * completeness +
        WEIGHTS['base']['structure'] * structure_score +
        WEIGHTS['base']['correctness'] * correctness
    )


def evaluate_baseline(answer: str, answer_length: int, expected_answer: str = None) -> Dict[str, Any]:
    """Evaluate Experiment 1: Baseline (No RAG)."""
    ukrainian_ratio = calculate_ukrainian_ratio(answer)
    structure_score = 1.0 if has_structure(answer) else 0.0
    completeness = calculate_completeness(answer_length)
    correctness = calculate_answer_correctness(answer, expected_answer) if expected_answer else 0.0

    base = calculate_base_score(ukrainian_ratio, completeness, structure_score, correctness)

    return {
        'ukrainian_ratio': ukrainian_ratio,
        'completeness': completeness,
        'has_structure': structure_score > 0,
        'structure_rate': structure_score,
        'citation_rate': 0.0,
        'correctness': correctness,
        'overall_score': base
    }


def evaluate_basic_rag(answer: str, answer_length: int, retrieval_quality: float, expected_answer: str = None) -> Dict[str, Any]:
    """Evaluate Experiment 2: Basic RAG."""
    ukrainian_ratio = calculate_ukrainian_ratio(answer)
    structure_score = 1.0 if has_structure(answer) else 0.0
    citation_score = 1.0 if has_citations(answer) else 0.0
    completeness = calculate_completeness(answer_length)
    correctness = calculate_answer_correctness(answer, expected_answer) if expected_answer else 0.0

    base = calculate_base_score(ukrainian_ratio, completeness, structure_score, correctness)
    bonus = WEIGHTS['bonus']['retrieval'] * 2 * retrieval_quality

    return {
        'ukrainian_ratio': ukrainian_ratio,
        'retrieval_quality': retrieval_quality,
        'completeness': completeness,
        'has_structure': structure_score > 0,
        'has_citations': citation_score > 0,
        'structure_rate': structure_score,
        'citation_rate': citation_score,
        'correctness': correctness,
        'overall_score': base + bonus
    }


def evaluate_advanced_rag(answer: str, answer_length: int, retrieval_quality: float, rerank_quality: float, expected_answer: str = None) -> Dict[str, Any]:
    """Evaluate Experiment 3: Advanced RAG."""
    ukrainian_ratio = calculate_ukrainian_ratio(answer)
    structure_score = 1.0 if has_structure(answer) else 0.0
    citation_score = 1.0 if has_citations(answer) else 0.0
    completeness = calculate_completeness(answer_length)
    correctness = calculate_answer_correctness(answer, expected_answer) if expected_answer else 0.0

    base = calculate_base_score(ukrainian_ratio, completeness, structure_score, correctness)
    bonus = (
        WEIGHTS['bonus']['retrieval'] * retrieval_quality +
        WEIGHTS['bonus']['rerank'] * rerank_quality
    )

    return {
        'ukrainian_ratio': ukrainian_ratio,
        'retrieval_quality': retrieval_quality,
        'rerank_quality': rerank_quality,
        'completeness': completeness,
        'has_structure': structure_score > 0,
        'has_citations': citation_score > 0,
        'structure_rate': structure_score,
        'citation_rate': citation_score,
        'correctness': correctness,
        'overall_score': base + bonus
    }


def evaluate_rag_tools(answer: str, answer_length: int, retrieval_quality: float, tool_usage: bool, verified: bool, expected_answer: str = None) -> Dict[str, Any]:
    """Evaluate Experiment 4: RAG + Tools."""
    ukrainian_ratio = calculate_ukrainian_ratio(answer)
    structure_score = 1.0 if has_structure(answer) else 0.0
    citation_score = 1.0 if has_citations(answer) else 0.0
    completeness = calculate_completeness(answer_length)
    correctness = calculate_answer_correctness(answer, expected_answer) if expected_answer else 0.0

    base = calculate_base_score(ukrainian_ratio, completeness, structure_score, correctness)
    bonus = (
        WEIGHTS['bonus']['retrieval'] * retrieval_quality +
        WEIGHTS['bonus']['tool_usage'] * (1.0 if tool_usage else 0.0) +
        WEIGHTS['bonus']['verification'] * (1.0 if verified else 0.0)
    )

    return {
        'ukrainian_ratio': ukrainian_ratio,
        'retrieval_quality': retrieval_quality,
        'tool_usage': tool_usage,
        'verified': verified,
        'completeness': completeness,
        'has_structure': structure_score > 0,
        'has_citations': citation_score > 0,
        'structure_rate': structure_score,
        'citation_rate': citation_score,
        'correctness': correctness,
        'overall_score': base + bonus
    }


def evaluate_multi_agent(answer: str, answer_length: int, retrieval_quality: float, quality_score: float, iterations: int, expected_answer: str = None) -> Dict[str, Any]:
    """Evaluate Experiment 5: Multi-Agent."""
    ukrainian_ratio = calculate_ukrainian_ratio(answer)
    structure_score = 1.0 if has_structure(answer) else 0.0
    citation_score = 1.0 if has_citations(answer) else 0.0
    completeness = calculate_completeness(answer_length)
    correctness = calculate_answer_correctness(answer, expected_answer) if expected_answer else 0.0
    collaboration_quality = 1.0 / (iterations + 1)

    base = calculate_base_score(ukrainian_ratio, completeness, structure_score, correctness)
    bonus = (
        WEIGHTS['bonus']['retrieval'] * retrieval_quality +
        WEIGHTS['bonus']['quality_score'] * quality_score
    )

    return {
        'ukrainian_ratio': ukrainian_ratio,
        'retrieval_quality': retrieval_quality,
        'quality_score': quality_score,
        'completeness': completeness,
        'collaboration_quality': collaboration_quality,
        'has_structure': structure_score > 0,
        'has_citations': citation_score > 0,
        'structure_rate': structure_score,
        'citation_rate': citation_score,
        'correctness': correctness,
        'overall_score': base + bonus
    }


def create_metrics_visualization(
    evaluations: List[Dict],
    avg_metrics: Dict[str, float],
    output_path: Path,
    experiment_name: str,
    metric_names: List[str] = None
) -> None:
    """
    Create standardized 2-panel visualization for experiment metrics.

    Args:
        evaluations: List of evaluation dicts with 'metrics' key
        avg_metrics: Dict of average metrics
        output_path: Path to save the PNG file
        experiment_name: Name for plot title (e.g., "Baseline", "Basic RAG")
        metric_names: Optional list of metric names to display in right panel
    """
    import pandas as pd

    # Default metrics to display
    if metric_names is None:
        metric_names = ['ukrainian_ratio', 'completeness', 'structure_rate']

    # Create dataframe for questions
    df = pd.DataFrame([
        {
            'question_num': i+1,
            'overall': e['metrics']['overall_score']
        }
        for i, e in enumerate(evaluations)
    ])

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Per-question scores
    ax = axes[0]
    bars = ax.bar(df['question_num'], df['overall'],
                  color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(y=avg_metrics['overall_score'], color='red', linestyle='--',
               label=f"Avg: {avg_metrics['overall_score']:.3f}")
    ax.set_xlabel('Question')
    ax.set_ylabel('Overall Score')
    ax.set_title(f'{experiment_name}: Overall Scores', fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Right panel: Average metrics
    ax = axes[1]
    metrics_data = [avg_metrics.get(m, 0.0) for m in metric_names]
    labels = [m.replace('_', ' ').title() for m in metric_names]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#FF5722'][:len(metric_names)]

    bars = ax.bar(labels, metrics_data, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Score')
    ax.set_title('Average Metrics', fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, metrics_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")