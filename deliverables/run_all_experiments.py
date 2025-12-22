#!/usr/bin/env python3
"""
Run all Math Tutor experiments sequentially.

Usage:
    cd deliverables
    python run_all_experiments.py
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import papermill as pm
    PAPERMILL_AVAILABLE = True
except ImportError:
    PAPERMILL_AVAILABLE = False
    print("Warning: papermill not installed. Install with: pip install papermill")

DELIVERABLES_DIR = Path(".")
RESULTS_DIR = Path("results")
EVALUATION_DIR = Path("../evaluation")
REPORT_DIR = EVALUATION_DIR / "final_report"

RESULTS_DIR.mkdir(exist_ok=True, parents=True)
REPORT_DIR.mkdir(exist_ok=True, parents=True)

EXPERIMENTS = [
    {
        'id': 'experiment_01',
        'name': 'Baseline (No RAG)',
        'notebook': 'experiment_01_baseline_no_rag.ipynb',
        'difficulty': 'Easy',
        'output_dir': EVALUATION_DIR / 'experiment_01'
    },
    {
        'id': 'experiment_02',
        'name': 'Basic RAG',
        'notebook': 'experiment_02_basic_rag.ipynb',
        'difficulty': 'Easy-Medium',
        'output_dir': EVALUATION_DIR / 'experiment_02'
    },
    {
        'id': 'experiment_03',
        'name': 'Advanced RAG',
        'notebook': 'experiment_03_advanced_rag.ipynb',
        'difficulty': 'Medium',
        'output_dir': EVALUATION_DIR / 'experiment_03'
    },
    {
        'id': 'experiment_04',
        'name': 'RAG + Tools',
        'notebook': 'experiment_04_rag_with_tools.ipynb',
        'difficulty': 'Medium-Hard',
        'output_dir': EVALUATION_DIR / 'experiment_04'
    },
    {
        'id': 'experiment_05',
        'name': 'Multi-Agent',
        'notebook': 'experiment_05_multi_agent.ipynb',
        'difficulty': 'Hard',
        'output_dir': EVALUATION_DIR / 'experiment_05'
    }
]


def print_header(text: str, char: str = "="):
    """Print formatted header."""
    width = 80
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}\n")


def clear_gpu_memory():
    """Clear GPU memory before running experiment."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    import gc
    gc.collect()


def run_notebook(notebook_path: Path, output_path: Path, timeout: int = 3600) -> bool:
    """Run a Jupyter notebook using papermill."""
    if not PAPERMILL_AVAILABLE:
        print(f"Error: Papermill not available")
        return False

    print(f"Running: {notebook_path.name}")

    try:
        start_time = time.time()
        pm.execute_notebook(
            input_path=str(notebook_path),
            output_path=str(output_path),
            kernel_name='python3',
            execution_timeout=timeout,
            progress_bar=True,
            log_output=False
        )
        elapsed = time.time() - start_time
        print(f"Success ({elapsed:.1f}s)")
        return True

    except pm.PapermillExecutionError as e:
        elapsed = time.time() - start_time
        print(f"Execution failed ({elapsed:.1f}s)")
        print(f"Error: {str(e)[:200]}")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Exception ({elapsed:.1f}s)")
        print(f"Error: {str(e)}")
        return False


def load_results(experiment: Dict) -> Dict[str, Any]:
    """Load results.json from experiment output directory."""
    results_path = experiment['output_dir'] / 'results.json'

    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def extract_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics from results."""
    if not results or 'avg_metrics' not in results:
        return {
            'overall_score': 0.0,
            'ukrainian_ratio': 0.0,
            'completeness': 0.0
        }

    metrics = results['avg_metrics']

    extracted = {
        'overall_score': metrics.get('overall_score', 0.0),
        'ukrainian_ratio': metrics.get('ukrainian_ratio', 0.0),
        'completeness': metrics.get('completeness', 0.0)
    }

    if 'retrieval_quality' in metrics:
        extracted['retrieval_quality'] = metrics['retrieval_quality']
    if 'rerank_quality' in metrics:
        extracted['rerank_quality'] = metrics['rerank_quality']
    if 'tool_usage_rate' in metrics:
        extracted['tool_usage_rate'] = metrics['tool_usage_rate']
    if 'quality_score' in metrics:
        extracted['quality_score'] = metrics['quality_score']
    if 'structure_rate' in metrics:
        extracted['structure_rate'] = metrics['structure_rate']
    if 'citation_rate' in metrics:
        extracted['citation_rate'] = metrics['citation_rate']

    return extracted


def create_comparison_table(all_results: List[Dict]) -> pd.DataFrame:
    """Create comparison table of all experiments."""
    data = []

    for exp_config, results in all_results:
        metrics = extract_metrics(results)

        row = {
            'Experiment': exp_config['name'],
            'Difficulty': exp_config['difficulty'],
            'Overall Score': metrics.get('overall_score', 0.0),
            'Ukrainian': metrics.get('ukrainian_ratio', 0.0),
            'Completeness': metrics.get('completeness', 0.0),
            'Retrieval': metrics.get('retrieval_quality', 0.0),
            'Structure': metrics.get('structure_rate', 0.0),
            'Citations': metrics.get('citation_rate', 0.0)
        }

        if 'rerank_quality' in metrics:
            row['Re-rank'] = metrics['rerank_quality']
        if 'tool_usage_rate' in metrics:
            row['Tool Usage'] = metrics['tool_usage_rate']
        if 'quality_score' in metrics:
            row['QA Score'] = metrics['quality_score']

        data.append(row)

    return pd.DataFrame(data)


def plot_comparison(df: pd.DataFrame, output_dir: Path):
    """Create comparison visualizations."""
    sns.set_style('whitegrid')

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = ax.bar(df['Experiment'], df['Overall Score'],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('Overall Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_overall.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    metrics_to_plot = ['Overall Score', 'Ukrainian', 'Completeness', 'Retrieval']

    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]

    for i, exp in enumerate(df['Experiment']):
        values = df.iloc[i][metrics_to_plot].fillna(0).tolist()
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=exp, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Metric Performance Radar', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_radar.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))

    heatmap_metrics = ['Overall Score', 'Ukrainian', 'Completeness', 'Retrieval', 'Structure', 'Citations']
    heatmap_data = df[['Experiment'] + heatmap_metrics].set_index('Experiment')
    heatmap_data = heatmap_data.fillna(0)

    sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, cbar_kws={'label': 'Score'},
                linewidths=0.5, ax=ax)

    ax.set_title('Metrics Heatmap Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualizations to {output_dir}")

def main():
    """Main execution function."""
    print_header("MATH TUTOR - RUN ALL EXPERIMENTS", "=")

    print(f"Deliverables: {DELIVERABLES_DIR.resolve()}")
    print(f"Evaluation: {EVALUATION_DIR.resolve()}")
    print(f"Report: {REPORT_DIR.resolve()}")

    print(f"\nFound {len(EXPERIMENTS)} experiments:")
    for exp in EXPERIMENTS:
        notebook_path = DELIVERABLES_DIR / exp['notebook']
        status = "OK" if notebook_path.exists() else "MISSING"
        print(f"  [{status}] {exp['name']}")

    if not PAPERMILL_AVAILABLE:
        print("\nError: Papermill not installed!")
        print("Install with: pip install papermill")
        return

    print("\n" + "="*80)
    response = input("Run all experiments? This may take 1-2 hours. (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nResults will be saved to: {run_dir.resolve()}")

    print_header("RUNNING EXPERIMENTS")

    results_collection = []
    success_count = 0

    for i, exp_config in enumerate(EXPERIMENTS, 1):
        print_header(f"Experiment {i}/5: {exp_config['name']}", "-")

        clear_gpu_memory()

        notebook_path = DELIVERABLES_DIR / exp_config['notebook']
        output_notebook = run_dir / f"{exp_config['id']}_executed.ipynb"

        success = run_notebook(notebook_path, output_notebook, timeout=3600)

        if success:
            success_count += 1
            results = load_results(exp_config)
            results_collection.append((exp_config, results))
            print(f"Results loaded from {exp_config['output_dir']}")
        else:
            print(f"Experiment failed, using empty results")
            results_collection.append((exp_config, {}))

        print(f"\nProgress: {i}/{len(EXPERIMENTS)} experiments completed")

    print_header("GENERATING COMPARISON REPORT")

    if results_collection:
        df = create_comparison_table(results_collection)
        print("\nComparison Table:")
        print(df.to_string(index=False))

        csv_path = REPORT_DIR / 'comparison_table.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nSaved to {csv_path}")

        print("\nCreating visualizations...")
        plot_comparison(df, REPORT_DIR)

        run_summary = {
            'timestamp': timestamp,
            'run_directory': str(run_dir.resolve()),
            'total_experiments': len(EXPERIMENTS),
            'successful': success_count,
            'failed': len(EXPERIMENTS) - success_count,
            'experiments': [
                {
                    'id': exp['id'],
                    'name': exp['name'],
                    'executed_notebook': str((run_dir / f"{exp['id']}_executed.ipynb").resolve())
                }
                for exp in EXPERIMENTS
            ]
        }

        with open(run_dir / 'run_summary.json', 'w', encoding='utf-8') as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)

        print(f"\nRun summary saved to {run_dir / 'run_summary.json'}")

        print_header("ALL EXPERIMENTS COMPLETE!", "=")
        print(f"Successful: {success_count}/{len(EXPERIMENTS)}")
        print(f"Executed notebooks: {run_dir.resolve()}")
        print(f"Report: {(REPORT_DIR / 'FINAL_REPORT.md').resolve()}")
        print(f"Visualizations: {REPORT_DIR.resolve()}")
        print(f"\nDone! Check {REPORT_DIR.resolve()} for report and {run_dir.resolve()} for notebooks.")
    else:
        print("No results to compare")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)