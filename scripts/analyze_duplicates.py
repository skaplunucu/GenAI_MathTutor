#!/usr/bin/env python3
"""
Analyze and Handle Duplicate Content in Embeddings

This script helps you:
1. Find near-duplicate chunks across different textbooks
2. Analyze content overlap between books
3. Optionally deduplicate or merge similar chunks
4. Visualize semantic clusters
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(embeddings_path: Path, chunks_path: Path) -> Tuple[np.ndarray, List[Dict]]:
    """Load embeddings and chunks metadata."""
    print("Loading data...")
    embeddings = np.load(embeddings_path)
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f" Loaded {len(chunks):,} chunks with {embeddings.shape[1]}-dim embeddings")
    return embeddings, chunks


def find_near_duplicates(
    embeddings: np.ndarray,
    chunks: List[Dict],
    similarity_threshold: float = 0.95,
    max_pairs: int = 1000
) -> List[Tuple[int, int, float]]:
    """
    Find pairs of chunks that are very similar (likely duplicates).

    Args:
        embeddings: Embedding vectors
        chunks: Chunk metadata
        similarity_threshold: Cosine similarity threshold (0.95 = 95% similar)
        max_pairs: Maximum pairs to return

    Returns:
        List of (idx1, idx2, similarity_score) tuples
    """
    print(f"\nFinding near-duplicates (similarity >= {similarity_threshold})...")

    # For large datasets, sample or use approximate methods
    # For now, compute pairwise similarities in batches

    duplicates = []
    batch_size = 1000

    for i in range(0, len(embeddings), batch_size):
        batch_end = min(i + batch_size, len(embeddings))

        # Compute similarities for this batch against all
        similarities = cosine_similarity(embeddings[i:batch_end], embeddings)

        # Find pairs above threshold (excluding self-similarity)
        for local_idx in range(batch_end - i):
            global_idx = i + local_idx

            # Only look at upper triangle to avoid duplicate pairs
            for j in range(global_idx + 1, len(embeddings)):
                sim = similarities[local_idx, j]

                if sim >= similarity_threshold:
                    # Check if from different books (same book duplicates are less interesting)
                    if chunks[global_idx]['metadata']['filename'] != chunks[j]['metadata']['filename']:
                        duplicates.append((global_idx, j, float(sim)))

                        if len(duplicates) >= max_pairs:
                            break

            if len(duplicates) >= max_pairs:
                break

        if len(duplicates) >= max_pairs:
            break

        if (i // batch_size + 1) % 5 == 0:
            print(f"   Processed {batch_end:,}/{len(embeddings):,} chunks, found {len(duplicates)} duplicates...")

    print(f" Found {len(duplicates):,} near-duplicate pairs")
    return duplicates


def analyze_book_overlap(embeddings: np.ndarray, chunks: List[Dict]) -> Dict:
    """
    Analyze content overlap between different books.
    """
    print("\nAnalyzing book overlap...")

    # Group chunks by book
    book_chunks = defaultdict(list)
    book_embeddings = defaultdict(list)

    for idx, chunk in enumerate(chunks):
        book = chunk['metadata']['filename']
        book_chunks[book].append(idx)
        book_embeddings[book].append(embeddings[idx])

    # Convert to arrays
    for book in book_embeddings:
        book_embeddings[book] = np.array(book_embeddings[book])

    # Compute pairwise book similarities
    book_names = list(book_embeddings.keys())
    overlap_matrix = np.zeros((len(book_names), len(book_names)))

    for i, book1 in enumerate(book_names):
        for j, book2 in enumerate(book_names):
            if i == j:
                overlap_matrix[i, j] = 1.0
            elif i < j:
                # Sample for speed (compare max 500 chunks from each)
                emb1 = book_embeddings[book1][:500]
                emb2 = book_embeddings[book2][:500]

                # Compute average max similarity
                sims = cosine_similarity(emb1, emb2)
                avg_max_sim = np.mean(np.max(sims, axis=1))

                overlap_matrix[i, j] = avg_max_sim
                overlap_matrix[j, i] = avg_max_sim

    return {
        'books': book_names,
        'overlap_matrix': overlap_matrix,
        'book_sizes': {book: len(indices) for book, indices in book_chunks.items()}
    }


def cluster_chunks(
    embeddings: np.ndarray,
    chunks: List[Dict],
    eps: float = 0.15,
    min_samples: int = 3
) -> np.ndarray:
    """
    Cluster chunks to find semantic topics.

    Args:
        embeddings: Embedding vectors
        chunks: Chunk metadata
        eps: DBSCAN epsilon (smaller = tighter clusters)
        min_samples: Minimum cluster size

    Returns:
        Cluster labels (-1 = noise)
    """
    print(f"\nClustering chunks (eps={eps}, min_samples={min_samples})...")

    # For large datasets, sample first
    if len(embeddings) > 5000:
        print(f"   Sampling 5000 chunks for clustering (dataset too large)...")
        indices = np.random.choice(len(embeddings), 5000, replace=False)
        sample_embeddings = embeddings[indices]
    else:
        sample_embeddings = embeddings
        indices = np.arange(len(embeddings))

    # Cluster using DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1)
    labels = clustering.fit_predict(sample_embeddings)

    # Expand labels back to full dataset
    full_labels = np.full(len(embeddings), -1)
    full_labels[indices] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f" Found {n_clusters} clusters, {n_noise} noise points")

    return full_labels


def analyze_cluster_diversity(
    labels: np.ndarray,
    chunks: List[Dict]
) -> Dict:
    """
    Analyze diversity of books/content types within each cluster.
    """
    print("\nAnalyzing cluster diversity...")

    cluster_info = defaultdict(lambda: {
        'size': 0,
        'books': Counter(),
        'content_types': Counter(),
        'grades': Counter()
    })

    for idx, label in enumerate(labels):
        if label == -1:
            continue

        chunk = chunks[idx]
        cluster_info[label]['size'] += 1
        cluster_info[label]['books'][chunk['metadata']['filename']] += 1
        cluster_info[label]['content_types'][chunk['content_type']] += 1

        # Extract grade from filename (if present)
        filename = chunk['metadata']['filename']
        for grade in range(1, 12):
            if f'-{grade}-' in filename or f'_{grade}_' in filename:
                cluster_info[label]['grades'][grade] += 1
                break

    return dict(cluster_info)


def recommend_deduplication_strategy(
    duplicates: List[Tuple[int, int, float]],
    chunks: List[Dict],
    overlap_analysis: Dict
) -> Dict:
    """
    Recommend whether to keep duplicates or deduplicate.
    """
    print("\n" + "="*80)
    print("DEDUPLICATION RECOMMENDATION")
    print("="*80)

    total_chunks = len(chunks)
    n_duplicates = len(duplicates)
    duplication_rate = n_duplicates / (total_chunks * (total_chunks - 1) / 2) * 100

    # Analyze duplicate characteristics
    high_sim_count = sum(1 for _, _, sim in duplicates if sim >= 0.98)

    # Analyze book diversity in duplicates
    book_pairs = Counter()
    for idx1, idx2, _ in duplicates:
        book1 = chunks[idx1]['metadata']['filename']
        book2 = chunks[idx2]['metadata']['filename']
        pair = tuple(sorted([book1, book2]))
        book_pairs[pair] += 1

    recommendation = {
        'total_chunks': total_chunks,
        'duplicate_pairs': n_duplicates,
        'high_similarity_pairs': high_sim_count,
        'duplication_rate': duplication_rate,
        'top_duplicate_book_pairs': book_pairs.most_common(5),
    }

    # Decision logic
    if duplication_rate < 1.0:
        recommendation['action'] = 'KEEP_ALL'
        recommendation['reason'] = "Low duplication rate - diversity is valuable"
    elif high_sim_count > n_duplicates * 0.5:
        recommendation['action'] = 'DEDUPLICATE'
        recommendation['reason'] = "Many near-exact duplicates - can safely remove"
    else:
        recommendation['action'] = 'KEEP_BEST'
        recommendation['reason'] = "Moderate duplicates - keep best example from each book"

    return recommendation


def deduplicate_embeddings(
    embeddings: np.ndarray,
    chunks: List[Dict],
    duplicates: List[Tuple[int, int, float]],
    strategy: str = 'keep_highest_confidence'
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Remove or merge duplicate chunks.

    Strategies:
    - 'keep_highest_confidence': Keep chunk with highest LLM confidence
    - 'keep_first': Keep first occurrence
    - 'merge_metadata': Merge sources into single chunk
    """
    print(f"\nDeduplicating using strategy: {strategy}...")

    # Build duplicate groups
    from collections import defaultdict
    import networkx as nx

    # Build graph of duplicates
    G = nx.Graph()
    for idx1, idx2, sim in duplicates:
        G.add_edge(idx1, idx2, similarity=sim)

    # Find connected components (groups of duplicates)
    duplicate_groups = list(nx.connected_components(G))

    print(f"   Found {len(duplicate_groups)} duplicate groups")

    # Decide which to keep
    indices_to_keep = set(range(len(chunks)))

    for group in duplicate_groups:
        group_list = list(group)

        if strategy == 'keep_highest_confidence':
            # Keep the one with highest LLM confidence
            best_idx = max(group_list, key=lambda i: chunks[i]['confidence'])
        elif strategy == 'keep_first':
            best_idx = min(group_list)
        else:
            best_idx = group_list[0]

        # Remove others
        for idx in group_list:
            if idx != best_idx:
                indices_to_keep.discard(idx)

    # Filter
    indices_to_keep = sorted(list(indices_to_keep))
    new_embeddings = embeddings[indices_to_keep]
    new_chunks = [chunks[i] for i in indices_to_keep]

    print(f" Reduced from {len(chunks):,} to {len(new_chunks):,} chunks ({len(chunks) - len(new_chunks):,} removed)")

    return new_embeddings, new_chunks


def visualize_book_overlap(overlap_analysis: Dict, output_path: Path):
    """Create heatmap of book content overlap."""
    print(f"\nGenerating overlap heatmap...")

    overlap_matrix = overlap_analysis['overlap_matrix']
    books = overlap_analysis['books']

    # Shorten book names for display
    short_names = [b.split('.')[0][:30] for b in books]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        overlap_matrix,
        xticklabels=short_names,
        yticklabels=short_names,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Content Similarity'}
    )
    plt.title('Content Overlap Between Textbooks', fontsize=16, fontweight='bold')
    plt.xlabel('Book', fontsize=12)
    plt.ylabel('Book', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f" Saved heatmap to {output_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze duplicate content in embeddings")
    parser.add_argument('data_dir', type=Path, help="Data directory with embeddings/")
    parser.add_argument('--similarity-threshold', type=float, default=0.95,
                        help="Similarity threshold for duplicates (default: 0.95)")
    parser.add_argument('--deduplicate', action='store_true',
                        help="Actually perform deduplication")
    parser.add_argument('--strategy', default='keep_highest_confidence',
                        choices=['keep_highest_confidence', 'keep_first'],
                        help="Deduplication strategy")

    args = parser.parse_args()

    # Paths
    embeddings_path = args.data_dir / "embeddings" / "embeddings.npy"
    chunks_path = args.data_dir / "embeddings" / "chunks_metadata.json"
    output_dir = args.data_dir / "analysis"
    output_dir.mkdir(exist_ok=True)

    # Load data
    embeddings, chunks = load_data(embeddings_path, chunks_path)

    # Find near-duplicates
    duplicates = find_near_duplicates(
        embeddings,
        chunks,
        similarity_threshold=args.similarity_threshold,
        max_pairs=2000
    )

    # Analyze book overlap
    overlap_analysis = analyze_book_overlap(embeddings, chunks)

    # Visualize
    visualize_book_overlap(overlap_analysis, output_dir / "book_overlap_heatmap.png")

    # Get recommendation
    recommendation = recommend_deduplication_strategy(duplicates, chunks, overlap_analysis)

    # Print recommendation
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nTotal chunks: {recommendation['total_chunks']:,}")
    print(f"Duplicate pairs found: {recommendation['duplicate_pairs']:,}")
    print(f"High similarity pairs (>98%): {recommendation['high_similarity_pairs']:,}")
    print(f"Duplication rate: {recommendation['duplication_rate']:.2f}%")

    print(f"\n**RECOMMENDATION: {recommendation['action']}**")
    print(f"Reason: {recommendation['reason']}")

    print(f"\nTop duplicate book pairs:")
    for (book1, book2), count in recommendation['top_duplicate_book_pairs']:
        print(f"  {book1[:40]:40s} <-> {book2[:40]:40s}: {count} duplicates")

    # Perform deduplication if requested
    if args.deduplicate:
        print("\n" + "="*80)
        print("PERFORMING DEDUPLICATION")
        print("="*80)

        new_embeddings, new_chunks = deduplicate_embeddings(
            embeddings,
            chunks,
            duplicates,
            strategy=args.strategy
        )

        # Save deduplicated data
        output_emb_path = args.data_dir / "embeddings" / "embeddings_deduplicated.npy"
        output_chunks_path = args.data_dir / "embeddings" / "chunks_metadata_deduplicated.json"

        np.save(output_emb_path, new_embeddings)
        with open(output_chunks_path, 'w', encoding='utf-8') as f:
            json.dump(new_chunks, f, ensure_ascii=False, indent=2)

        print(f"\n Saved deduplicated embeddings to {output_emb_path}")
        print(f" Saved deduplicated chunks to {output_chunks_path}")
        print(f"\nTo use deduplicated version, run:")
        print(f"  python create_db_from_embeddings.py --embeddings {output_emb_path.name}")

    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
