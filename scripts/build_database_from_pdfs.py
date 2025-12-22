#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
import time

sys.path.append(str(Path(__file__).parent.parent))


def find_pdfs(directory: Path, recursive: bool = True) -> List[Path]:
    if recursive:
        pdf_files = list(directory.rglob("*.pdf"))
    else:
        pdf_files = list(directory.glob("*.pdf"))

    return sorted(pdf_files)

def is_pdf_processed(pdf_path: Path, output_dir: Path) -> bool:
    safe_name = pdf_path.stem.replace(" ", "_")
    chunks_file = output_dir / "chunks" / f"{safe_name}_chunks.jsonl"

    return chunks_file.exists()

def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    llm_classifier,
    vlm_analyzer,
    batch_size: int = 20,
) -> Dict[str, Any]:
    import pdfplumber
    import json

    safe_name = pdf_path.stem.replace(" ", "_")

    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunks_file = chunks_dir / f"{safe_name}_chunks.jsonl"
    metadata_file = chunks_dir / f"{safe_name}_metadata.json"

    start_time = time.time()

    try:
        print("\n Extracting PDF layout...")
        blocks = []
        images = []

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"   Total pages: {total_pages}")

            for page_num, page in enumerate(pdf.pages, 1):
                if page_num % 50 == 0 or page_num == 1:
                    print(f"   Processing page {page_num}/{total_pages}...")

                text = page.extract_text()
                if text:
                    # Split into blocks (paragraphs)
                    for block_text in text.split('\n\n'):
                        if block_text.strip():
                            blocks.append({
                                'text': block_text.strip(),
                                'page': page_num
                            })

                page_images = page.images
                for img_idx, img in enumerate(page_images):
                    images.append({
                        'page': page_num,
                        'index': img_idx,
                        'bbox': (img['x0'], img['top'], img['x1'], img['bottom'])
                    })

        print(f" Extracted {len(blocks)} text blocks and {len(images)} images")

        print(f"\n Classifying {len(blocks)} blocks with LLM...")
        texts = [block['text'] for block in blocks]

        classifications = llm_classifier.classify_batch(
            texts,
            batch_size=batch_size,
            temperature=0.1
        )

        for block, classification in zip(blocks, classifications):
            block['content_type'] = classification.type
            block['confidence'] = classification.confidence

        llm_stats = llm_classifier.get_statistics(classifications)
        print(f" LLM classification complete (avg confidence: {llm_stats['average_confidence']:.2f})")

        vlm_stats = None
        if images and vlm_analyzer:
            print(f"\n  Analyzing {len(images)} images with VLM...")
            # Simplified: would need actual image extraction here
            vlm_stats = {
                'total_images': len(images),
                'note': 'VLM processing skipped in batch mode (implement if needed)'
            }
            print(f" Found {len(images)} images (extraction skipped in batch mode)")

        print(f"\n Creating semantic chunks...")
        chunks = create_semantic_chunks(blocks, pdf_path.name)
        print(f" Created {len(chunks)} chunks")

        print(f"\n Saving results...")

        # Save chunks
        with open(chunks_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

        metadata = {
            'pdf_name': pdf_path.name,
            'pdf_path': str(pdf_path),
            'total_chunks': len(chunks),
            'total_images': len(images),
            'llm_stats': llm_stats,
            'vlm_stats': vlm_stats,
            'processing_time': time.time() - start_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f" Saved to {chunks_file}")
        print(f" Processing time: {metadata['processing_time']:.1f}s")

        return metadata

    except Exception as e:
        print(f"\n ERROR processing {pdf_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'pdf_name': pdf_path.name,
            'error': str(e),
            'processing_time': time.time() - start_time
        }


def create_semantic_chunks(blocks: List[Dict], pdf_name: str) -> List[Dict]:
    chunks = []

    current_chunk = {
        'blocks': [],
        'content_type': None,
        'page_start': None,
        'page_end': None
    }

    for block in blocks:
        should_start_new = False

        if not current_chunk['blocks']:
            should_start_new = False
        elif current_chunk['content_type'] != block['content_type']:
            should_start_new = True
        elif sum(len(b['text']) for b in current_chunk['blocks']) > 1000:
            should_start_new = True
        elif current_chunk['page_end'] and (block['page'] - current_chunk['page_end']) > 1:
            should_start_new = True

        if should_start_new and current_chunk['blocks']:
            chunks.append(finalize_chunk(current_chunk, pdf_name))
            current_chunk = {
                'blocks': [],
                'content_type': None,
                'page_start': None,
                'page_end': None
            }

        current_chunk['blocks'].append(block)
        current_chunk['content_type'] = block['content_type']
        if current_chunk['page_start'] is None:
            current_chunk['page_start'] = block['page']
        current_chunk['page_end'] = block['page']

    if current_chunk['blocks']:
        chunks.append(finalize_chunk(current_chunk, pdf_name))

    return chunks


def finalize_chunk(chunk_data: Dict, pdf_name: str) -> Dict:
    text = '\n'.join(b['text'] for b in chunk_data['blocks'])

    confidences = [b['confidence'] for b in chunk_data['blocks']]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

    return {
        'text': text,
        'content_type': chunk_data['content_type'],
        'confidence': avg_confidence,
        'metadata': {
            'filename': pdf_name,
            'page_start': chunk_data['page_start'],
            'page_end': chunk_data['page_end'],
            'char_count': len(text),
            'block_count': len(chunk_data['blocks'])
        }
    }


def combine_all_chunks(output_dir: Path) -> List[Dict]:
    print("\n Combining chunks from all PDFs...")

    chunks_dir = output_dir / "chunks"
    all_chunks = []

    chunk_files = sorted(chunks_dir.glob("*_chunks.jsonl"))

    for chunk_file in chunk_files:
        with open(chunk_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_chunks.append(json.loads(line))

    print(f" Combined {len(all_chunks)} chunks from {len(chunk_files)} PDFs")

    return all_chunks


def create_embeddings(chunks: List[Dict], output_dir: Path):
    print("\n Creating embeddings...")

    from sentence_transformers import SentenceTransformer
    import numpy as np

    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    print(f"   Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    texts = [chunk['text'] for chunk in chunks]
    print(f"   Encoding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )

    embeddings_file = embeddings_dir / "embeddings.npy"
    np.save(embeddings_file, embeddings)
    print(f" Saved embeddings to {embeddings_file}")

    metadata_file = embeddings_dir / "chunks_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f" Saved metadata to {metadata_file}")

    # Save summary
    content_types = Counter(c['content_type'] for c in chunks)
    summary = {
        'num_chunks': len(chunks),
        'embedding_dim': embeddings.shape[1],
        'model_name': model_name,
        'content_types': dict(content_types),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    summary_file = embeddings_dir / "embeddings_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f" Saved summary to {summary_file}")

    return embeddings


def create_vector_db(embeddings, chunks: List[Dict], output_dir: Path):
    print("\n  Creating ChromaDB vector database...")

    import chromadb
    from chromadb.utils import embedding_functions

    db_path = output_dir / "vector_db"

    client = chromadb.PersistentClient(path=str(db_path))

    try:
        client.delete_collection(name="ukrainian_math")
        print("   Deleted existing collection")
    except:
        pass

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    collection = client.create_collection(
        name="ukrainian_math",
        embedding_function=embedding_function,
        metadata={
            "description": "Ukrainian mathematics textbooks - AI processed",
            "model": "paraphrase-multilingual-mpnet-base-v2",
            "num_chunks": len(chunks)
        }
    )
    print(f" Created collection: ukrainian_math")

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [chunk['text'] for chunk in chunks]
    embeddings_list = embeddings.tolist()

    metadatas = []
    for chunk in chunks:
        meta = {
            'content_type': chunk['content_type'],
            'confidence': float(chunk['confidence']),
            'filename': chunk['metadata']['filename'],
            'page_start': int(chunk['metadata']['page_start']),
            'page_end': int(chunk['metadata']['page_end']),
            'char_count': int(chunk['metadata']['char_count'])
        }
        metadatas.append(meta)

    print(f"   Adding {len(chunks)} chunks in batches...")

    BATCH_SIZE = 5000
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(0, len(chunks), BATCH_SIZE):
        batch_end = min(batch_idx + BATCH_SIZE, len(chunks))
        batch_num = batch_idx // BATCH_SIZE + 1

        print(f"   Batch {batch_num}/{total_batches}: Adding chunks {batch_idx}-{batch_end}...")

        collection.add(
            ids=ids[batch_idx:batch_end],
            embeddings=embeddings_list[batch_idx:batch_end],
            documents=documents[batch_idx:batch_end],
            metadatas=metadatas[batch_idx:batch_end]
        )

    print(f" Vector database created at {db_path} with {len(chunks):,} chunks")

    return collection


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build vector database from PDF directory"
    )
    parser.add_argument(
        "pdf_directory",
        type=Path,
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: ./data)"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="LLM batch size (default: 20)"
    )
    parser.add_argument(
        "--skip-processed",
        action="store_true",
        help="Skip already processed PDFs (default: True)",
        default=True
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/home/sskaplun/study/genAI/kaggle/models/gemma-2-9b-it"),
        help="Path to LLM model"
    )

    args = parser.parse_args()

    if args.output is None:
        output_dir = Path(__file__).parent.parent / "data"
    else:
        output_dir = args.output

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  PDF Directory: {args.pdf_directory}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Recursive: {not args.no_recursive}")
    print(f"  LLM Model: {args.model_path}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Skip Processed: {args.skip_processed}")

    print(f"\n{'='*80}")
    print("STEP 1: Finding PDFs")
    print("="*80)

    pdf_files = find_pdfs(args.pdf_directory, recursive=not args.no_recursive)
    print(f"Found {len(pdf_files)} PDF files")

    if not pdf_files:
        print(" No PDF files found!")
        sys.exit(1)

    for i, pdf in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf.name}")

    print(f"\n{'='*80}")
    print("STEP 2: Initializing AI Models")
    print("="*80)

    from llm_content_classifier import LLMContentClassifier

    print("\nLoading LLM classifier...")
    llm_classifier = LLMContentClassifier(
        model_path=str(args.model_path),
        load_in_4bit=True,
        verbose=False
    )
    print(" LLM classifier ready")

    vlm_analyzer = None  # TODO: Add VLM if needed

    print(f"\n{'='*80}")
    print("STEP 3: Processing PDFs")
    print("="*80)

    results = []
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] {pdf_path.name}")

        if args.skip_processed and is_pdf_processed(pdf_path, output_dir):
            print("⏭  Already processed, skipping...")
            skipped_count += 1
            continue

        result = process_single_pdf(
            pdf_path,
            output_dir,
            llm_classifier,
            vlm_analyzer,
            batch_size=args.batch_size
        )

        results.append(result)

        if 'error' in result:
            error_count += 1
        else:
            processed_count += 1

    print(f"\n{'='*80}")
    print("STEP 4: Combining Results")
    print("="*80)

    all_chunks = combine_all_chunks(output_dir)

    print(f"\n{'='*80}")
    print("STEP 5: Creating Embeddings")
    print("="*80)

    embeddings = create_embeddings(all_chunks, output_dir)

    # Create vector database
    print(f"\n{'='*80}")
    print("STEP 6: Building Vector Database")
    print("="*80)

    collection = create_vector_db(embeddings, all_chunks, output_dir)

    # Final summary
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\n Processing Summary:")
    print(f"   Total PDFs found: {len(pdf_files)}")
    print(f"   Processed: {processed_count}")
    print(f"   Skipped: {skipped_count}")
    print(f"   Errors: {error_count}")
    print(f"\n Database Statistics:")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Embedding dimensions: {embeddings.shape[1]}")
    print(f"   Database location: {output_dir / 'vector_db'}")

    content_types = Counter(c['content_type'] for c in all_chunks)
    print(f"\n Content Distribution:")
    for ctype, count in sorted(content_types.items(), key=lambda x: -x[1]):
        pct = count / len(all_chunks) * 100
        print(f"   {ctype:12s}: {count:5d} ({pct:5.1f}%)")

if __name__ == "__main__":
    main()
