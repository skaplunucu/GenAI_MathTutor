#!/usr/bin/env python3
"""
Interactive Retrieval Demo

Takes user questions and retrieves complete information from ChromaDB.
Shows full text of retrieved chunks for quality evaluation.
"""

import chromadb
from pathlib import Path
from typing import List, Dict, Any
from chromadb.utils import embedding_functions
import sys


class InteractiveRetrieval:
    """Interactive retrieval from ChromaDB."""

    def __init__(self, db_path: Path):
        """Initialize retrieval system."""
        print("Initializing retrieval system...")

        # Initialize client
        self.client = chromadb.PersistentClient(path=str(db_path))

        # Get collection
        self.collection = self.client.get_collection(
            name="ukrainian_math",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            )
        )

        print(f" Loaded collection: {self.collection.name}")
        print(f" Total chunks in database: {self.collection.count()}")
        print()

    def retrieve(
        self,
        question: str,
        n_results: int = 5,
        content_type_filter: str = None
    ) -> Dict[str, Any]:
        """
        Retrieve information for a question.

        Args:
            question: The user's question
            n_results: Number of results to return
            content_type_filter: Optional filter (e.g., 'explanation', 'problem')

        Returns:
            Query results
        """
        where = None
        if content_type_filter:
            where = {"content_type": content_type_filter}

        results = self.collection.query(
            query_texts=[question],
            n_results=n_results,
            where=where
        )

        return results

    def display_results(
        self,
        question: str,
        results: Dict[str, Any]
    ):
        """Display complete retrieval results."""
        print("="*80)
        print(f"QUESTION: {question}")
        print("="*80)
        print(f"\nFound {len(results['documents'][0])} relevant chunks\n")

        for i, (doc, meta, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            print(f"\n{'─'*80}")
            print(f"RESULT #{i}")
            print(f"{'─'*80}")
            print(f"Content Type:  {meta['content_type']}")
            print(f"Confidence:    {meta['confidence']:.2f}")
            print(f"Relevance:     {1 - distance:.3f} (distance: {distance:.3f})")
            print(f"Source:        {meta['filename']}")
            print(f"Pages:         {meta['page_start']}-{meta['page_end']}")
            print(f"Chunk Size:    {meta['char_count']} characters")

            if meta['has_images'] == 'yes':
                print(f"Images:        {meta['num_images']} image(s) attached")

            print(f"\n{'─'*80}")
            print("FULL TEXT:")
            print(f"{'─'*80}")
            print(doc)
            print()

        # Summary statistics
        print("\n" + "="*80)
        print("RETRIEVAL SUMMARY")
        print("="*80)

        avg_confidence = sum(m['confidence'] for m in results['metadatas'][0]) / len(results['metadatas'][0])
        avg_relevance = sum(1 - d for d in results['distances'][0]) / len(results['distances'][0])

        print(f"Average Confidence: {avg_confidence:.2f}")
        print(f"Average Relevance:  {avg_relevance:.3f}")

        # Content type breakdown
        from collections import Counter
        content_types = [m['content_type'] for m in results['metadatas'][0]]
        type_counts = Counter(content_types)

        print(f"\nContent Types Retrieved:")
        for ctype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  - {ctype}: {count}")

        print()

    def interactive_mode(self):
        """Run in interactive mode."""
        print("="*80)
        print("INTERACTIVE RETRIEVAL DEMO")
        print("="*80)
        print("\nEnter your questions in Ukrainian or English.")
        print("Commands:")
        print("  - Type your question to search")
        print("  - 'filter:explanation' - Show only theory/explanations")
        print("  - 'filter:problem' - Show only practice problems")
        print("  - 'filter:theorem' - Show only theorems")
        print("  - 'filter:none' - Clear filter")
        print("  - 'n:X' - Set number of results (e.g., 'n:10')")
        print("  - 'quit' or 'exit' - Exit")
        print()

        n_results = 5
        content_filter = None

        while True:
            try:
                print("─"*80)
                user_input = input("Your question: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                # Handle commands
                if user_input.startswith('filter:'):
                    filter_type = user_input.split(':', 1)[1].strip()
                    if filter_type.lower() == 'none':
                        content_filter = None
                        print(f" Filter cleared\n")
                    else:
                        content_filter = filter_type
                        print(f" Filter set to: {content_filter}\n")
                    continue

                if user_input.startswith('n:'):
                    try:
                        n_results = int(user_input.split(':', 1)[1].strip())
                        print(f" Number of results set to: {n_results}\n")
                    except ValueError:
                        print(" Invalid number format\n")
                    continue

                # Perform retrieval
                print(f"\n Searching for: '{user_input}'")
                if content_filter:
                    print(f"   Filter: {content_filter}")
                print(f"   Number of results: {n_results}\n")

                results = self.retrieve(user_input, n_results, content_filter)
                self.display_results(user_input, results)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n Error: {e}\n")


def main():
    # Path to database
    db_path = Path(__file__).parent.parent / "data" / "vector_db"

    if not db_path.exists():
        print(f" ERROR: Database not found at {db_path}")
        print("Please run create_vector_db.py first!")
        sys.exit(1)

    # Initialize retrieval system
    retriever = InteractiveRetrieval(db_path)

    # Check if question provided as command line argument
    if len(sys.argv) > 1:
        # Single question mode
        question = " ".join(sys.argv[1:])
        print(f"Single query mode: '{question}'\n")
        results = retriever.retrieve(question, n_results=5)
        retriever.display_results(question, results)
    else:
        # Interactive mode
        retriever.interactive_mode()


if __name__ == "__main__":
    main()
