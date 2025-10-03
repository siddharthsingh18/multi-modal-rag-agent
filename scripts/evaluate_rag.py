#!/usr/bin/env python3
"""Script for evaluating RAG system quality."""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict

import typer
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.rag_agent import create_rag_agent
from src.evaluation.metrics import (
    RAGMetrics,
    answer_relevancy,
    context_precision,
    context_recall,
)
from src.observability.logging import get_logger, setup_logging
from src.retrieval.vector_store import get_retriever

setup_logging()
logger = get_logger(__name__)
console = Console()

app = typer.Typer()


@app.command()
def evaluate(
    test_file: str = typer.Argument(
        "configs/evaluation/sample_queries.json",
        help="Path to test queries JSON file",
    ),
    collection: str = typer.Option("documents", help="Collection name"),
):
    """
    Evaluate RAG system with test queries.

    Example:
        python scripts/evaluate_rag.py configs/evaluation/sample_queries.json
    """
    asyncio.run(_evaluate_async(test_file, collection))


async def _evaluate_async(test_file: str, collection: str):
    """Async evaluation logic."""
    console.print(f"\n[bold blue]RAG System Evaluation[/bold blue]")
    console.print(f"Test file: {test_file}")
    console.print(f"Collection: {collection}\n")

    try:
        # Load test queries
        test_path = Path(test_file)
        if not test_path.exists():
            # Create sample test file
            _create_sample_test_file(test_path)
            console.print(f"[yellow]Created sample test file: {test_file}[/yellow]\n")

        with open(test_path, "r") as f:
            test_data = json.load(f)

        test_queries = test_data.get("queries", [])

        if not test_queries:
            console.print("[red]No test queries found![/red]")
            return

        # Initialize RAG agent
        console.print("[yellow]Initializing RAG agent...[/yellow]")
        retriever = await get_retriever(collection_name=collection)
        agent = await create_rag_agent(retriever=retriever)

        # Evaluate queries
        metrics = RAGMetrics()
        results = []

        console.print(f"[yellow]Evaluating {len(test_queries)} queries...[/yellow]\n")

        for i, test_query in enumerate(test_queries, 1):
            query = test_query["query"]
            expected_keywords = test_query.get("expected_keywords", [])
            relevant_docs = test_query.get("relevant_docs", [])

            console.print(f"[cyan]Query {i}/{len(test_queries)}:[/cyan] {query}")

            # Run query
            result = await agent.run(query=query)

            # Calculate metrics
            answer = result["answer"]
            retrieved_ids = [doc["id"] for doc in result["documents"]]

            relevancy = answer_relevancy(query, answer, expected_keywords)
            precision = (
                context_precision(retrieved_ids, relevant_docs)
                if relevant_docs
                else None
            )
            recall = (
                context_recall(retrieved_ids, relevant_docs)
                if relevant_docs
                else None
            )
            reflection_score = result.get("reflection", {}).get("score", None)

            # Store results
            results.append(
                {
                    "query": query,
                    "answer": answer[:100] + "...",
                    "num_docs": len(result["documents"]),
                    "relevancy": relevancy,
                    "precision": precision,
                    "recall": recall,
                    "reflection_score": reflection_score,
                }
            )

            # Update metrics
            metrics.add_query_result(
                num_docs=len(result["documents"]),
                reflection_score=reflection_score,
            )

        # Display results
        _display_results(results, metrics)

    except Exception as e:
        console.print(f"\n[bold red]✗ Evaluation failed: {e}[/bold red]\n")
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


def _display_results(results: List[Dict], metrics: RAGMetrics):
    """Display evaluation results in a table."""
    console.print("\n[bold green]Evaluation Results[/bold green]\n")

    # Create table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Query", style="cyan", width=40)
    table.add_column("Relevancy", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Reflection", justify="right")

    for result in results:
        table.add_row(
            result["query"][:37] + "...",
            f"{result['relevancy']:.2f}" if result["relevancy"] else "N/A",
            f"{result['precision']:.2f}" if result["precision"] else "N/A",
            f"{result['recall']:.2f}" if result["recall"] else "N/A",
            f"{result['reflection_score']:.1f}" if result["reflection_score"] else "N/A",
        )

    console.print(table)

    # Summary statistics
    summary = metrics.get_summary()
    console.print("\n[bold]Summary Statistics:[/bold]")
    console.print(f"Total queries: {summary['total_queries']}")
    console.print(f"Avg docs per query: {summary['avg_docs_per_query']:.1f}")
    console.print(f"Avg reflection score: {summary['avg_reflection_score']:.1f}/10\n")


def _create_sample_test_file(path: Path):
    """Create sample test queries file."""
    sample_data = {
        "queries": [
            {
                "query": "What is machine learning?",
                "expected_keywords": ["machine", "learning", "algorithm", "data"],
                "relevant_docs": [],
            },
            {
                "query": "Explain neural networks",
                "expected_keywords": ["neural", "network", "layer", "neuron"],
                "relevant_docs": [],
            },
            {
                "query": "What is the difference between supervised and unsupervised learning?",
                "expected_keywords": ["supervised", "unsupervised", "labeled", "data"],
                "relevant_docs": [],
            },
        ]
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sample_data, f, indent=2)


if __name__ == "__main__":
    app()
