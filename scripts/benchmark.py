#!/usr/bin/env python3
"""Performance benchmarking script."""

import asyncio
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.rag_agent import create_rag_agent
from src.observability.logging import get_logger, setup_logging
from src.retrieval.vector_store import get_retriever

setup_logging()
logger = get_logger(__name__)
console = Console()

app = typer.Typer()


@app.command()
def benchmark(
    num_queries: int = typer.Option(10, help="Number of queries to run"),
    collection: str = typer.Option("documents", help="Collection name"),
):
    """
    Run performance benchmarks on the RAG system.

    Example:
        python scripts/benchmark.py --num-queries 20
    """
    asyncio.run(_benchmark_async(num_queries, collection))


async def _benchmark_async(num_queries: int, collection: str):
    """Async benchmark logic."""
    console.print(f"\n[bold blue]RAG System Performance Benchmark[/bold blue]")
    console.print(f"Number of queries: {num_queries}")
    console.print(f"Collection: {collection}\n")

    # Sample queries
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?",
        "How does gradient descent work?",
        "What is the difference between AI and ML?",
    ]

    try:
        # Initialize
        console.print("[yellow]Initializing RAG agent...[/yellow]")
        retriever = await get_retriever(collection_name=collection)
        agent = await create_rag_agent(retriever=retriever)

        # Warm-up
        console.print("[yellow]Running warm-up query...[/yellow]")
        await agent.run(query=queries[0])

        # Benchmark
        console.print(f"[yellow]Running {num_queries} benchmark queries...[/yellow]\n")

        times = []
        doc_counts = []

        for i in range(num_queries):
            query = queries[i % len(queries)]

            start_time = time.time()
            result = await agent.run(query=query)
            end_time = time.time()

            elapsed = end_time - start_time
            times.append(elapsed)
            doc_counts.append(len(result["documents"]))

            console.print(
                f"Query {i+1}/{num_queries}: {elapsed:.2f}s "
                f"({len(result['documents'])} docs)"
            )

        # Calculate statistics
        _display_statistics(times, doc_counts)

    except Exception as e:
        console.print(f"\n[bold red]✗ Benchmark failed: {e}[/bold red]\n")
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


def _display_statistics(times: list, doc_counts: list):
    """Display benchmark statistics."""
    console.print("\n[bold green]Benchmark Results[/bold green]\n")

    # Create table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total queries", str(len(times)))
    table.add_row("Mean response time", f"{mean(times):.2f}s")
    table.add_row("Median response time", f"{median(times):.2f}s")
    table.add_row("Min response time", f"{min(times):.2f}s")
    table.add_row("Max response time", f"{max(times):.2f}s")

    if len(times) > 1:
        table.add_row("Std dev response time", f"{stdev(times):.2f}s")

    table.add_row("Avg docs retrieved", f"{mean(doc_counts):.1f}")
    table.add_row("Throughput", f"{len(times)/sum(times):.2f} queries/s")

    console.print(table)
    console.print()


if __name__ == "__main__":
    app()
