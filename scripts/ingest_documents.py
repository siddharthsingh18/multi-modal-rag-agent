#!/usr/bin/env python3
"""Script for batch document ingestion."""

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.chunking import DocumentChunker
from src.ingestion.document_loaders import DirectoryLoader
from src.observability.logging import get_logger, setup_logging
from src.retrieval.vector_store import get_retriever

setup_logging()
logger = get_logger(__name__)
console = Console()

app = typer.Typer()


@app.command()
def ingest(
    directory: str = typer.Argument(..., help="Directory containing documents"),
    collection: str = typer.Option("documents", help="Collection name"),
    recursive: bool = typer.Option(True, help="Recursively scan subdirectories"),
):
    """
    Ingest documents from a directory into the vector store.

    Example:
        python scripts/ingest_documents.py /path/to/docs --collection my_docs
    """
    asyncio.run(_ingest_async(directory, collection, recursive))


async def _ingest_async(directory: str, collection: str, recursive: bool):
    """Async ingestion logic."""
    console.print(f"\n[bold blue]Starting document ingestion[/bold blue]")
    console.print(f"Directory: {directory}")
    console.print(f"Collection: {collection}")
    console.print(f"Recursive: {recursive}\n")

    try:
        # Load documents
        console.print("[yellow]Loading documents...[/yellow]")
        loader = DirectoryLoader(recursive=recursive)
        documents = await loader.load(directory)

        if not documents:
            console.print("[red]No documents found![/red]")
            return

        console.print(f"[green]Loaded {len(documents)} documents[/green]")

        # Chunk documents
        console.print("[yellow]Chunking documents...[/yellow]")
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(documents)
        console.print(f"[green]Created {len(chunks)} chunks[/green]")

        # Initialize retriever
        console.print("[yellow]Initializing vector store...[/yellow]")
        retriever = await get_retriever(collection_name=collection)

        # Ingest with progress bar
        console.print("[yellow]Ingesting to vector store...[/yellow]")

        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Ingesting...", total=len(chunks)
            )

            # Batch ingestion
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]

                texts = [chunk.text for chunk in batch]
                metadata = [chunk.metadata for chunk in batch]
                ids = [chunk.chunk_id for chunk in batch]

                await retriever.add_texts(
                    texts=texts,
                    metadata=metadata,
                    ids=ids,
                )

                progress.update(task, advance=len(batch))

        console.print(
            f"\n[bold green]✓ Successfully ingested {len(documents)} documents "
            f"({len(chunks)} chunks) into collection '{collection}'[/bold green]\n"
        )

    except Exception as e:
        console.print(f"\n[bold red]✗ Ingestion failed: {e}[/bold red]\n")
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    app()
