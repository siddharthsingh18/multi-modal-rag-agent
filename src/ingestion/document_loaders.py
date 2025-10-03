"""Document loaders for various file formats."""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
from bs4 import BeautifulSoup
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf

from ..observability.logging import get_logger
from ..utils.exceptions import IngestionError

logger = get_logger(__name__)


class Document:
    """Document container with content and metadata."""

    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ):
        """
        Initialize document.

        Args:
            content: Document text content
            metadata: Document metadata
            doc_id: Unique document identifier
        """
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id or self.metadata.get("source", "unknown")

    def __repr__(self) -> str:
        return f"Document(id={self.doc_id}, length={len(self.content)})"


class DocumentLoader:
    """Base document loader."""

    async def load(self, source: str) -> List[Document]:
        """
        Load documents from source.

        Args:
            source: File path or URL

        Returns:
            List of documents
        """
        raise NotImplementedError


class PDFLoader(DocumentLoader):
    """PDF document loader using unstructured."""

    async def load(self, file_path: str) -> List[Document]:
        """
        Load PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            List of documents (one per page)

        Raises:
            IngestionError: If loading fails
        """
        try:
            logger.info(f"Loading PDF: {file_path}")

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            elements = await loop.run_in_executor(
                None,
                lambda: partition_pdf(
                    filename=file_path,
                    strategy="auto",
                    extract_images_in_pdf=False,
                ),
            )

            # Group elements by page
            documents = []
            current_page = 1
            page_content = []

            for element in elements:
                element_page = getattr(element.metadata, "page_number", 1)

                if element_page != current_page:
                    # Save previous page
                    if page_content:
                        documents.append(
                            Document(
                                content="\n".join(page_content),
                                metadata={
                                    "source": file_path,
                                    "page": current_page,
                                    "type": "pdf",
                                },
                            )
                        )
                    page_content = []
                    current_page = element_page

                page_content.append(str(element))

            # Save last page
            if page_content:
                documents.append(
                    Document(
                        content="\n".join(page_content),
                        metadata={
                            "source": file_path,
                            "page": current_page,
                            "type": "pdf",
                        },
                    )
                )

            logger.info(f"Loaded {len(documents)} pages from PDF")
            return documents

        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            raise IngestionError(
                f"Failed to load PDF {file_path}",
                original_error=e,
            )


class TextLoader(DocumentLoader):
    """Plain text file loader."""

    async def load(self, file_path: str) -> List[Document]:
        """
        Load text file.

        Args:
            file_path: Path to text file

        Returns:
            List with single document

        Raises:
            IngestionError: If loading fails
        """
        try:
            logger.info(f"Loading text file: {file_path}")

            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()

            return [
                Document(
                    content=content,
                    metadata={
                        "source": file_path,
                        "type": "text",
                    },
                )
            ]

        except Exception as e:
            logger.error(f"Failed to load text file: {e}")
            raise IngestionError(
                f"Failed to load text file {file_path}",
                original_error=e,
            )


class MarkdownLoader(DocumentLoader):
    """Markdown file loader."""

    async def load(self, file_path: str) -> List[Document]:
        """
        Load markdown file.

        Args:
            file_path: Path to markdown file

        Returns:
            List with single document

        Raises:
            IngestionError: If loading fails
        """
        try:
            logger.info(f"Loading markdown file: {file_path}")

            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()

            return [
                Document(
                    content=content,
                    metadata={
                        "source": file_path,
                        "type": "markdown",
                    },
                )
            ]

        except Exception as e:
            logger.error(f"Failed to load markdown file: {e}")
            raise IngestionError(
                f"Failed to load markdown file {file_path}",
                original_error=e,
            )


class HTMLLoader(DocumentLoader):
    """HTML file loader with BeautifulSoup."""

    async def load(self, file_path: str) -> List[Document]:
        """
        Load HTML file.

        Args:
            file_path: Path to HTML file

        Returns:
            List with single document

        Raises:
            IngestionError: If loading fails
        """
        try:
            logger.info(f"Loading HTML file: {file_path}")

            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                html_content = await f.read()

            # Parse HTML and extract text
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style tags
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text(separator="\n", strip=True)

            return [
                Document(
                    content=text,
                    metadata={
                        "source": file_path,
                        "type": "html",
                    },
                )
            ]

        except Exception as e:
            logger.error(f"Failed to load HTML file: {e}")
            raise IngestionError(
                f"Failed to load HTML file {file_path}",
                original_error=e,
            )


class DirectoryLoader:
    """Load all documents from a directory."""

    def __init__(self, recursive: bool = True):
        """
        Initialize directory loader.

        Args:
            recursive: Whether to load files recursively
        """
        self.recursive = recursive
        self.loaders = {
            ".pdf": PDFLoader(),
            ".txt": TextLoader(),
            ".md": MarkdownLoader(),
            ".html": HTMLLoader(),
            ".htm": HTMLLoader(),
        }

    async def load(self, directory: str) -> List[Document]:
        """
        Load all supported documents from directory.

        Args:
            directory: Path to directory

        Returns:
            List of all documents

        Raises:
            IngestionError: If loading fails
        """
        try:
            logger.info(f"Loading documents from directory: {directory}")

            path = Path(directory)
            if not path.is_dir():
                raise IngestionError(f"Not a directory: {directory}")

            pattern = "**/*" if self.recursive else "*"
            files = list(path.glob(pattern))

            all_documents = []
            for file_path in files:
                if file_path.is_file():
                    suffix = file_path.suffix.lower()
                    if suffix in self.loaders:
                        try:
                            loader = self.loaders[suffix]
                            documents = await loader.load(str(file_path))
                            all_documents.extend(documents)
                        except Exception as e:
                            logger.warning(f"Failed to load {file_path}: {e}")

            logger.info(f"Loaded {len(all_documents)} documents from directory")
            return all_documents

        except Exception as e:
            logger.error(f"Failed to load directory: {e}")
            raise IngestionError(
                f"Failed to load directory {directory}",
                original_error=e,
            )


def get_loader(file_path: str) -> DocumentLoader:
    """
    Get appropriate loader for file type.

    Args:
        file_path: Path to file

    Returns:
        Document loader instance

    Raises:
        IngestionError: If file type not supported
    """
    suffix = Path(file_path).suffix.lower()

    loaders = {
        ".pdf": PDFLoader(),
        ".txt": TextLoader(),
        ".md": MarkdownLoader(),
        ".html": HTMLLoader(),
        ".htm": HTMLLoader(),
    }

    if suffix not in loaders:
        raise IngestionError(f"Unsupported file type: {suffix}")

    return loaders[suffix]
