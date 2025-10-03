"""Multi-modal document processing for images and tables."""

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytesseract
from PIL import Image
from unstructured.partition.pdf import partition_pdf

from ..observability.logging import get_logger
from ..utils.exceptions import IngestionError

logger = get_logger(__name__)


class ImageProcessor:
    """Process and extract text from images."""

    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize image processor.

        Args:
            tesseract_cmd: Path to tesseract executable
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract_text(self, image_path: str) -> str:
        """
        Extract text from image using OCR.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text

        Raises:
            IngestionError: If OCR fails
        """
        try:
            logger.debug(f"Extracting text from image: {image_path}")
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            logger.debug(f"Extracted {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            raise IngestionError(
                f"Failed to extract text from image {image_path}",
                original_error=e,
            )

    def extract_text_from_pil(self, image: Image.Image) -> str:
        """
        Extract text from PIL Image object.

        Args:
            image: PIL Image object

        Returns:
            Extracted text
        """
        try:
            text = pytesseract.image_to_string(image)
            return text

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            raise IngestionError(
                "Failed to extract text from image",
                original_error=e,
            )

    def get_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        Get image metadata.

        Args:
            image_path: Path to image file

        Returns:
            Image metadata dict
        """
        try:
            image = Image.open(image_path)
            return {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
            }

        except Exception as e:
            logger.error(f"Failed to get image metadata: {e}")
            return {}

    def encode_image_base64(self, image_path: str) -> str:
        """
        Encode image as base64 string.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image
        """
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            return base64.b64encode(image_bytes).decode("utf-8")

        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            raise IngestionError(
                f"Failed to encode image {image_path}",
                original_error=e,
            )


class TableExtractor:
    """Extract and structure tables from documents."""

    def extract_tables_from_pdf(
        self,
        pdf_path: str,
        strategy: str = "auto",
    ) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF.

        Args:
            pdf_path: Path to PDF file
            strategy: Extraction strategy ('auto', 'fast', 'hi_res')

        Returns:
            List of table dictionaries

        Raises:
            IngestionError: If extraction fails
        """
        try:
            logger.info(f"Extracting tables from PDF: {pdf_path}")

            elements = partition_pdf(
                filename=pdf_path,
                strategy=strategy,
                infer_table_structure=True,
            )

            tables = []
            for element in elements:
                if hasattr(element, "metadata") and element.metadata.category == "Table":
                    table_data = {
                        "text": str(element),
                        "page": getattr(element.metadata, "page_number", None),
                        "html": getattr(element.metadata, "text_as_html", None),
                    }
                    tables.append(table_data)

            logger.info(f"Extracted {len(tables)} tables from PDF")
            return tables

        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            raise IngestionError(
                f"Failed to extract tables from {pdf_path}",
                original_error=e,
            )

    def table_to_text(self, table_html: str) -> str:
        """
        Convert table HTML to readable text format.

        Args:
            table_html: Table in HTML format

        Returns:
            Formatted text representation
        """
        # Simple text conversion
        # For production, consider using a proper HTML parser
        text = table_html.replace("<tr>", "\n").replace("</tr>", "")
        text = text.replace("<td>", " | ").replace("</td>", "")
        text = text.replace("<th>", " | ").replace("</th>", "")
        return text.strip()


class MultiModalProcessor:
    """Process multi-modal documents with text, images, and tables."""

    def __init__(self):
        """Initialize multi-modal processor."""
        self.image_processor = ImageProcessor()
        self.table_extractor = TableExtractor()

    def process_pdf_with_images(
        self,
        pdf_path: str,
        extract_images: bool = True,
        extract_tables: bool = True,
    ) -> Dict[str, Any]:
        """
        Process PDF with image and table extraction.

        Args:
            pdf_path: Path to PDF file
            extract_images: Whether to extract images
            extract_tables: Whether to extract tables

        Returns:
            Dictionary with text, images, and tables

        Raises:
            IngestionError: If processing fails
        """
        try:
            logger.info(f"Processing multi-modal PDF: {pdf_path}")

            result = {
                "text": "",
                "images": [],
                "tables": [],
                "metadata": {"source": pdf_path},
            }

            # Extract text and images
            elements = partition_pdf(
                filename=pdf_path,
                strategy="hi_res" if extract_images else "fast",
                extract_images_in_pdf=extract_images,
                infer_table_structure=extract_tables,
            )

            text_parts = []
            for element in elements:
                element_type = (
                    element.metadata.category if hasattr(element, "metadata") else "text"
                )

                if element_type == "Table" and extract_tables:
                    table_data = {
                        "text": str(element),
                        "page": getattr(element.metadata, "page_number", None),
                    }
                    result["tables"].append(table_data)
                else:
                    text_parts.append(str(element))

            result["text"] = "\n\n".join(text_parts)

            logger.info(
                f"Processed PDF: {len(result['text'])} chars, "
                f"{len(result['tables'])} tables"
            )

            return result

        except Exception as e:
            logger.error(f"Multi-modal processing failed: {e}")
            raise IngestionError(
                f"Failed to process multi-modal PDF {pdf_path}",
                original_error=e,
            )

    def process_image_document(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image document.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with text and metadata

        Raises:
            IngestionError: If processing fails
        """
        try:
            logger.info(f"Processing image document: {image_path}")

            text = self.image_processor.extract_text(image_path)
            metadata = self.image_processor.get_image_metadata(image_path)
            metadata["source"] = image_path
            metadata["type"] = "image"

            return {
                "text": text,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Image document processing failed: {e}")
            raise IngestionError(
                f"Failed to process image document {image_path}",
                original_error=e,
            )


# Global processor instance
_multi_modal_processor: Optional[MultiModalProcessor] = None


def get_multi_modal_processor() -> MultiModalProcessor:
    """Get global multi-modal processor instance."""
    global _multi_modal_processor
    if _multi_modal_processor is None:
        _multi_modal_processor = MultiModalProcessor()
    return _multi_modal_processor
