"""
PDF Processing Module
Converts PDFs to images for OCR processing
"""
import io
from pathlib import Path
from typing import List, Optional, Dict, Any
import fitz  # PyMuPDF
import img2pdf
from PIL import Image
from tqdm import tqdm


class PDFProcessor:
    """
    Handles PDF to image conversion and image to PDF conversion
    """

    def __init__(self, dpi: int = 144):
        """
        Initialize PDF processor

        Args:
            dpi: Resolution for PDF to image conversion (default: 144)
                 Higher DPI = better quality but larger files
                 Common values: 72 (low), 144 (medium), 300 (high)
        """
        self.dpi = dpi

    def pdf_to_images(
        self,
        pdf_path: str,
        output_format: str = "PNG"
    ) -> List[Image.Image]:
        """
        Convert PDF to list of PIL Images

        Args:
            pdf_path: Path to PDF file
            output_format: Output image format (PNG or JPEG)

        Returns:
            List of PIL Image objects, one per page
        """
        images = []

        try:
            pdf_document = fitz.open(pdf_path)
            zoom = self.dpi / 72.0  # 72 DPI is the default
            matrix = fitz.Matrix(zoom, zoom)

            print(f"📄 Converting PDF: {Path(pdf_path).name}")
            print(f"   Pages: {pdf_document.page_count}")
            print(f"   DPI: {self.dpi}")

            for page_num in tqdm(range(pdf_document.page_count), desc="Converting pages"):
                page = pdf_document[page_num]

                # Render page to pixmap
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)

                # Disable PIL's image size limit
                Image.MAX_IMAGE_PIXELS = None

                if output_format.upper() == "PNG":
                    img_data = pixmap.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                else:
                    # Convert to JPEG (requires RGB)
                    img_data = pixmap.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))

                    if img.mode in ('RGBA', 'LA'):
                        # Create white background for transparency
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'RGBA':
                            background.paste(img, mask=img.split()[-1])
                        else:
                            background.paste(img, mask=None)
                        img = background

                images.append(img)

            pdf_document.close()
            print(f"✅ Converted {len(images)} pages successfully")

            return images

        except Exception as e:
            print(f"❌ Error converting PDF: {e}")
            return []

    def save_images(
        self,
        images: List[Image.Image],
        output_dir: str,
        prefix: str = "page",
        format: str = "PNG"
    ) -> List[str]:
        """
        Save images to disk

        Args:
            images: List of PIL Image objects
            output_dir: Directory to save images
            prefix: Filename prefix (default: "page")
            format: Image format (PNG or JPEG)

        Returns:
            List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for idx, img in enumerate(images):
            filename = f"{prefix}_{idx + 1:04d}.{format.lower()}"
            filepath = output_path / filename

            if format.upper() == "JPEG" and img.mode != "RGB":
                img = img.convert("RGB")

            img.save(filepath, format=format)
            saved_paths.append(str(filepath))

        print(f"✅ Saved {len(saved_paths)} images to {output_dir}")
        return saved_paths

    def images_to_pdf(
        self,
        images: List[Image.Image],
        output_path: str,
        quality: int = 95
    ) -> bool:
        """
        Convert list of images to a single PDF

        Args:
            images: List of PIL Image objects
            output_path: Output PDF file path
            quality: JPEG quality (1-100, default: 95)

        Returns:
            True if successful, False otherwise
        """
        if not images:
            print("❌ No images to convert")
            return False

        image_bytes_list = []

        try:
            for img in images:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Save to bytes buffer
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='JPEG', quality=quality)
                img_bytes = img_buffer.getvalue()
                image_bytes_list.append(img_bytes)

            # Convert to PDF
            pdf_bytes = img2pdf.convert(image_bytes_list)

            # Write to file
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)

            print(f"✅ Created PDF: {output_path}")
            return True

        except Exception as e:
            print(f"❌ Error creating PDF: {e}")
            return False

    def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """
        Get metadata and information about a PDF

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with PDF information
        """
        try:
            pdf_document = fitz.open(pdf_path)

            info = {
                "filename": Path(pdf_path).name,
                "page_count": pdf_document.page_count,
                "metadata": pdf_document.metadata,
                "is_encrypted": pdf_document.is_encrypted,
                "page_sizes": [],
            }

            # Get size of each page
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                rect = page.rect
                info["page_sizes"].append({
                    "page": page_num + 1,
                    "width": rect.width,
                    "height": rect.height,
                })

            pdf_document.close()
            return info

        except Exception as e:
            return {
                "error": str(e),
                "filename": Path(pdf_path).name
            }

    def extract_page_range(
        self,
        pdf_path: str,
        start_page: int,
        end_page: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Extract specific page range from PDF

        Args:
            pdf_path: Path to PDF file
            start_page: Starting page number (1-indexed)
            end_page: Ending page number (1-indexed, None = last page)

        Returns:
            List of PIL Image objects for the specified range
        """
        try:
            pdf_document = fitz.open(pdf_path)
            total_pages = pdf_document.page_count

            # Validate page range
            start_idx = max(0, start_page - 1)
            end_idx = min(total_pages, end_page) if end_page else total_pages

            if start_idx >= end_idx:
                print(f"❌ Invalid page range: {start_page} to {end_page}")
                return []

            zoom = self.dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            images = []

            print(f"📄 Extracting pages {start_page} to {end_idx}")

            for page_num in range(start_idx, end_idx):
                page = pdf_document[page_num]
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)

                Image.MAX_IMAGE_PIXELS = None
                img_data = pixmap.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                images.append(img)

            pdf_document.close()
            print(f"✅ Extracted {len(images)} pages")

            return images

        except Exception as e:
            print(f"❌ Error extracting pages: {e}")
            return []

    def split_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        pages_per_split: int = 10
    ) -> List[str]:
        """
        Split a large PDF into smaller PDFs

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory for output PDFs
            pages_per_split: Number of pages per output PDF

        Returns:
            List of paths to split PDF files
        """
        try:
            pdf_document = fitz.open(pdf_path)
            total_pages = pdf_document.page_count

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            base_name = Path(pdf_path).stem
            split_pdfs = []

            for i in range(0, total_pages, pages_per_split):
                # Create new PDF for this split
                new_pdf = fitz.open()

                end_page = min(i + pages_per_split, total_pages)

                for page_num in range(i, end_page):
                    new_pdf.insert_pdf(
                        pdf_document,
                        from_page=page_num,
                        to_page=page_num
                    )

                # Save split PDF
                split_filename = f"{base_name}_part_{i // pages_per_split + 1}.pdf"
                split_path = output_path / split_filename
                new_pdf.save(str(split_path))
                new_pdf.close()

                split_pdfs.append(str(split_path))
                print(f"✅ Created: {split_filename} (pages {i + 1}-{end_page})")

            pdf_document.close()
            print(f"✅ Split PDF into {len(split_pdfs)} files")

            return split_pdfs

        except Exception as e:
            print(f"❌ Error splitting PDF: {e}")
            return []


# Convenience functions
def convert_pdf_to_images(pdf_path: str, dpi: int = 144) -> List[Image.Image]:
    """
    Quick function to convert PDF to images

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution (default: 144)

    Returns:
        List of PIL Images
    """
    processor = PDFProcessor(dpi=dpi)
    return processor.pdf_to_images(pdf_path)


def get_pdf_page_count(pdf_path: str) -> int:
    """
    Get number of pages in a PDF

    Args:
        pdf_path: Path to PDF file

    Returns:
        Number of pages
    """
    try:
        pdf_document = fitz.open(pdf_path)
        count = pdf_document.page_count
        pdf_document.close()
        return count
    except Exception:
        return 0


if __name__ == "__main__":
    # Test PDF processor
    print("PDF Processor Test")
    print("=" * 50)

    # Example usage
    # processor = PDFProcessor(dpi=144)
    # images = processor.pdf_to_images("sample.pdf")
    # processor.save_images(images, "output", prefix="page")
