"""
Image Processing Module
Handles image preprocessing, enhancement, and validation
"""
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np


class ImageProcessor:
    """
    Handles image preprocessing and enhancement for better OCR results
    """

    def __init__(self):
        """Initialize image processor"""
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}

    def is_valid_image(self, image_path: str) -> bool:
        """
        Check if file is a valid image

        Args:
            image_path: Path to image file

        Returns:
            True if valid image, False otherwise
        """
        path = Path(image_path)

        # Check extension
        if path.suffix.lower() not in self.supported_formats:
            return False

        # Try to open
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load and correct image orientation

        Args:
            image_path: Path to image file

        Returns:
            PIL Image object or None if error
        """
        try:
            image = Image.open(image_path)

            # Correct orientation based on EXIF data
            corrected_image = ImageOps.exif_transpose(image)

            # Convert to RGB if needed
            if corrected_image.mode not in ('RGB', 'L'):
                corrected_image = corrected_image.convert('RGB')

            return corrected_image

        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None

    def resize_image(
        self,
        image: Image.Image,
        max_size: Optional[int] = None,
        min_size: Optional[int] = None,
        maintain_aspect: bool = True
    ) -> Image.Image:
        """
        Resize image with constraints

        Args:
            image: PIL Image object
            max_size: Maximum dimension (width or height)
            min_size: Minimum dimension (width or height)
            maintain_aspect: Maintain aspect ratio

        Returns:
            Resized PIL Image
        """
        width, height = image.size

        if max_size and max(width, height) > max_size:
            if maintain_aspect:
                # Resize keeping aspect ratio
                ratio = max_size / max(width, height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
            else:
                new_width = new_height = max_size

            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"📐 Resized to {new_width}x{new_height}")

        elif min_size and min(width, height) < min_size:
            if maintain_aspect:
                ratio = min_size / min(width, height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
            else:
                new_width = new_height = min_size

            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"📐 Upscaled to {new_width}x{new_height}")

        return image

    def enhance_for_ocr(
        self,
        image: Image.Image,
        auto_contrast: bool = True,
        sharpen: bool = False,
        denoise: bool = False
    ) -> Image.Image:
        """
        Enhance image for better OCR results

        Args:
            image: PIL Image object
            auto_contrast: Apply automatic contrast adjustment
            sharpen: Apply sharpening filter
            denoise: Apply denoising filter

        Returns:
            Enhanced PIL Image
        """
        enhanced = image.copy()

        # Auto contrast
        if auto_contrast:
            enhanced = ImageOps.autocontrast(enhanced, cutoff=1)

        # Sharpen
        if sharpen:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.5)

        # Denoise (median filter)
        if denoise:
            enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))

        return enhanced

    def convert_to_grayscale(self, image: Image.Image) -> Image.Image:
        """
        Convert image to grayscale

        Args:
            image: PIL Image object

        Returns:
            Grayscale PIL Image
        """
        return image.convert('L')

    def apply_threshold(
        self,
        image: Image.Image,
        threshold: int = 128,
        adaptive: bool = False
    ) -> Image.Image:
        """
        Apply thresholding to create binary image

        Args:
            image: PIL Image object (should be grayscale)
            threshold: Threshold value (0-255)
            adaptive: Use adaptive thresholding

        Returns:
            Binary PIL Image
        """
        if image.mode != 'L':
            image = image.convert('L')

        if adaptive:
            # Simple adaptive threshold using local mean
            img_array = np.array(image)
            from scipy.ndimage import uniform_filter
            local_mean = uniform_filter(img_array, size=15)
            binary = (img_array > local_mean).astype(np.uint8) * 255
            return Image.fromarray(binary)
        else:
            # Simple threshold
            return image.point(lambda x: 255 if x > threshold else 0, mode='L')

    def remove_borders(
        self,
        image: Image.Image,
        border_size: int = 10
    ) -> Image.Image:
        """
        Remove borders from image (crop)

        Args:
            image: PIL Image object
            border_size: Size of border to remove (pixels)

        Returns:
            Cropped PIL Image
        """
        width, height = image.size

        if border_size * 2 >= min(width, height):
            print("⚠️ Border size too large, skipping")
            return image

        return image.crop((
            border_size,
            border_size,
            width - border_size,
            height - border_size
        ))

    def auto_crop(self, image: Image.Image, background_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """
        Automatically crop image to content (remove uniform background)

        Args:
            image: PIL Image object
            background_color: Background color to remove (RGB tuple)

        Returns:
            Cropped PIL Image
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Get bounding box
        bg = Image.new('RGB', image.size, background_color)
        diff = ImageOps.subtract(image, bg)
        bbox = diff.getbbox()

        if bbox:
            return image.crop(bbox)
        else:
            return image

    def rotate_image(self, image: Image.Image, angle: float) -> Image.Image:
        """
        Rotate image by specified angle

        Args:
            image: PIL Image object
            angle: Rotation angle in degrees (counter-clockwise)

        Returns:
            Rotated PIL Image
        """
        return image.rotate(angle, expand=True, fillcolor='white')

    def detect_orientation(self, image: Image.Image) -> float:
        """
        Detect image orientation (requires pytesseract)
        This is a placeholder - would need tesseract for actual implementation

        Args:
            image: PIL Image object

        Returns:
            Estimated rotation angle
        """
        # TODO: Implement with pytesseract or custom ML model
        # For now, return 0 (no rotation needed)
        return 0.0

    def get_image_stats(self, image: Image.Image) -> dict:
        """
        Get statistics about image

        Args:
            image: PIL Image object

        Returns:
            Dictionary with image statistics
        """
        img_array = np.array(image)

        stats = {
            "size": image.size,
            "mode": image.mode,
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "aspect_ratio": image.width / image.height,
        }

        # Calculate mean and std if grayscale or RGB
        if len(img_array.shape) == 2:  # Grayscale
            stats["mean_intensity"] = float(np.mean(img_array))
            stats["std_intensity"] = float(np.std(img_array))
        elif len(img_array.shape) == 3:  # RGB
            stats["mean_intensity"] = {
                "R": float(np.mean(img_array[:, :, 0])),
                "G": float(np.mean(img_array[:, :, 1])),
                "B": float(np.mean(img_array[:, :, 2])),
            }

        return stats

    def batch_load_images(self, image_paths: List[str]) -> List[Image.Image]:
        """
        Load multiple images

        Args:
            image_paths: List of image paths

        Returns:
            List of PIL Image objects
        """
        images = []
        for path in image_paths:
            img = self.load_image(path)
            if img:
                images.append(img)
            else:
                print(f"⚠️ Skipped: {path}")

        print(f"✅ Loaded {len(images)}/{len(image_paths)} images")
        return images

    def save_image(
        self,
        image: Image.Image,
        output_path: str,
        format: Optional[str] = None,
        quality: int = 95
    ) -> bool:
        """
        Save image to disk

        Args:
            image: PIL Image object
            output_path: Output file path
            format: Image format (auto-detect if None)
            quality: JPEG quality (1-100)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Auto-detect format from extension
            if format is None:
                format = Path(output_path).suffix[1:].upper()

            # Convert to RGB for JPEG
            if format == 'JPEG' and image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            # Save
            if format == 'JPEG':
                image.save(output_path, format=format, quality=quality, optimize=True)
            else:
                image.save(output_path, format=format)

            return True

        except Exception as e:
            print(f"❌ Error saving image: {e}")
            return False

    def create_thumbnail(
        self,
        image: Image.Image,
        size: Tuple[int, int] = (200, 200)
    ) -> Image.Image:
        """
        Create thumbnail of image

        Args:
            image: PIL Image object
            size: Thumbnail size (width, height)

        Returns:
            Thumbnail PIL Image
        """
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        return thumbnail


# Convenience functions
def quick_load(image_path: str) -> Optional[Image.Image]:
    """Quick load image with orientation correction"""
    processor = ImageProcessor()
    return processor.load_image(image_path)


def quick_enhance(image: Image.Image) -> Image.Image:
    """Quick enhancement for OCR"""
    processor = ImageProcessor()
    return processor.enhance_for_ocr(image, auto_contrast=True, sharpen=True)


if __name__ == "__main__":
    # Test image processor
    print("Image Processor Test")
    print("=" * 50)

    # Example usage
    # processor = ImageProcessor()
    # img = processor.load_image("sample.jpg")
    # enhanced = processor.enhance_for_ocr(img)
    # processor.save_image(enhanced, "output.jpg")
