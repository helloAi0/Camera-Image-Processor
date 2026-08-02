from pathlib import Path
from typing import Dict, Any, Union, Optional
import cv2
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("ImageIO")


class ImageIO:
    """Production Image Input/Output Engine for loading, saving, and validating images."""

    @staticmethod
    def load_image(
        file_path: Union[str, Path],
        as_grayscale: bool = False,
        convert_to_rgb: bool = True,
    ) -> np.ndarray:
        """Loads an image file from disk, validates data integrity, and manages color spaces.

        Args:
            file_path (Union[str, Path]): Path to the target image file on disk.
            as_grayscale (bool): If True, loads image in 1-channel Grayscale format.
            convert_to_rgb (bool): If True, converts OpenCV BGR default to standard RGB.

        Returns:
            np.ndarray: Validated image matrix (Height, Width, Channels) or (Height, Width).

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is corrupt or cannot be decoded into a image matrix.
        """
        path = Path(file_path)

        if not path.exists():
            logger.error(f"Image file does not exist at path: '{path.resolve()}'")
            raise FileNotFoundError(f"File not found: '{path.resolve()}'")

        # Determine OpenCV read flag
        read_flag = cv2.IMREAD_GRAYSCALE if as_grayscale else cv2.IMREAD_COLOR

        # Read image array from disk (OpenCV loads BGR by default)
        image = cv2.imread(str(path), read_flag)

        if image is None:
            logger.error(f"Failed to decode image at path: '{path.resolve()}'. File may be corrupted.")
            raise ValueError(f"Corrupt or unreadable image file: '{path.resolve()}'")

        # Handle BGR to RGB color space conversion if requested
        if not as_grayscale and convert_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        logger.info(
            f"Successfully loaded image '{path.name}' - Shape: {image.shape}, "
            f"Dtype: {image.dtype}, Memory: {image.nbytes / 1024:.2f} KB"
        )
        return image

    @staticmethod
    def save_image(
        image: np.ndarray,
        output_path: Union[str, Path],
        is_rgb: bool = True,
        quality: int = 95,
    ) -> Path:
        """Saves a NumPy image array to disk with automated directory creation and quality controls.

        Args:
            image (np.ndarray): Image array to write.
            output_path (Union[str, Path]): Target output file path.
            is_rgb (bool): If True, converts standard RGB array back to BGR for OpenCV saving.
            quality (int): Compression quality for JPEG (0-100) or PNG compression (0-9).

        Returns:
            Path: Resolved Path instance pointing to saved file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if image is None or image.size == 0:
            logger.error(f"Cannot save empty or None image matrix to '{path}'")
            raise ValueError("Attempted to save an empty image matrix.")

        save_image = image.copy()

        # Convert RGB back to OpenCV BGR color format before saving color images
        if save_image.ndim == 3 and is_rgb:
            save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)

        # Set format-specific compression flags
        params = []
        ext = path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            params = [int(cv2.IMWRITE_JPEG_QUALITY), max(0, min(100, quality))]
        elif ext == ".png":
            # Map quality (0-100) down to PNG compression levels (0-9)
            png_compression = int((100 - quality) / 100.0 * 9)
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression]

        success = cv2.imwrite(str(path), save_image, params)

        if not success:
            logger.error(f"OpenCV failed to write image to disk at '{path.resolve()}'")
            raise IOError(f"Failed to write image file to path: '{path.resolve()}'")

        logger.info(f"Successfully saved output image to '{path.resolve()}'")
        return path.resolve()

    @staticmethod
    def get_metadata(image: np.ndarray) -> Dict[str, Any]:
        """Extracts key structural metadata from a NumPy image matrix.

        Args:
            image (np.ndarray): Target image array.

        Returns:
            Dict[str, Any]: Metadata dictionary containing height, width, channels, dtype, memory.
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid image array provided for metadata extraction.")

        has_channels = image.ndim == 3
        height = image.shape[0]
        width = image.shape[1]
        channels = image.shape[2] if has_channels else 1

        return {
            "height": height,
            "width": width,
            "channels": channels,
            "aspect_ratio": round(width / float(height), 4),
            "dtype": str(image.dtype),
            "min_pixel_value": int(np.min(image)),
            "max_pixel_value": int(np.max(image)),
            "mean_pixel_value": float(round(np.mean(image), 2)),
            "std_pixel_value": float(round(np.std(image), 2)),
            "memory_size_kb": round(image.nbytes / 1024.0, 2),
        }


if __name__ == "__main__":
    # Test Module Execution
    test_out_dir = Path("output")
    test_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create a synthetic test image (200x200 RGB image with colored squares)
    synthetic_img = np.zeros((200, 200, 3), dtype=np.uint8)
    synthetic_img[0:100, 0:100] = [255, 0, 0]    # Red quadrant
    synthetic_img[0:100, 100:200] = [0, 255, 0]  # Green quadrant
    synthetic_img[100:200, 0:100] = [0, 0, 255]  # Blue quadrant
    synthetic_img[100:200, 100:200] = [255, 255, 0] # Yellow quadrant

    test_path = test_out_dir / "synthetic_test.png"

    # 2. Save synthetic image
    saved_path = ImageIO.save_image(synthetic_img, test_path, is_rgb=True)

    # 3. Reload saved image
    reloaded_img = ImageIO.load_image(saved_path, convert_to_rgb=True)

    # 4. Extract and print metadata
    metadata = ImageIO.get_metadata(reloaded_img)
    logger.debug(f"Synthetic Image Metadata: {metadata}")