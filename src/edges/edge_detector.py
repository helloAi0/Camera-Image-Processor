import logging
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from src.core.base_processor import BaseProcessor
from src.utils.logger import setup_logger

logger = setup_logger("EdgeDetector", log_level=logging.DEBUG)


class EdgeDetector(BaseProcessor):
    """Production Edge Detection and Spatial Gradient Analysis Processor supporting Sobel, Laplacian, and Canny algorithms."""

    def __init__(self) -> None:
        """Initializes the EdgeDetector processor."""
        super().__init__(name="EdgeDetector")

    def _ensure_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Helper method converting multi-channel RGB/BGR matrices to single-channel 8-bit grayscale.

        Args:
            image (np.ndarray): Input NumPy image matrix.

        Returns:
            np.ndarray: Single-channel grayscale image array.
        """
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def compute_sobel_gradients(
        self,
        image: np.ndarray,
        ksize: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculates 1st-order spatial derivatives (Gx, Gy), Gradient Magnitude, and Direction Angle.

        Args:
            image (np.ndarray): Input image matrix.
            ksize (int): Size of the extended Sobel kernel. Must be 1, 3, 5, or 7. Default is 3.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - grad_x (np.ndarray): Horizontal intensity gradient (float32).
                - grad_y (np.ndarray): Vertical intensity gradient (float32).
                - magnitude (np.ndarray): Normalized gradient magnitude (uint8).
                - angle (np.ndarray): Gradient orientation in radians [-pi, pi] (float32).
        """
        self.validate_image(image)
        gray = self._ensure_grayscale(image)

        # Compute Gx and Gy using CV_32F precision to prevent negative value truncation
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)

        # Compute Euclidean Magnitude: M = sqrt(Gx^2 + Gy^2)
        magnitude_raw, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=False)

        # Normalize magnitude array safely to uint8 range [0, 255]
        magnitude = cv2.normalize(magnitude_raw, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return grad_x, grad_y, magnitude, angle

    def compute_laplacian(
        self,
        image: np.ndarray,
        ksize: int = 3,
    ) -> np.ndarray:
        """Calculates 2nd-order isotropic spatial derivative (Laplacian) to locate zero-crossings.

        Args:
            image (np.ndarray): Input image array.
            ksize (int): Aperture size used to compute second-derivative filters. Default is 3.

        Returns:
            np.ndarray: Normalized 8-bit Laplacian edge response array.
        """
        self.validate_image(image)
        gray = self._ensure_grayscale(image)

        # Compute second derivative using float64 to capture signed zero-crossings safely
        laplacian_raw = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)

        # Convert back to uint8 using absolute scaling
        laplacian = cv2.convertScaleAbs(laplacian_raw)
        return laplacian

    def apply_canny(
        self,
        image: np.ndarray,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        sigma: float = 0.33,
    ) -> np.ndarray:
        """Applies multi-stage Canny Edge Detection with automatic median thresholding fallback.

        Args:
            image (np.ndarray): Input image array.
            low_threshold (Optional[float]): Hysteresis lower bound threshold.
            high_threshold (Optional[float]): Hysteresis upper bound threshold.
            sigma (float): Empirical multiplier for automatic threshold calculation. Default is 0.33.

        Returns:
            np.ndarray: Binary single-pixel thin edge mask (uint8: 0 or 255).
        """
        self.validate_image(image)
        gray = self._ensure_grayscale(image)

        # Auto-compute dynamic thresholds if not explicitly provided using median image intensity
        if low_threshold is None or high_threshold is None:
            v = np.median(gray)
            low_threshold = int(max(0, (1.0 - sigma) * v))
            high_threshold = int(min(255, (1.0 + sigma) * v))
            logger.debug(f"Auto-calculated Canny thresholds: Low={low_threshold}, High={high_threshold}")

        # Execute 5-stage Canny Edge Algorithm
        canny_edges = cv2.Canny(gray, threshold1=int(low_threshold), threshold2=int(high_threshold))
        return canny_edges

    def process(self, image: np.ndarray, mode: str = "canny", **kwargs: Any) -> np.ndarray:
        """Executes edge detection based on mode selection.

        Args:
            image (np.ndarray): Input image matrix.
            mode (str): Mode: 'sobel', 'laplacian', 'canny'.
            **kwargs (Any): Parameters passed to specific edge algorithms.

        Returns:
            np.ndarray: Edge response matrix.
        """
        self.validate_image(image)
        mode_lower = mode.lower()

        if mode_lower == "sobel":
            ksize = kwargs.get("ksize", 3)
            _, _, magnitude, _ = self.compute_sobel_gradients(image, ksize=ksize)
            return magnitude

        elif mode_lower == "laplacian":
            ksize = kwargs.get("ksize", 3)
            return self.compute_laplacian(image, ksize=ksize)

        elif mode_lower == "canny":
            low_t = kwargs.get("low_threshold", None)
            high_t = kwargs.get("high_threshold", None)
            sigma = kwargs.get("sigma", 0.33)
            return self.apply_canny(image, low_threshold=low_t, high_threshold=high_t, sigma=sigma)

        else:
            logger.warning(f"Unknown edge detection mode '{mode}'. Returning original image.")
            return image


if __name__ == "__main__":
    detector = EdgeDetector()

    # Generate synthetic geometric test matrix (400x400 uint8 circle)
    test_img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(test_img, (200, 200), 80, (255, 255, 255), -1)

    # 1. Test Sobel Gradient Extraction
    grad_x, grad_y, mag, angle = detector.compute_sobel_gradients(test_img, ksize=3)
    
    # 2. Test Laplacian Operator
    lap_img, t_lap = detector.execute_with_timing(test_img, mode="laplacian", ksize=3)

    # 3. Test Canny Edge Detector (Auto Dynamic Thresholding)
    canny_img, t_canny = detector.execute_with_timing(test_img, mode="canny", sigma=0.33)

    logger.info(f"Sobel Magnitude shape: {mag.shape}")
    logger.info(f"Laplacian completed in: {t_lap:.3f} ms")
    logger.info(f"Canny Edge Detector completed in: {t_canny:.3f} ms")