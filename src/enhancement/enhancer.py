from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from src.core.base_processor import BaseProcessor
from src.utils.logger import setup_logger

import logging
logger = setup_logger("ImageEnhancer", log_level=logging.DEBUG)


class ImageEnhancer(BaseProcessor):
    """Production image enhancement processor supporting linear scaling, gamma correction, and CLAHE."""

    def __init__(self) -> None:
        """Initializes the ImageEnhancer processor."""
        super().__init__(name="ImageEnhancer")

    def adjust_brightness_contrast(
        self,
        image: np.ndarray,
        alpha: float = 1.0,
        beta: int = 0,
    ) -> np.ndarray:
        """Adjusts image contrast (alpha) and brightness (beta) using saturating arithmetic.

        Args:
            image (np.ndarray): Input NumPy image matrix (H, W, C) or (H, W).
            alpha (float): Gain parameter for contrast control (> 0.0). Default is 1.0.
            beta (int): Bias parameter for brightness control (-255 to 255). Default is 0.

        Returns:
            np.ndarray: Enhanced image matrix with uint8 precision.
        """
        self.validate_image(image)

        if alpha <= 0.0:
            logger.warning(f"Alpha value must be positive. Received {alpha}. Forcing alpha=0.1")
            alpha = 0.1

        # Perform vectorized linear transformation with uint8 saturation clipping:
        # dst(x,y) = saturate_cast<uchar>(alpha * src(x,y) + beta)
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced

    def apply_gamma_correction(
        self,
        image: np.ndarray,
        gamma: float = 1.0,
    ) -> np.ndarray:
        """Applies non-linear gamma correction via precalculated 256-element lookup table (LUT).

        Args:
            image (np.ndarray): Input NumPy image matrix.
            gamma (float): Gamma correction exponent.
                           gamma < 1.0 brightens shadows.
                           gamma > 1.0 darkens highlights.

        Returns:
            np.ndarray: Gamma-corrected image matrix.
        """
        self.validate_image(image)

        if gamma <= 0.0:
            logger.warning(f"Gamma must be greater than 0. Received {gamma}. Forcing gamma=0.1")
            gamma = 0.1

        # 1. Build a 256-element lookup table mapping input uint8 -> output uint8
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

        # 2. Map image intensity values through the precomputed lookup table instantly
        corrected = cv2.LUT(image, table)
        return corrected

    def equalize_histogram(self, image: np.ndarray) -> np.ndarray:
        """Applies global histogram equalization to redistribute pixel intensity frequencies.

        Args:
            image (np.ndarray): Grayscale or RGB/BGR color image array.

        Returns:
            np.ndarray: Histogram equalized image array.
        """
        self.validate_image(image)

        if image.ndim == 2:
            # Grayscale image direct global histogram equalization
            return cv2.equalizeHist(image)

        # For color images: convert to YCrCb space and equalize ONLY the Y (Luminance) channel
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        return equalized

    def apply_clahe(
        self,
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).

        Args:
            image (np.ndarray): Input image array (Grayscale or RGB).
            clip_limit (float): Threshold for contrast limiting (prevents noise over-amplification).
            tile_grid_size (Tuple[int, int]): Grid size for contextual region dividing (Rows, Cols).

        Returns:
            np.ndarray: CLAHE enhanced image matrix.
        """
        self.validate_image(image)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        if image.ndim == 2:
            return clahe.apply(image)

        # Color Image: Convert to LAB color space and apply CLAHE strictly to L (Lightness) channel
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab_planes = list(cv2.split(lab))
        lab_planes[0] = clahe.apply(lab_planes[0])
        updated_lab = cv2.merge(lab_planes)
        enhanced_rgb = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2RGB)

        return enhanced_rgb

    def process(self, image: np.ndarray, mode: str = "clahe", **kwargs: Any) -> np.ndarray:
        """Executes enhancement based on mode parameter ('brightness_contrast', 'gamma', 'equalize', 'clahe').

        Args:
            image (np.ndarray): Input NumPy image matrix.
            mode (str): Enhancement algorithm selection.
            **kwargs (Any): Algorithm parameters passed to specific enhancement functions.

        Returns:
            np.ndarray: Enhanced image matrix.
        """
        self.validate_image(image)
        mode_lower = mode.lower()

        if mode_lower == "brightness_contrast":
            alpha = kwargs.get("alpha", 1.2)
            beta = kwargs.get("beta", 10)
            return self.adjust_brightness_contrast(image, alpha=alpha, beta=beta)

        elif mode_lower == "gamma":
            gamma = kwargs.get("gamma", 1.5)
            return self.apply_gamma_correction(image, gamma=gamma)

        elif mode_lower == "equalize":
            return self.equalize_histogram(image)

        elif mode_lower == "clahe":
            clip_limit = kwargs.get("clip_limit", 2.0)
            tile_grid_size = kwargs.get("tile_grid_size", (8, 8))
            return self.apply_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

        else:
            logger.warning(f"Unknown enhancement mode '{mode}'. Returning original image.")
            return image


if __name__ == "__main__":
    # Internal validation test
    enhancer = ImageEnhancer()

    # Generate synthetic low-contrast test image (200x200 uint8 array)
    test_img = np.random.randint(80, 140, (200, 200, 3), dtype=np.uint8)

    # 1. Test Brightness and Contrast Adjustment
    bc_img, t_bc = enhancer.execute_with_timing(
        test_img, mode="brightness_contrast", alpha=1.5, beta=20
    )

    # 2. Test Gamma Correction
    gamma_img, t_g = enhancer.execute_with_timing(test_img, mode="gamma", gamma=0.5)

    # 3. Test Global Histogram Equalization
    eq_img, t_eq = enhancer.execute_with_timing(test_img, mode="equalize")

    # 4. Test CLAHE
    clahe_img, t_cl = enhancer.execute_with_timing(
        test_img, mode="clahe", clip_limit=3.0, tile_grid_size=(8, 8)
    )

    logger.debug(f"Brightness/Contrast completed in: {t_bc:.3f} ms")
    logger.debug(f"Gamma Correction completed in: {t_g:.3f} ms")
    logger.debug(f"Global Histogram Equalization completed in: {t_eq:.3f} ms")
    logger.debug(f"CLAHE completed in: {t_cl:.3f} ms")