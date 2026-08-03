import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.core.base_processor import BaseProcessor
from src.utils.logger import setup_logger

logger = setup_logger("ISPProcessor", log_level=logging.DEBUG)


class ISPProcessor(BaseProcessor):
    """Production Image Signal Processor (ISP) supporting Auto White Balance, Color Correction, and Gamma Mapping."""

    def __init__(self) -> None:
        """Initializes the ISPProcessor."""
        super().__init__(name="ISPProcessor")

    def auto_white_balance(
        self, image: np.ndarray, method: str = "gray_world"
    ) -> np.ndarray:
        """Applies Auto White Balance (AWB) to eliminate illuminant color casts.

        Args:
            image (np.ndarray): Input BGR image array (uint8).
            method (str): AWB algorithm ('gray_world' or 'white_patch'). Default is 'gray_world'.

        Returns:
            np.ndarray: Color-balanced BGR uint8 image array.
        """
        self.validate_image(image)
        if image.ndim == 2:
            logger.warning("Single-channel grayscale image passed to AWB. Skipping.")
            return image

        canvas = image.copy().astype(np.float32)

        if method.lower() == "gray_world":
            # Calculate mean intensity for B, G, R channels
            mean_b, mean_g, mean_r = np.mean(canvas, axis=(0, 1))
            mean_gray = (mean_b + mean_g + mean_r) / 3.0

            # Scaling coefficients to match neutral gray
            kb = mean_gray / (mean_b + 1e-6)
            kg = mean_gray / (mean_g + 1e-6)
            kr = mean_gray / (mean_r + 1e-6)

            canvas[:, :, 0] *= kb
            canvas[:, :, 1] *= kg
            canvas[:, :, 2] *= kr

        elif method.lower() == "white_patch":
            # Scale max channel intensities to max target (255)
            max_b, max_g, max_r = np.max(canvas, axis=(0, 1))

            kb = 255.0 / (max_b + 1e-6)
            kg = 255.0 / (max_g + 1e-6)
            kr = 255.0 / (max_r + 1e-6)

            canvas[:, :, 0] *= kb
            canvas[:, :, 1] *= kg
            canvas[:, :, 2] *= kr
        else:
            raise ValueError(f"Unsupported AWB method '{method}'. Use 'gray_world' or 'white_patch'.")

        balanced = np.clip(canvas, 0, 255).astype(np.uint8)
        logger.debug(f"Applied AWB mode='{method}'")
        return balanced

    def apply_ccm(
        self, image: np.ndarray, matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Applies a 3x3 Color Correction Matrix (CCM) to align sensor response with sRGB space.

        Args:
            image (np.ndarray): Input BGR uint8 image array.
            matrix (Optional[np.ndarray]): 3x3 transformation matrix. Defaults to standard sRGB CCM.

        Returns:
            np.ndarray: Transformed BGR uint8 image array.
        """
        self.validate_image(image)
        if image.ndim == 2:
            logger.warning("Single-channel image passed to CCM. Skipping.")
            return image

        if matrix is None:
            # Standard sRGB sensor cross-channel linear gain calibration matrix
            matrix = np.array([
                [1.05, -0.03, -0.02],
                [-0.04, 1.08, -0.04],
                [-0.01, -0.05, 1.06]
            ], dtype=np.float32)

        if matrix.shape != (3, 3):
            raise ValueError(f"CCM matrix must have shape (3, 3), got {matrix.shape}")

        # Linear combination across pixel color channels
        corrected = cv2.transform(image, matrix)
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        logger.debug("Applied Color Correction Matrix (CCM)")
        return corrected

    def apply_gamma(self, image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """Applies non-linear tone mapping via precomputed Look-Up Table (LUT).

        Args:
            image (np.ndarray): Input uint8 image array.
            gamma (float): Target gamma factor (typical display gamma = 2.2).

        Returns:
            np.ndarray: Tone-mapped output uint8 image.
        """
        self.validate_image(image)
        if gamma <= 0:
            raise ValueError(f"Gamma factor must be > 0, got {gamma}")

        # Precompute 256-element lookup table for high performance
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype(np.uint8)

        # Apply LUT across all pixels
        tone_mapped = cv2.LUT(image, table)
        logger.debug(f"Applied Gamma Correction (gamma={gamma})")
        return tone_mapped

    def process(self, image: np.ndarray, mode: str = "full", **kwargs: Any) -> np.ndarray:
        """Executes ISP pipeline operations.

        Args:
            image (np.ndarray): Input image array.
            mode (str): Execution mode ('awb', 'ccm', 'gamma', 'full').
            **kwargs (Any): Mode-specific parameters (awb_method, ccm_matrix, gamma).

        Returns:
            np.ndarray: Processed image matrix.
        """
        self.validate_image(image)
        mode_lower = mode.lower()

        if mode_lower == "awb":
            awb_method = kwargs.get("awb_method", "gray_world")
            return self.auto_white_balance(image, method=awb_method)

        elif mode_lower == "ccm":
            matrix = kwargs.get("ccm_matrix", None)
            return self.apply_ccm(image, matrix=matrix)

        elif mode_lower == "gamma":
            gamma = kwargs.get("gamma", 2.2)
            return self.apply_gamma(image, gamma=gamma)

        elif mode_lower == "full":
            awb_method = kwargs.get("awb_method", "gray_world")
            ccm_matrix = kwargs.get("ccm_matrix", None)
            gamma = kwargs.get("gamma", 2.2)

            out = self.auto_white_balance(image, method=awb_method)
            out = self.apply_ccm(out, matrix=ccm_matrix)
            out = self.apply_gamma(out, gamma=gamma)
            return out

        else:
            logger.warning(f"Unknown ISP mode '{mode}'. Returning original image.")
            return image


if __name__ == "__main__":
    isp = ISPProcessor()

    # Generate synthetic 400x400 BGR canvas with strong warm ambient tint (High Red, Low Blue)
    test_canvas = np.full((400, 400, 3), (80, 140, 220), dtype=np.uint8)

    # 1. Test Auto White Balance (Gray World)
    awb_img = isp.auto_white_balance(test_canvas, method="gray_world")

    # 2. Test Color Correction Matrix
    ccm_img = isp.apply_ccm(awb_img)

    # 3. Test Gamma Correction
    gamma_img = isp.apply_gamma(ccm_img, gamma=2.2)

    # 4. Test Full Pipeline Execution with timing (unpacking single return matrix)
    full_out, t_full = isp.execute_with_timing(test_canvas, mode="full", gamma=2.2)

    logger.info(f"Input Average BGR: {np.mean(test_canvas, axis=(0,1)).round(2)}")
    logger.info(f"AWB Corrected Average BGR: {np.mean(awb_img, axis=(0,1)).round(2)}")
    logger.info(f"Full ISP Pipeline executed in {t_full:.3f} ms")