import logging
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from src.core.base_processor import BaseProcessor
from src.utils.logger import setup_logger

logger = setup_logger("NoiseFilter", log_level=logging.DEBUG)


class NoiseFilter(BaseProcessor):
    """Production Noise Simulation and Denoising Processor supporting Gaussian noise, Salt & Pepper, and Bilateral filtering."""

    def __init__(self) -> None:
        """Initializes the NoiseFilter processor."""
        super().__init__(name="NoiseFilter")

    def add_gaussian_noise(
        self,
        image: np.ndarray,
        mean: float = 0.0,
        sigma: float = 25.0,
    ) -> np.ndarray:
        """Injects additive white Gaussian noise (AWGN) to simulate sensor electronic noise.

        Args:
            image (np.ndarray): Input NumPy image array.
            mean (float): Mean of the normal distribution. Default is 0.0.
            sigma (float): Standard deviation (spread/intensity) of noise. Default is 25.0.

        Returns:
            np.ndarray: Noisy image array with uint8 clamping.
        """
        self.validate_image(image)

        row, col, ch = image.shape if image.ndim == 3 else (*image.shape, 1)
        gauss = np.random.normal(mean, sigma, (row, col, ch)).reshape(image.shape)
        noisy = np.clip(image.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
        return noisy

    def add_salt_and_pepper_noise(
        self,
        image: np.ndarray,
        salt_probability: float = 0.02,
        pepper_probability: float = 0.02,
    ) -> np.ndarray:
        """Injects Salt & Pepper impulse noise (dead pixels or sensor bit flips).

        Args:
            image (np.ndarray): Input image array.
            salt_probability (float): Ratio of white pixels.
            pepper_probability (float): Ratio of black pixels.

        Returns:
            np.ndarray: Corrupted image array.
        """
        self.validate_image(image)
        noisy = image.copy()

        # 1. Apply Salt (White pixels across all channels)
        salt_mask = np.random.rand(*image.shape[:2]) < salt_probability
        if image.ndim == 3:
            noisy[salt_mask] = [255, 255, 255]
        else:
            noisy[salt_mask] = 255

        # 2. Apply Pepper (Black pixels across all channels)
        pepper_mask = np.random.rand(*image.shape[:2]) < pepper_probability
        if image.ndim == 3:
            noisy[pepper_mask] = [0, 0, 0]
        else:
            noisy[pepper_mask] = 0

        return noisy

    def apply_gaussian_blur(
        self,
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (5, 5),
        sigma_x: float = 0.0,
    ) -> np.ndarray:
        """Applies linear Gaussian smoothing to suppress high-frequency noise.

        Args:
            image (np.ndarray): Input image array.
            kernel_size (Tuple[int, int]): Kernel dimensions (must be odd positive integers).
            sigma_x (float): Gaussian kernel standard deviation along X axis. 0 calculates from kernel size.

        Returns:
            np.ndarray: Smoothed image array.
        """
        self.validate_image(image)

        k_w, k_h = kernel_size
        if k_w % 2 == 0 or k_h % 2 == 0:
            raise ValueError(f"Gaussian kernel dimensions must be odd integers. Got ({k_w}, {k_h})")

        smoothed = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma_x)
        return smoothed

    def apply_median_filter(
        self,
        image: np.ndarray,
        kernel_size: int = 5,
    ) -> np.ndarray:
        """Applies non-linear Median Filtering, highly effective against Salt & Pepper noise.

        Args:
            image (np.ndarray): Input image array (must be uint8).
            kernel_size (int): Aperture linear size (must be an odd integer, e.g., 3, 5, 7).

        Returns:
            np.ndarray: Filtered image array.
        """
        self.validate_image(image)

        if kernel_size % 2 == 0:
            raise ValueError(f"Median filter kernel size must be an odd integer. Got {kernel_size}")

        filtered = cv2.medianBlur(image, ksize=kernel_size)
        return filtered

    def apply_bilateral_filter(
        self,
        image: np.ndarray,
        d: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0,
    ) -> np.ndarray:
        """Applies non-linear Bilateral Filtering to smooth noise while preserving sharp edges.

        Args:
            image (np.ndarray): Input image array.
            d (int): Diameter of each pixel neighborhood.
            sigma_color (float): Filter sigma in the color space (controls color edge preservation).
            sigma_space (float): Filter sigma in the coordinate space (controls spatial smoothing).

        Returns:
            np.ndarray: Edge-preserved denoised image matrix.
        """
        self.validate_image(image)

        filtered = cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        return filtered

    def process(self, image: np.ndarray, mode: str = "bilateral", **kwargs: Any) -> np.ndarray:
        """Executes noise injection or filtering based on mode selection.

        Args:
            image (np.ndarray): Input image array.
            mode (str): Mode: 'gaussian_noise', 'salt_pepper', 'gaussian_blur', 'median', 'bilateral'.
            **kwargs (Any): Algorithm parameters.

        Returns:
            np.ndarray: Processed image array.
        """
        self.validate_image(image)
        mode_lower = mode.lower()

        if mode_lower == "gaussian_noise":
            mean = kwargs.get("mean", 0.0)
            sigma = kwargs.get("sigma", 25.0)
            return self.add_gaussian_noise(image, mean=mean, sigma=sigma)

        elif mode_lower == "salt_pepper":
            salt = kwargs.get("salt_probability", 0.02)
            pepper = kwargs.get("pepper_probability", 0.02)
            return self.add_salt_and_pepper_noise(image, salt_probability=salt, pepper_probability=pepper)

        elif mode_lower == "gaussian_blur":
            k_size = kwargs.get("kernel_size", (5, 5))
            sigma_x = kwargs.get("sigma_x", 0.0)
            return self.apply_gaussian_blur(image, kernel_size=k_size, sigma_x=sigma_x)

        elif mode_lower == "median":
            k_size = kwargs.get("kernel_size", 5)
            return self.apply_median_filter(image, kernel_size=k_size)

        elif mode_lower == "bilateral":
            d = kwargs.get("d", 9)
            s_color = kwargs.get("sigma_color", 75.0)
            s_space = kwargs.get("sigma_space", 75.0)
            return self.apply_bilateral_filter(image, d=d, sigma_color=s_color, sigma_space=s_space)

        else:
            logger.warning(f"Unknown noise/filter mode '{mode}'. Returning original image.")
            return image


if __name__ == "__main__":
    filter_engine = NoiseFilter()

    test_img = np.zeros((400, 400, 3), dtype=np.uint8)
    gradient = np.linspace(0, 255, 400, dtype=np.uint8)
    for i in range(400):
        test_img[i, :, 0] = gradient
        test_img[:, i, 1] = gradient
    test_img[:, :, 2] = 128

    noisy_g, t_ng = filter_engine.execute_with_timing(test_img, mode="gaussian_noise", sigma=30.0)
    noisy_sp, t_nsp = filter_engine.execute_with_timing(test_img, mode="salt_pepper", salt_probability=0.03, pepper_probability=0.03)
    denoise_gb, t_gb = filter_engine.execute_with_timing(noisy_g, mode="gaussian_blur", kernel_size=(5, 5))
    denoise_med, t_med = filter_engine.execute_with_timing(noisy_sp, mode="median", kernel_size=5)
    denoise_bilat, t_bilat = filter_engine.execute_with_timing(noisy_g, mode="bilateral", d=9, sigma_color=75.0, sigma_space=75.0)

    logger.info(f"Gaussian Noise Injection completed in: {t_ng:.3f} ms")
    logger.info(f"Salt & Pepper Noise Injection completed in: {t_nsp:.3f} ms")
    logger.info(f"Gaussian Blur completed in: {t_gb:.3f} ms")
    logger.info(f"Median Filter completed in: {t_med:.3f} ms")
    logger.info(f"Bilateral Filter completed in: {t_bilat:.3f} ms")