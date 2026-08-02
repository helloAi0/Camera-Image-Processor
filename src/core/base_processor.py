from abc import ABC, abstractmethod
import time
from typing import Dict, Any, Tuple
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("BaseProcessor")


class BaseProcessor(ABC):
    """Abstract Base Class (ABC) serving as the contract for all image processing modules."""

    def __init__(self, name: str = "BaseProcessor") -> None:
        """Initializes the processor base with a identifier name.

        Args:
            name (str): Identifier name for logging and pipeline tracing.
        """
        self.name: str = name
        self.last_execution_time_ms: float = 0.0

    @abstractmethod
    def process(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Abstract method that must be overridden by all concrete processor subclasses.

        Args:
            image (np.ndarray): Input image array (H, W, C) or (H, W).
            **kwargs (Any): Dynamic keyword arguments specific to individual algorithms.

        Returns:
            np.ndarray: Processed output image array.
        """
        pass

    def validate_image(self, image: np.ndarray) -> bool:
        """Validates input image matrix format, dimension integrity, and data type.

        Args:
            image (np.ndarray): NumPy array to validate.

        Returns:
            bool: True if image is valid, raises ValueError otherwise.
        """
        if image is None:
            raise ValueError(f"[{self.name}] Input image cannot be None.")

        if not isinstance(image, np.ndarray):
            raise TypeError(f"[{self.name}] Expected np.ndarray, received {type(image)}.")

        if image.size == 0:
            raise ValueError(f"[{self.name}] Input image array is empty (size 0).")

        if image.ndim not in (2, 3):
            raise ValueError(
                f"[{self.name}] Image must be 2D (Grayscale) or 3D (Color). Got shape {image.shape}."
            )

        return True

    def execute_with_timing(self, image: np.ndarray, **kwargs: Any) -> Tuple[np.ndarray, float]:
        """Executes the concrete process() implementation while benchmarking performance.

        Args:
            image (np.ndarray): Input image array.
            **kwargs (Any): Algorithm parameters.

        Returns:
            Tuple[np.ndarray, float]: Processed image array and execution duration in milliseconds.
        """
        self.validate_image(image)

        start_time = time.perf_counter()
        result = self.process(image, **kwargs)
        end_time = time.perf_counter()

        self.last_execution_time_ms = (end_time - start_time) * 1000.0
        logger.debug(f"[{self.name}] Executed in {self.last_execution_time_ms:.3f} ms")

        return result, self.last_execution_time_ms