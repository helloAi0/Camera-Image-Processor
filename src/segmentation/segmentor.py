import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.core.base_processor import BaseProcessor
from src.utils.logger import setup_logger

logger = setup_logger("Segmentor", log_level=logging.DEBUG)


class Segmentor(BaseProcessor):
    """Production Image Segmentation Processor supporting HSV Masking, Watershed, and GrabCut algorithms."""

    def __init__(self) -> None:
        """Initializes the Segmentor processor."""
        super().__init__(name="Segmentor")

    def create_hsv_mask(
        self,
        image: np.ndarray,
        lower_bound: Tuple[int, int, int],
        upper_bound: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segments regions matching a specific color range in HSV space.

        Args:
            image (np.ndarray): Input RGB/BGR image array.
            lower_bound (Tuple[int, int, int]): Lower HSV bound (H: 0-179, S: 0-255, V: 0-255).
            upper_bound (Tuple[int, int, int]): Upper HSV bound (H: 0-179, S: 0-255, V: 0-255).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - mask (np.ndarray): Binary uint8 mask (255 inside range, 0 outside).
                - segmented (np.ndarray): Segmented foreground image matrix.
        """
        self.validate_image(image)
        canvas = image.copy()

        # Convert to HSV color space
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
        lower = np.array(lower_bound, dtype=np.uint8)
        upper = np.array(upper_bound, dtype=np.uint8)

        # Create threshold mask and apply bitwise AND
        mask = cv2.inRange(hsv, lower, upper)
        segmented = cv2.bitwise_and(canvas, canvas, mask=mask)

        logger.debug(f"Created HSV mask with lower={lower_bound}, upper={upper_bound}")
        return mask, segmented

    def apply_watershed(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs distance-transform based Watershed segmentation to separate overlapping objects.

        Args:
            image (np.ndarray): Input RGB/BGR image array.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - markers (np.ndarray): Int32 marker map (-1 for boundaries, >0 for region IDs).
                - segmented (np.ndarray): Output image with boundary lines rendered in Red.
        """
        self.validate_image(image)
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            gray = image.copy()
        else:
            bgr = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Otsu thresholding to get estimated foreground
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 2. Morphological opening to remove background noise
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # 3. Determine sure background area via dilation
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # 4. Determine sure foreground area using Euclidean Distance Transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)

        # 5. Determine unknown boundary regions
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 6. Label connected components for sure foreground
        _, markers = cv2.connectedComponents(sure_fg)

        # Shift marker IDs by 1 so background is 1 instead of 0
        markers = markers + 1
        markers[unknown == 255] = 0

        # 7. Apply Watershed algorithm
        markers_output = cv2.watershed(bgr, markers)

        # Highlight boundary edges (-1) in bright Red
        segmented = bgr.copy()
        segmented[markers_output == -1] = [0, 0, 255]

        return markers_output, segmented

    def apply_grabcut(
        self,
        image: np.ndarray,
        rect: Tuple[int, int, int, int],
        iter_count: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs iterative Graph-Cut segmentation (GrabCut) given an initial bounding box.

        Args:
            image (np.ndarray): Input RGB/BGR image array.
            rect (Tuple[int, int, int, int]): Bounding rectangle (x, y, w, h) containing foreground target.
            iter_count (int): Number of EM optimization iterations. Default is 5.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - mask (np.ndarray): Binary uint8 mask (255 for foreground, 0 for background).
                - segmented (np.ndarray): Extracted foreground image matrix.
        """
        self.validate_image(image)
        bgr = image.copy()
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

        mask = np.zeros(bgr.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        x, y, w, h = rect
        if x < 0 or y < 0 or (x + w) > bgr.shape[1] or (y + h) > bgr.shape[0]:
            raise ValueError(f"GrabCut bounding box {rect} exceeds image dimensions {bgr.shape[:2]}")

        cv2.grabCut(bgr, mask, (x, y, w, h), bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT)

        binary_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
        segmented = cv2.bitwise_and(bgr, bgr, mask=binary_mask)

        return binary_mask, segmented

    def process(self, image: np.ndarray, mode: str = "hsv", **kwargs: Any) -> Any:
        """Executes segmentation algorithm based on mode selection.

        Args:
            image (np.ndarray): Input image array.
            mode (str): Mode: 'hsv', 'watershed', 'grabcut'.
            **kwargs (Any): Additional parameters.

        Returns:
            Any: Segmentation results depending on mode.
        """
        self.validate_image(image)
        mode_lower = mode.lower()

        if mode_lower in ["hsv", "hsv_mask"]:
            lower = kwargs.get("lower_bound", (0, 50, 50))
            upper = kwargs.get("upper_bound", (10, 255, 255))
            return self.create_hsv_mask(image, lower_bound=lower, upper_bound=upper)

        elif mode_lower == "watershed":
            return self.apply_watershed(image)

        elif mode_lower == "grabcut":
            h, w = image.shape[:2]
            default_rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
            rect = kwargs.get("rect", default_rect)
            iters = kwargs.get("iter_count", 5)
            return self.apply_grabcut(image, rect=rect, iter_count=iters)

        else:
            logger.warning(f"Unknown mode '{mode}'. Returning original image.")
            return image


if __name__ == "__main__":
    segmentor = Segmentor()

    test_canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(test_canvas, (160, 200), 60, (255, 255, 255), -1)
    cv2.circle(test_canvas, (240, 200), 60, (255, 255, 255), -1)

    # 1. Test HSV Masking
    hsv_mask, hsv_seg = segmentor.create_hsv_mask(test_canvas, (0, 0, 200), (179, 30, 255))

    # 2. Test Watershed Segmentation (Proper tuple unpacking from execute_with_timing)
    (markers, watershed_canvas), t_watershed = segmentor.execute_with_timing(test_canvas, mode="watershed")

    # 3. Test GrabCut Foreground Extraction (Proper tuple unpacking from execute_with_timing)
    (gc_mask, gc_seg), t_grabcut = segmentor.execute_with_timing(
        test_canvas, mode="grabcut", rect=(80, 100, 240, 200), iter_count=5
    )

    logger.info(f"HSV Mask active pixels: {np.sum(hsv_mask > 0)}")
    logger.info(f"Watershed identified unique region IDs: {np.unique(markers)} (completed in {t_watershed:.3f} ms)")
    logger.info(f"GrabCut Mask foreground active pixels: {np.sum(gc_mask > 0)} (completed in {t_grabcut:.3f} ms)")