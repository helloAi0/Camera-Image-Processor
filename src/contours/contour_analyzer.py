import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.core.base_processor import BaseProcessor
from src.utils.logger import setup_logger

logger = setup_logger("ContourAnalyzer", log_level=logging.DEBUG)


class ContourAnalyzer(BaseProcessor):
    """Production Contour Extraction and Geometric Shape Analysis Processor."""

    def __init__(self) -> None:
        """Initializes the ContourAnalyzer processor."""
        super().__init__(name="ContourAnalyzer")

    def _ensure_binary(self, image: np.ndarray) -> np.ndarray:
        """Helper method ensuring input matrix is single-channel 8-bit binary.

        Args:
            image (np.ndarray): Input NumPy image matrix.

        Returns:
            np.ndarray: Single-channel 8-bit binary image array.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # If image contains multi-level intensities, apply dynamic Otsu thresholding
        unique_vals = np.unique(gray)
        if len(unique_vals) > 2:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        return gray

    def find_contours(
        self,
        image: np.ndarray,
        retrieval_mode: int = cv2.RETR_TREE,
        method: int = cv2.CHAIN_APPROX_SIMPLE,
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Extracts spatial contours and topological hierarchy array.

        Args:
            image (np.ndarray): Input binary or grayscale image.
            retrieval_mode (int): Contour retrieval mode (e.g., RETR_TREE, RETR_EXTERNAL).
            method (int): Approximation method (e.g., CHAIN_APPROX_SIMPLE).

        Returns:
            Tuple[List[np.ndarray], Optional[np.ndarray]]: List of contours and hierarchy array.
        """
        self.validate_image(image)
        binary = self._ensure_binary(image)

        contours, hierarchy = cv2.findContours(binary, retrieval_mode, method)
        logger.debug(f"Extracted {len(contours)} contours using retrieval mode {retrieval_mode}.")
        return list(contours), hierarchy

    def classify_shape(self, contour: np.ndarray, epsilon_factor: float = 0.04) -> str:
        """Classifies geometric shapes using Douglas-Peucker polygon approximation.

        Args:
            contour (np.ndarray): Input boundary contour.
            epsilon_factor (float): Approximation accuracy relative to perimeter. Default is 0.04.

        Returns:
            str: Shape classification string ('Triangle', 'Square', 'Rectangle', 'Pentagon', 'Circle', 'Polygon').
        """
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        vertices = len(approx)

        if vertices == 3:
            return "Triangle"
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ar = float(w) / h if h != 0 else 0.0
            return "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
        elif vertices == 5:
            return "Pentagon"
        else:
            area = cv2.contourArea(contour)
            circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
            if circularity > 0.70:
                return "Circle"
            return "Polygon"

    def analyze_contour(self, contour: np.ndarray) -> Dict[str, Any]:
        """Calculates invariant geometric metrics for a given contour curve.

        Args:
            contour (np.ndarray): Single contour boundary array.

        Returns:
            Dict[str, Any]: Dictionary containing geometric properties (area, perimeter, solidity, etc.).
        """
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, closed=True))

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0.0

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0 else 0.0

        circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        shape_label = self.classify_shape(contour)

        return {
            "area": area,
            "perimeter": perimeter,
            "bounding_box": (x, y, w, h),
            "aspect_ratio": aspect_ratio,
            "hull_area": hull_area,
            "solidity": solidity,
            "circularity": circularity,
            "enclosing_circle": ((float(cx), float(cy)), float(radius)),
            "shape": shape_label,
        }

    def draw_contour_analysis(
        self,
        image: np.ndarray,
        contours: List[np.ndarray],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Renders contours, bounding boxes, and shape label annotations on an image canvas.

        Args:
            image (np.ndarray): Original image array.
            contours (List[np.ndarray]): List of contour boundary matrices.
            color (Tuple[int, int, int]): BGR color for contour outlines. Default is Green.
            thickness (int): Outline thickness. Default is 2.

        Returns:
            np.ndarray: Annotated output image canvas.
        """
        self.validate_image(image)
        canvas = image.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        for cnt in contours:
            if len(cnt) < 3:
                continue
            metrics = self.analyze_contour(cnt)
            x, y, w, h = metrics["bounding_box"]
            shape_label = metrics["shape"]

            # Draw contour vector curve
            cv2.drawContours(canvas, [cnt], -1, color, thickness)

            # Draw axis-aligned bounding box (Blue)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 0, 0), 1)

            # Render text label (Red)
            cv2.putText(
                canvas,
                f"{shape_label}",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        return canvas

    def process(self, image: np.ndarray, mode: str = "analyze", **kwargs: Any) -> Union[np.ndarray, List[Dict[str, Any]]]:
        """Executes contour analysis based on selected mode.

        Args:
            image (np.ndarray): Input image matrix.
            mode (str): Mode: 'find', 'analyze', 'draw'.
            **kwargs (Any): Algorithm parameters.

        Returns:
            Union[np.ndarray, List[Dict[str, Any]]]: Metrics list or annotated image canvas.
        """
        self.validate_image(image)
        mode_lower = mode.lower()

        retrieval_mode = kwargs.get("retrieval_mode", cv2.RETR_TREE)
        method = kwargs.get("method", cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = self.find_contours(image, retrieval_mode=retrieval_mode, method=method)

        if mode_lower == "find":
            return contours

        elif mode_lower == "analyze":
            return [self.analyze_contour(c) for c in contours if len(c) >= 3]

        elif mode_lower == "draw":
            color = kwargs.get("color", (0, 255, 0))
            thickness = kwargs.get("thickness", 2)
            return self.draw_contour_analysis(image, contours, color=color, thickness=thickness)

        else:
            logger.warning(f"Unknown contour mode '{mode}'. Returning original image.")
            return image


if __name__ == "__main__":
    analyzer = ContourAnalyzer()

    # Generate synthetic black canvas (400x400 uint8) with distinct white geometric shapes
    test_canvas = np.zeros((400, 400, 3), dtype=np.uint8)

    # Draw synthetic shapes: Square, Circle, Triangle
    cv2.rectangle(test_canvas, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(test_canvas, (300, 100), 45, (255, 255, 255), -1)
    
    triangle_pts = np.array([[200, 350], [120, 250], [280, 250]], np.int32)
    cv2.fillPoly(test_canvas, [triangle_pts], (255, 255, 255))

    # 1. Extract contours and analyze metrics
    metrics_list, t_analysis = analyzer.execute_with_timing(test_canvas, mode="analyze")

    # 2. Render annotated image
    annotated_canvas, t_draw = analyzer.execute_with_timing(test_canvas, mode="draw")

    logger.info(f"Analyzed {len(metrics_list)} shapes in {t_analysis:.3f} ms")
    for idx, item in enumerate(metrics_list):
        logger.info(f"Shape #{idx + 1}: {item['shape']} | Area: {item['area']:.1f} | Solidity: {item['solidity']:.2f} | Circularity: {item['circularity']:.2f}")

    logger.info(f"Annotation rendering completed in {t_draw:.3f} ms")