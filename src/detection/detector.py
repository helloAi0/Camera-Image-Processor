import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.core.base_processor import BaseProcessor
from src.utils.logger import setup_logger

logger = setup_logger("ObjectDetector", log_level=logging.DEBUG)


class ObjectDetector(BaseProcessor):
    """Production Classical Object Detection Engine supporting Contour Analytics, Bounding Geometry, and NMS."""

    def __init__(self) -> None:
        """Initializes the ObjectDetector processor."""
        super().__init__(name="ObjectDetector")

    def extract_contours(
        self,
        binary_image: np.ndarray,
        min_area: float = 100.0,
        max_area: Optional[float] = None,
    ) -> List[np.ndarray]:
        """Extracts and filters continuous contours from a binary image mask.

        Args:
            binary_image (np.ndarray): Single-channel binary image (uint8).
            min_area (float): Minimum area threshold in pixels to filter noise.
            max_area (Optional[float]): Maximum area threshold in pixels.

        Returns:
            List[np.ndarray]: List of filtered contour array coordinates.
        """
        self.validate_image(binary_image)
        if binary_image.ndim == 3:
            gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = binary_image.copy()

        # Find external contours using Topological Structural Analysis
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours: List[np.ndarray] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
            filtered_contours.append(cnt)

        logger.debug(f"Extracted {len(filtered_contours)} contours (from total {len(contours)})")
        return filtered_contours

    def compute_bounding_geometry(
        self, contour: np.ndarray, geometry_type: str = "aabb"
    ) -> Dict[str, Any]:
        """Computes bounding spatial geometry representations for a contour.

        Args:
            contour (np.ndarray): Contour coordinate array.
            geometry_type (str): Geometry model ('aabb', 'obb', 'circle', 'hull').

        Returns:
            Dict[str, Any]: Geometric descriptors dictionary.
        """
        gtype = geometry_type.lower()

        if gtype == "aabb":
            # Axis-Aligned Bounding Box (x, y, w, h)
            x, y, w, h = cv2.boundingRect(contour)
            return {"type": "aabb", "box": (x, y, w, h), "area": float(w * h)}

        elif gtype == "obb":
            # Oriented Bounding Box (RotatedRect: center(x,y), size(w,h), angle)
            rect = cv2.minAreaRect(contour)
            box_points = cv2.boxPoints(rect).astype(np.int32)
            return {"type": "obb", "rect": rect, "box_points": box_points}

        elif gtype == "circle":
            # Minimum Enclosing Circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            return {"type": "circle", "center": (int(x), int(y)), "radius": int(radius)}

        elif gtype == "hull":
            # Convex Hull boundary
            hull = cv2.convexHull(contour)
            return {"type": "hull", "contour": hull}

        else:
            raise ValueError(f"Unsupported geometry type '{geometry_type}'")

    def compute_iou(self, box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        """Computes Intersection over Union (IoU) between two Axis-Aligned Bounding Boxes.

        Args:
            box_a (Tuple[int, int, int, int]): Bounding box A (x, y, w, h).
            box_b (Tuple[int, int, int, int]): Bounding box B (x, y, w, h).

        Returns:
            float: IoU ratio ranging from 0.0 (disjoint) to 1.0 (identical).
        """
        x1, y1, w1, h1 = box_a
        x2, y2, w2, h2 = box_b

        # Compute overlap coordinates
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = float((x_right - x_left) * (y_bottom - y_top))
        box_a_area = float(w1 * h1)
        box_b_area = float(w2 * h2)

        union_area = box_a_area + box_b_area - intersection_area
        if union_area <= 0.0:
            return 0.0

        return intersection_area / union_area

    def apply_nms(
        self,
        boxes: List[Tuple[int, int, int, int]],
        scores: List[float],
        iou_threshold: float = 0.4,
    ) -> List[int]:
        """Applies Non-Maximum Suppression (NMS) to eliminate overlapping redundant candidate boxes.

        Args:
            boxes (List[Tuple[int, int, int, int]]): List of bounding boxes (x, y, w, h).
            scores (List[float]): Confidence/area scores associated with each box.
            iou_threshold (float): Overlap threshold above which redundant boxes are suppressed.

        Returns:
            List[int]: Indices of retained bounding boxes.
        """
        if not boxes:
            return []

        boxes_arr = np.array(boxes, dtype=np.float32)
        scores_arr = np.array(scores, dtype=np.float32)

        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 0] + boxes_arr[:, 2]
        y2 = boxes_arr[:, 1] + boxes_arr[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores_arr.argsort()[::-1]

        keep: List[int] = []

        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            if order.size == 1:
                break

            # Calculate pairwise overlap with remaining candidates
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            # Keep boxes with IoU less than or equal to threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        logger.debug(f"NMS suppressed {len(boxes) - len(keep)} redundant boxes")
        return keep

    def process(self, image: np.ndarray, mode: str = "detect", **kwargs: Any) -> Any:
        """Executes object detection and geometry analytics.

        Args:
            image (np.ndarray): Input binary or grayscale image array.
            mode (str): Execution mode ('detect', 'contours', 'nms').
            **kwargs (Any): Mode parameters (min_area, max_area, geometry_type, iou_thresh).

        Returns:
            Any: Detection spatial descriptors or annotated binary outputs.
        """
        self.validate_image(image)
        mode_lower = mode.lower()

        min_area = kwargs.get("min_area", 100.0)
        max_area = kwargs.get("max_area", None)
        geom_type = kwargs.get("geometry_type", "aabb")

        if mode_lower in ["detect", "contours"]:
            contours = self.extract_contours(image, min_area=min_area, max_area=max_area)
            geometries = [self.compute_bounding_geometry(c, geometry_type=geom_type) for c in contours]
            return {"contours": contours, "geometries": geometries}

        elif mode_lower == "nms":
            boxes = kwargs.get("boxes", [])
            scores = kwargs.get("scores", [])
            iou_thresh = kwargs.get("iou_threshold", 0.4)
            keep_indices = self.apply_nms(boxes, scores, iou_threshold=iou_thresh)
            return [boxes[idx] for idx in keep_indices]

        else:
            logger.warning(f"Unknown mode '{mode}'. Returning original image.")
            return image


if __name__ == "__main__":
    detector = ObjectDetector()

    # Generate synthetic 400x400 binary canvas with multiple shapes and overlapping boxes
    test_canvas = np.zeros((400, 400), dtype=np.uint8)

    # Draw synthetic foreground shapes
    cv2.rectangle(test_canvas, (50, 50), (150, 150), 255, -1)   # Square target
    cv2.circle(test_canvas, (300, 100), 40, 255, -1)             # Circle target
    cv2.rectangle(test_canvas, (180, 220), (280, 320), 255, -1) # Square target 2

    # 1. Extract contours and bounding geometry
    results, t_detect = detector.execute_with_timing(
        test_canvas, mode="detect", min_area=200.0, geometry_type="aabb"
    )

    # 2. Test Non-Maximum Suppression with overlapping candidate proposals
    candidate_boxes = [
        (50, 50, 100, 100),    # Target 1 exact
        (52, 48, 98, 102),     # Overlapping candidate 1 (high IoU)
        (300, 100, 80, 80),    # Target 2 exact
        (298, 102, 82, 78),    # Overlapping candidate 2 (high IoU)
    ]
    candidate_scores = [0.95, 0.88, 0.92, 0.85]

    nms_boxes, t_nms = detector.execute_with_timing(
        test_canvas, mode="nms", boxes=candidate_boxes, scores=candidate_scores, iou_threshold=0.4
    )

    logger.info(f"Extracted {len(results['contours'])} object contours in {t_detect:.3f} ms")
    logger.info(f"NMS suppressed candidate boxes from {len(candidate_boxes)} down to {len(nms_boxes)} in {t_nms:.3f} ms")