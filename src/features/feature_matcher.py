import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.core.base_processor import BaseProcessor
from src.utils.logger import setup_logger

logger = setup_logger("FeatureMatcher", log_level=logging.DEBUG)


class FeatureMatcher(BaseProcessor):
    """Production Feature Detection, Keypoint Extraction, and Descriptor Matching Processor."""

    def __init__(self) -> None:
        """Initializes the FeatureMatcher processor."""
        super().__init__(name="FeatureMatcher")

    def _ensure_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Helper method converting multi-channel images to single-channel 8-bit grayscale.

        Args:
            image (np.ndarray): Input image array.

        Returns:
            np.ndarray: Single-channel grayscale image array.
        """
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def detect_harris_corners(
        self,
        image: np.ndarray,
        block_size: int = 2,
        ksize: int = 3,
        k: float = 0.04,
        threshold: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detects corners using the Harris Corner Response function.

        Args:
            image (np.ndarray): Input image array.
            block_size (int): Neighborhood size considered for corner detection.
            ksize (int): Aperture parameter for Sobel derivative kernel.
            k (float): Harris detector free parameter in the equation.
            threshold (float): Threshold factor relative to max response for filtering corners.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - corner_map (np.ndarray): Raw float32 Harris response matrix.
                - corners_mask (np.ndarray): Binary uint8 mask of detected corner positions.
        """
        self.validate_image(image)
        gray = self._ensure_grayscale(image)

        gray_f32 = np.float32(gray)
        dst = cv2.cornerHarris(gray_f32, block_size, ksize, k)

        # Dilate response map to mark corner points clearly
        dst_dilated = cv2.dilate(dst, None)

        corners_mask = np.zeros_like(gray, dtype=np.uint8)
        corners_mask[dst_dilated > threshold * dst_dilated.max()] = 255

        return dst, corners_mask

    def extract_features(
        self,
        image: np.ndarray,
        algorithm: str = "orb",
        max_features: int = 500,
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Extracts keypoints and computes high-dimensional descriptors.

        Args:
            image (np.ndarray): Input image array.
            algorithm (str): Feature algorithm ('orb', 'sift', 'fast').
            max_features (int): Maximum number of keypoints to retain.

        Returns:
            Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]: Keypoints list and descriptor matrix.
        """
        self.validate_image(image)
        gray = self._ensure_grayscale(image)
        algo_lower = algorithm.lower()

        if algo_lower == "orb":
            detector = cv2.ORB_create(nfeatures=max_features)
            keypoints, descriptors = detector.detectAndCompute(gray, None)

        elif algo_lower == "sift":
            detector = cv2.SIFT_create(nfeatures=max_features)
            keypoints, descriptors = detector.detectAndCompute(gray, None)

        elif algo_lower == "fast":
            fast = cv2.FastFeatureDetector_create()
            keypoints = fast.detect(gray, None)
            # Use ORB descriptor extractor on FAST keypoints
            orb = cv2.ORB_create()
            keypoints, descriptors = orb.compute(gray, keypoints)

        else:
            raise ValueError(f"Unsupported feature extraction algorithm '{algorithm}'. Choose 'orb', 'sift', or 'fast'.")

        keypoints = keypoints[:max_features] if keypoints else []
        descriptors = descriptors[:max_features] if descriptors is not None else None

        logger.debug(f"Extracted {len(keypoints)} keypoints using {algorithm.upper()}")
        return keypoints, descriptors

    def match_descriptors(
        self,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray,
        matcher_type: str = "bf",
        norm_type: Optional[int] = None,
        ratio_threshold: float = 0.75,
    ) -> List[cv2.DMatch]:
        """Matches descriptors between two images using KNN with Lowe's Ratio Test.

        Args:
            descriptors1 (np.ndarray): Query image descriptor matrix.
            descriptors2 (np.ndarray): Train image descriptor matrix.
            matcher_type (str): Matcher backend ('bf' for Brute-Force, 'flann' for FLANN).
            norm_type (Optional[int]): Distance metric (cv2.NORM_HAMMING or cv2.NORM_L2).
            ratio_threshold (float): Lowe's ratio test threshold (d1/d2 < ratio_threshold).

        Returns:
            List[cv2.DMatch]: Filtered list of robust matches.
        """
        if descriptors1 is None or descriptors2 is None or len(descriptors1) == 0 or len(descriptors2) == 0:
            logger.warning("One or both descriptor sets are empty. Returning zero matches.")
            return []

        # Auto-select norm type if not specified based on descriptor type
        if norm_type is None:
            if descriptors1.dtype == np.uint8:
                norm_type = cv2.NORM_HAMMING
            else:
                norm_type = cv2.NORM_L2

        if matcher_type.lower() == "bf":
            matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        elif matcher_type.lower() == "flann":
            if norm_type == cv2.NORM_HAMMING:
                # Index parameters for binary descriptors (ORB)
                index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            else:
                # Index parameters for float descriptors (SIFT)
                index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unsupported matcher type '{matcher_type}'. Choose 'bf' or 'flann'.")

        # Perform k-Nearest Neighbors matching with k=2 for ratio test
        raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Apply David Lowe's Ratio Test filtering
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        logger.debug(f"Matched {len(good_matches)} valid pairs out of {len(raw_matches)} candidates.")
        return good_matches

    def draw_matches_canvas(
        self,
        image1: np.ndarray,
        kp1: List[cv2.KeyPoint],
        image2: np.ndarray,
        kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        max_draw: int = 50,
    ) -> np.ndarray:
        """Generates visual side-by-side match connections.

        Args:
            image1 (np.ndarray): Query image.
            kp1 (List[cv2.KeyPoint]): Query keypoints.
            image2 (np.ndarray): Train image.
            kp2 (List[cv2.KeyPoint]): Train keypoints.
            matches (List[cv2.DMatch]): Matched descriptor pairs.
            max_draw (int): Maximum matches to render visually for clarity.

        Returns:
            np.ndarray: Side-by-side visualization canvas showing matched feature lines.
        """
        self.validate_image(image1)
        self.validate_image(image2)

        sorted_matches = sorted(matches, key=lambda x: x.distance)[:max_draw]

        canvas = cv2.drawMatches(
            image1,
            kp1,
            image2,
            kp2,
            sorted_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        return canvas

    def process(self, image: np.ndarray, mode: str = "features", **kwargs: Any) -> Any:
        """Executes feature detection based on mode selection.

        Args:
            image (np.ndarray): Input image array.
            mode (str): Mode: 'harris', 'features', 'sift', 'orb'.
            **kwargs (Any): Additional parameters.

        Returns:
            Any: Detection results depending on mode.
        """
        self.validate_image(image)
        mode_lower = mode.lower()

        if mode_lower == "harris":
            block_size = kwargs.get("block_size", 2)
            ksize = kwargs.get("ksize", 3)
            k = kwargs.get("k", 0.04)
            return self.detect_harris_corners(image, block_size=block_size, ksize=ksize, k=k)

        elif mode_lower in ["features", "orb", "sift", "fast"]:
            algo = mode_lower if mode_lower != "features" else kwargs.get("algorithm", "orb")
            max_f = kwargs.get("max_features", 500)
            return self.extract_features(image, algorithm=algo, max_features=max_f)

        else:
            logger.warning(f"Unknown mode '{mode}'. Returning original image.")
            return image


if __name__ == "__main__":
    matcher_engine = FeatureMatcher()

    # Synthetic reference image (200x200 uint8 square with internal shape)
    ref_img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(ref_img, (30, 30), (170, 170), (255, 255, 255), -1)
    cv2.circle(ref_img, (100, 100), 30, (0, 0, 0), -1)

    # Synthetic target image (Rotated 45 degrees + translated)
    M = cv2.getRotationMatrix2D((100, 100), 45, 1.0)
    target_img = cv2.warpAffine(ref_img, M, (200, 200))

    # 1. Test Harris Corner Detection
    _, harris_mask = matcher_engine.detect_harris_corners(ref_img)

    # 2. Extract ORB Features
    kp1, des1 = matcher_engine.extract_features(ref_img, algorithm="orb", max_features=200)
    kp2, des2 = matcher_engine.extract_features(target_img, algorithm="orb", max_features=200)

    # 3. Match Features using BFMatcher + Lowe's Ratio Test
    good_matches = matcher_engine.match_descriptors(des1, des2, matcher_type="bf", ratio_threshold=0.8)

    # 4. Render Visualization Canvas
    match_canvas = matcher_engine.draw_matches_canvas(ref_img, kp1, target_img, kp2, good_matches)

    logger.info(f"Harris corners detected: {np.sum(harris_mask > 0)} pixels")
    logger.info(f"Image 1 Keypoints: {len(kp1)} | Image 2 Keypoints: {len(kp2)}")
    logger.info(f"Robust Matches found: {len(good_matches)}")
    logger.info(f"Match canvas generated with shape: {match_canvas.shape}")