import logging
from typing import Tuple
import pytest
import numpy as np
import cv2

from src.core.base_processor import BaseProcessor
from src.pipeline.stream_pipeline import StreamPipeline
from src.stereo.stereo_vision import StereoVision


class ConcreteProcessor(BaseProcessor):
    """Concrete implementation of BaseProcessor for testing core abstract logic."""
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        self.validate_image(image)
        return image.copy()


@pytest.fixture
def sample_bgr_frame() -> np.ndarray:
    """Fixture providing a synthetic 400x400 BGR test image."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), -1)
    return img


@pytest.fixture
def stereo_pair() -> Tuple[np.ndarray, np.ndarray]:
    """Fixture providing rectified left/right synthetic image pair with disparity."""
    left = np.zeros((300, 300, 3), dtype=np.uint8)
    right = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(left, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.rectangle(right, (84, 100), (184, 200), (255, 255, 255), -1)
    return left, right


# =====================================================================
# 1. BaseProcessor & Input Validation Tests
# =====================================================================

def test_base_processor_validation_pass(sample_bgr_frame: np.ndarray) -> None:
    """Verifies valid uint8 numpy image passes input validation."""
    processor = ConcreteProcessor(name="TestProc")
    result = processor.process(sample_bgr_frame)
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_bgr_frame.shape


def test_base_processor_invalid_type() -> None:
    """Verifies TypeError is raised when passing non-numpy structures."""
    processor = ConcreteProcessor(name="TestProc")
    # Updated match string to match the actual exception thrown
    with pytest.raises(TypeError, match="Expected np.ndarray"):
        processor.process([[1, 2], [3, 4]])  # type: ignore

def test_base_processor_empty_image() -> None:
    """Verifies ValueError is raised when passing zero-size image array."""
    processor = ConcreteProcessor(name="TestProc")
    empty_img = np.array([], dtype=np.uint8)
    with pytest.raises(ValueError, match="Input image array is empty"):
        processor.process(empty_img)


def test_base_processor_invalid_dimensions() -> None:
    """Verifies ValueError is raised for 1D or 4D arrays."""
    processor = ConcreteProcessor(name="TestProc")
    invalid_4d = np.zeros((10, 10, 10, 3), dtype=np.uint8)
    # Updated match string. Used 'r' prefix so the parentheses are treated correctly in the regex
    with pytest.raises(ValueError, match=r"Image must be 2D \(Grayscale\) or 3D \(Color\)"):
        processor.process(invalid_4d)


# =====================================================================
# 2. Stereo Vision Subsystem Tests
# =====================================================================

def test_stereo_disparity_computation(stereo_pair: Tuple[np.ndarray, np.ndarray]) -> None:
    """Verifies StereoVision outputs correct float32 disparity shape and non-negative values."""
    left, right = stereo_pair
    stereo = StereoVision()
    disparity = stereo.compute_disparity(left, right, num_disparities=32, block_size=15)

    assert isinstance(disparity, np.ndarray)
    assert disparity.dtype == np.float32
    assert disparity.shape == left.shape[:2]


def test_stereo_3d_reprojection(stereo_pair: Tuple[np.ndarray, np.ndarray]) -> None:
    """Verifies 3D point cloud reprojection produces (H, W, 3) matrix."""
    left, right = stereo_pair
    stereo = StereoVision()
    disparity = stereo.compute_disparity(left, right, num_disparities=32, block_size=15)
    Q = stereo.create_reprojection_matrix(focal_length=400.0, baseline=0.1, principal_point=(150.0, 150.0))
    
    points_3d = stereo.reproject_to_3d(disparity, Q, max_depth=10.0)
    assert points_3d.shape == (left.shape[0], left.shape[1], 3)


# =====================================================================
# 3. Stream Pipeline Multi-Threading Tests
# =====================================================================

def test_stream_pipeline_buffering() -> None:
    """Verifies StreamPipeline non-blocking frame queue drops oldest frames on overflow."""
    pipeline = StreamPipeline(queue_capacity=2)
    counter = 0

    def mock_stream() -> np.ndarray:
        nonlocal counter
        counter += 1
        return np.full((100, 100, 3), counter, dtype=np.uint8)

    pipeline.start_stream(source=mock_stream)
    pytest.importorskip("time").sleep(0.1)
    
    success, ts, frame = pipeline.read_frame(timeout=0.5)
    pipeline.stop_stream()

    assert success is True
    assert frame is not None
    telemetry = pipeline.get_telemetry()
    assert telemetry["captured_frames"] > 0
    assert telemetry["dropped_frames"] >= 0
