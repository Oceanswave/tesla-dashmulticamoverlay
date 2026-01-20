"""
Pytest configuration and fixtures for Tesla dashcam processor tests.

Provides reusable test fixtures for video configuration, mock SEI data,
and sample frame generation.
"""

import pytest
import numpy as np
from typing import Dict, Set
from dataclasses import dataclass
from unittest.mock import MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import OUTPUT_WIDTH, OUTPUT_HEIGHT, ALL_CAMERAS


@dataclass
class MockSeiMetadata:
    """Mock SEI metadata for testing without protobuf dependency.

    Field names must match the actual protobuf schema (dashcam.proto).
    """
    vehicle_speed_mps: float = 20.0
    gear_state: int = 4  # Drive (matches gear_state in proto)
    brake_applied: bool = False  # Boolean in proto
    accelerator_pedal_position: float = 0.25  # matches proto field
    steering_wheel_angle: float = 5.0  # matches proto field
    autopilot_state: int = 1
    latitude_deg: float = 37.7749
    longitude_deg: float = -122.4194
    heading_deg: float = 45.0
    # Acceleration fields (must match proto field names exactly)
    linear_acceleration_mps2_x: float = 0.0  # Lateral (left/right)
    linear_acceleration_mps2_y: float = 0.0  # Longitudinal (accel/brake)
    linear_acceleration_mps2_z: float = 9.81  # Vertical (gravity)
    # Blinker state
    blinker_on_left: bool = False
    blinker_on_right: bool = False


@pytest.fixture
def mock_sei_metadata():
    """Fixture providing a mock SEI metadata object."""
    return MockSeiMetadata()


@pytest.fixture
def mock_sei_data_dict():
    """Fixture providing a dictionary of frame index to mock SEI metadata."""
    return {
        0: MockSeiMetadata(vehicle_speed_mps=0.0, gear_state=1),  # Park
        1: MockSeiMetadata(vehicle_speed_mps=5.0, gear_state=4),  # Drive slow
        2: MockSeiMetadata(vehicle_speed_mps=20.0, gear_state=4),  # Drive normal
        3: MockSeiMetadata(vehicle_speed_mps=35.0, gear_state=4, autopilot_state=2),  # Autopilot
    }


@pytest.fixture
def sample_frame():
    """Fixture providing a sample BGR frame matching front camera dimensions."""
    # Front camera is typically 1280x960
    return np.zeros((960, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_canvas():
    """Fixture providing a blank output canvas at target resolution."""
    return np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)


@pytest.fixture
def all_cameras() -> Set[str]:
    """Fixture providing the full set of valid camera names."""
    return set(ALL_CAMERAS)


@pytest.fixture
def front_only_cameras() -> Set[str]:
    """Fixture providing front-only camera set."""
    return {"front"}


@pytest.fixture
def standard_cameras() -> Set[str]:
    """Fixture providing typical 4-camera setup (front, back, repeaters)."""
    return {"front", "back", "left_repeater", "right_repeater"}


@pytest.fixture
def temp_video_path(tmp_path):
    """Fixture providing a temporary path for video output."""
    return str(tmp_path / "test_output.mp4")


@pytest.fixture
def mock_video_capture():
    """Fixture providing a mock VideoCapture object."""
    mock = MagicMock()
    mock.isOpened.return_value = True
    mock.get.return_value = 30.0  # FPS
    mock.read.return_value = (True, np.zeros((960, 1280, 3), dtype=np.uint8))
    return mock


@pytest.fixture
def sample_gps_history():
    """Fixture providing sample GPS coordinate history."""
    return [
        (37.7749, -122.4194),
        (37.7750, -122.4190),
        (37.7752, -122.4185),
        (37.7755, -122.4180),
    ]
