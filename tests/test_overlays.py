"""
Tests for Overlay base class and OverlayRegistry.

Tests the abstract overlay pattern and registry for managing multiple overlays.
"""

import pytest
import numpy as np
from typing import Tuple, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from overlays import Overlay, OverlayRegistry


class SimpleOverlay(Overlay):
    """Concrete overlay for testing."""

    def __init__(self, width: int = 100, height: int = 50, **kwargs):
        super().__init__(**kwargs)
        self._width = width
        self._height = height
        self.render_count = 0

    @property
    def size(self) -> Tuple[int, int]:
        return (self._width, self._height)

    def render(self, data: Any) -> np.ndarray:
        self.render_count += 1
        img = np.ones((self._height, self._width, 3), dtype=np.uint8) * 255
        return img


class TestOverlay:
    """Tests for Overlay base class."""

    def test_position_default(self):
        """Default position should be (0, 0)."""
        overlay = SimpleOverlay()
        assert overlay.position == (0, 0)

    def test_position_custom(self):
        """Custom position should be stored."""
        overlay = SimpleOverlay(position=(100, 50))
        assert overlay.position == (100, 50)

    def test_position_setter(self):
        """Position should be mutable."""
        overlay = SimpleOverlay()
        overlay.position = (200, 100)
        assert overlay.position == (200, 100)

    def test_size_property(self):
        """Size should return (width, height)."""
        overlay = SimpleOverlay(width=150, height=75)
        assert overlay.size == (150, 75)

    def test_render_called(self):
        """Render should be called."""
        overlay = SimpleOverlay()
        result = overlay.render(None)
        assert overlay.render_count == 1
        assert result.shape == (50, 100, 3)

    def test_compose_applies_overlay(self):
        """Compose should apply overlay to canvas."""
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        overlay = SimpleOverlay(width=50, height=30, position=(10, 10))

        result = overlay.compose(canvas, None)

        # Overlay region should be modified (blended white)
        roi = result[10:40, 10:60]
        assert np.any(roi > 0), "ROI should be modified by overlay"

    def test_compose_outside_bounds_ignored(self):
        """Overlay outside canvas bounds should be ignored."""
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay = SimpleOverlay(width=50, height=30, position=(200, 200))

        result = overlay.compose(canvas, None)

        assert np.all(result == 0), "Canvas should be unchanged"

    def test_compose_partial_overlap_ignored(self):
        """Overlay extending past edge should be ignored."""
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay = SimpleOverlay(width=50, height=30, position=(80, 80))

        result = overlay.compose(canvas, None)

        assert np.all(result == 0), "Canvas should be unchanged for partial overlap"


class TestOverlayRegistry:
    """Tests for OverlayRegistry."""

    def test_register_overlay(self):
        """Overlay should be registered."""
        registry = OverlayRegistry()
        overlay = SimpleOverlay()

        registry.register('test', overlay)

        assert len(registry) == 1
        assert registry.get('test') is overlay

    def test_register_overwrites(self):
        """Registering same name should overwrite."""
        registry = OverlayRegistry()
        overlay1 = SimpleOverlay()
        overlay2 = SimpleOverlay()

        registry.register('test', overlay1)
        registry.register('test', overlay2)

        assert len(registry) == 1
        assert registry.get('test') is overlay2

    def test_unregister(self):
        """Unregister should remove overlay."""
        registry = OverlayRegistry()
        overlay = SimpleOverlay()
        registry.register('test', overlay)

        removed = registry.unregister('test')

        assert removed is overlay
        assert len(registry) == 0
        assert registry.get('test') is None

    def test_unregister_missing(self):
        """Unregister missing key should return None."""
        registry = OverlayRegistry()

        removed = registry.unregister('missing')

        assert removed is None

    def test_compose_all(self):
        """Compose all should apply all overlays in order."""
        registry = OverlayRegistry()
        overlay1 = SimpleOverlay(width=20, height=20, position=(10, 10))
        overlay2 = SimpleOverlay(width=20, height=20, position=(50, 50))

        registry.register('first', overlay1)
        registry.register('second', overlay2)

        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        result = registry.compose_all(canvas, None)

        # Both overlays should have been rendered
        assert overlay1.render_count == 1
        assert overlay2.render_count == 1

        # Both regions should be modified
        assert np.any(result[10:30, 10:30] > 0), "First overlay region modified"
        assert np.any(result[50:70, 50:70] > 0), "Second overlay region modified"

    def test_iteration_order(self):
        """Iteration should preserve registration order."""
        registry = OverlayRegistry()
        registry.register('alpha', SimpleOverlay())
        registry.register('beta', SimpleOverlay())
        registry.register('gamma', SimpleOverlay())

        names = [name for name, _ in registry]

        assert names == ['alpha', 'beta', 'gamma']

    def test_empty_registry(self):
        """Empty registry should have zero length."""
        registry = OverlayRegistry()
        assert len(registry) == 0
