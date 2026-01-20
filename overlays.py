"""
Overlay abstraction layer for Tesla dashcam video processor.

Provides a base class for renderable overlays with consistent positioning
and composition behavior, enabling plugin-style extensibility.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
import numpy as np

from constants import OVERLAY_OPACITY, CANVAS_OPACITY


class Overlay(ABC):
    """
    Abstract base class for all overlay renderers.

    Overlays are semi-transparent visual elements rendered on top of
    the composite video frame. Each overlay has a fixed size and position,
    with configurable opacity.

    Subclasses must implement:
        - render(data): Generate the overlay image from input data
        - size: Property returning (width, height) tuple

    Example:
        class SpeedometerOverlay(Overlay):
            @property
            def size(self) -> Tuple[int, int]:
                return (200, 200)

            def render(self, meta: SeiMetadata) -> np.ndarray:
                img = np.zeros((200, 200, 3), dtype=np.uint8)
                # Draw speedometer...
                return img

        overlay = SpeedometerOverlay(position=(100, 50))
        canvas = overlay.compose(canvas, meta)
    """

    def __init__(
        self,
        position: Tuple[int, int] = (0, 0),
        overlay_weight: float = OVERLAY_OPACITY,
        canvas_weight: float = CANVAS_OPACITY,
    ):
        """
        Initialize overlay with position and blending weights.

        Args:
            position: (x, y) top-left position on canvas
            overlay_weight: Alpha weight for overlay in blending (0-1)
            canvas_weight: Alpha weight for canvas in blending (0-1)
        """
        self._position = position
        self._overlay_weight = overlay_weight
        self._canvas_weight = canvas_weight

    @property
    def position(self) -> Tuple[int, int]:
        """Top-left (x, y) position on canvas."""
        return self._position

    @position.setter
    def position(self, value: Tuple[int, int]):
        self._position = value

    @property
    @abstractmethod
    def size(self) -> Tuple[int, int]:
        """Return (width, height) of the overlay."""
        pass

    @abstractmethod
    def render(self, data: Any) -> np.ndarray:
        """
        Render the overlay image from input data.

        Args:
            data: Input data for rendering (e.g., SeiMetadata, GPS coords)

        Returns:
            RGB image array of shape (height, width, 3)
        """
        pass

    def compose(self, canvas: np.ndarray, data: Any) -> np.ndarray:
        """
        Render and blend overlay onto canvas.

        This is a convenience method that calls render() and applies the
        overlay to the canvas at the configured position.

        Args:
            canvas: Target RGB image to overlay onto
            data: Input data passed to render()

        Returns:
            Modified canvas with overlay applied
        """
        overlay_img = self.render(data)
        self._apply_overlay(canvas, overlay_img)
        return canvas

    def _apply_overlay(self, canvas: np.ndarray, overlay: np.ndarray) -> None:
        """
        Blend overlay onto canvas at the configured position.

        Uses numpy for alpha blending. Overlay is skipped
        if it extends outside canvas bounds.

        Args:
            canvas: Target image (modified in-place)
            overlay: Overlay image to blend
        """
        x, y = self._position
        oh, ow = overlay.shape[:2]
        ch, cw = canvas.shape[:2]

        # Bounds check - skip if overlay would extend outside canvas
        if y < 0 or x < 0 or y + oh > ch or x + ow > cw:
            return

        roi = canvas[y:y + oh, x:x + ow]
        # Alpha blending using numpy (equivalent to cv2.addWeighted)
        blended = (roi.astype(np.float32) * self._canvas_weight +
                   overlay.astype(np.float32) * self._overlay_weight).astype(np.uint8)
        canvas[y:y + oh, x:x + ow] = blended


class OverlayRegistry:
    """
    Registry for managing multiple overlays.

    Provides ordered rendering of overlays onto a canvas, useful for
    building composite displays with multiple visual elements.

    Example:
        registry = OverlayRegistry()
        registry.register('dashboard', DashboardOverlay(position=(760, 20)))
        registry.register('map', MapOverlay(position=(1600, 20)))

        canvas = registry.compose_all(canvas, metadata)
    """

    def __init__(self):
        self._overlays: dict[str, Overlay] = {}
        self._order: list[str] = []

    def register(self, name: str, overlay: Overlay) -> None:
        """
        Register an overlay with a unique name.

        Args:
            name: Unique identifier for this overlay
            overlay: Overlay instance to register
        """
        if name not in self._overlays:
            self._order.append(name)
        self._overlays[name] = overlay

    def unregister(self, name: str) -> Optional[Overlay]:
        """
        Remove an overlay by name.

        Args:
            name: Identifier of overlay to remove

        Returns:
            The removed overlay, or None if not found
        """
        if name in self._overlays:
            self._order.remove(name)
            return self._overlays.pop(name)
        return None

    def get(self, name: str) -> Optional[Overlay]:
        """Get an overlay by name."""
        return self._overlays.get(name)

    def compose_all(self, canvas: np.ndarray, data: Any) -> np.ndarray:
        """
        Render all registered overlays onto canvas in registration order.

        Args:
            canvas: Target RGB image
            data: Input data passed to each overlay's render()

        Returns:
            Modified canvas with all overlays applied
        """
        for name in self._order:
            overlay = self._overlays[name]
            overlay.compose(canvas, data)
        return canvas

    def __len__(self) -> int:
        return len(self._overlays)

    def __iter__(self):
        for name in self._order:
            yield name, self._overlays[name]
