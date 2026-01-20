"""
Tests for Rich console configuration and output helpers.

Tests the console setup, progress bar creation, and styled output functions.
"""

import pytest
import logging
from io import StringIO

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich_console import (
    console,
    TESLA_THEME,
    setup_rich_logging,
    create_progress,
    create_scan_progress,
    create_concat_progress,
    create_render_progress,
    print_banner,
    print_config_summary,
    print_phase,
    print_completion_summary,
    print_error,
)


class TestConsoleSetup:
    """Tests for console initialization."""

    def test_console_exists(self):
        """Console should be initialized."""
        assert console is not None

    def test_theme_has_required_styles(self):
        """Theme should have required style definitions."""
        required_styles = ["info", "warning", "error", "success", "highlight", "camera"]
        for style in required_styles:
            assert style in TESLA_THEME.styles, f"Missing style: {style}"


class TestLogging:
    """Tests for Rich logging setup."""

    def test_setup_creates_logger(self):
        """Setup should configure root logger with WARNING level by default."""
        setup_rich_logging(verbose=False)
        logger = logging.getLogger()
        assert logger.level == logging.WARNING

    def test_verbose_sets_debug(self):
        """Verbose flag should set DEBUG level."""
        setup_rich_logging(verbose=True)
        logger = logging.getLogger()
        assert logger.level == logging.DEBUG


class TestProgressBars:
    """Tests for progress bar creation."""

    def test_create_progress(self):
        """Create progress should return Progress instance."""
        progress = create_progress()
        assert progress is not None

    def test_create_scan_progress(self):
        """Create scan progress should return Progress instance."""
        progress = create_scan_progress()
        assert progress is not None

    def test_create_concat_progress(self):
        """Create concat progress should return Progress instance with status tracking."""
        progress = create_concat_progress()
        assert progress is not None
        with progress:
            task_id = progress.add_task("Encoding", total=100, status="Loading clips...")
            progress.update(task_id, completed=50, status="45 fps • 1.5x")
            assert progress.tasks[0].completed == 50

    def test_create_render_progress(self):
        """Create render progress should return Progress instance with status tracking."""
        progress = create_render_progress()
        assert progress is not None
        with progress:
            task_id = progress.add_task("Rendering", total=10, status="Initializing...")
            progress.update(task_id, advance=1, status="45 fps • 1.5x")
            assert progress.tasks[0].completed == 1

    def test_progress_can_add_task(self):
        """Progress should support adding tasks."""
        progress = create_progress()
        with progress:
            task_id = progress.add_task("Test", total=10)
            assert task_id is not None
            progress.advance(task_id)


class TestOutputFunctions:
    """Tests for styled output functions."""

    def test_print_banner_no_error(self):
        """Print banner should not raise errors."""
        # Just verify it doesn't crash
        print_banner("1.0.0")

    def test_print_config_summary_no_error(self):
        """Print config summary should not raise errors."""
        print_config_summary(
            clip_count=5,
            cameras={"front", "back"},
            output_file="output.mp4",
            overlay_scale=1.0,
            map_style="simple",
            north_up=False,
        )

    def test_print_phase_no_error(self):
        """Print phase should not raise errors."""
        print_phase(1, 3, "Testing phase")

    def test_print_completion_summary_no_error(self):
        """Print completion summary should not raise errors."""
        print_completion_summary(
            output_file="output.mp4",
            clip_count=5,
            total_frames=1000,
            gps_points=500,
        )

    def test_print_completion_summary_optional_args(self):
        """Completion summary should work without optional args."""
        print_completion_summary(
            output_file="output.mp4",
            clip_count=5,
        )

    def test_print_error_no_error(self):
        """Print error should not raise errors."""
        print_error("Test error message")

    def test_print_error_with_hint(self):
        """Print error with hint should not raise errors."""
        print_error("Test error", hint="Try this instead")
