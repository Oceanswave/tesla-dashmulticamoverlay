"""
Rich console configuration for Tesla dashcam video processor.

Provides beautiful terminal output with progress bars, panels, and styled logging.
"""

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

# Custom theme for Tesla-inspired styling
TESLA_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "highlight": "bold magenta",
    "muted": "dim",
    "camera": "bold blue",
    "speed": "bold cyan",
    "gps": "green",
})

# Global console instance
console = Console(theme=TESLA_THEME)


def setup_rich_logging(verbose: bool = False) -> None:
    """
    Configure logging to use Rich handler for beautiful output.

    Args:
        verbose: Enable DEBUG level logging with full details
    """
    level = logging.DEBUG if verbose else logging.WARNING

    # Configure root logger with Rich handler
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                show_time=verbose,
                show_path=verbose,
                rich_tracebacks=True,
                tracebacks_show_locals=verbose,
                markup=True,
            )
        ],
        force=True,  # Override any existing configuration
    )


def create_progress() -> Progress:
    """
    Create a Rich progress bar with Tesla-styled columns.

    Returns:
        Configured Progress instance for tracking multi-step operations
    """
    return Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, style="cyan", complete_style="green"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def create_scan_progress() -> Progress:
    """
    Create a lighter progress bar for quick scanning operations.

    Returns:
        Configured Progress instance for GPS scanning
    """
    return Progress(
        SpinnerColumn("dots"),
        TextColumn("[cyan]{task.description}"),
        BarColumn(bar_width=30, style="dim cyan", complete_style="cyan"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def create_concat_progress() -> Progress:
    """
    Create a progress bar for FFmpeg concatenation with status tracking.

    Returns:
        Configured Progress instance for concatenation
    """
    return Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, style="cyan", complete_style="green"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[status]}[/]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def create_render_progress() -> Progress:
    """
    Create a progress bar for rendering with clip and frame tracking.

    Returns:
        Configured Progress instance for rendering
    """
    return Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, style="cyan", complete_style="green"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[status]}[/]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def print_banner(version: str = "1.0.0") -> None:
    """
    Print a styled startup banner.

    Args:
        version: Version string to display
    """
    banner = """
[bold cyan]████████╗███████╗███████╗██╗      █████╗ [/]
[bold cyan]╚══██╔══╝██╔════╝██╔════╝██║     ██╔══██╗[/]
[bold cyan]   ██║   █████╗  ███████╗██║     ███████║[/]
[bold cyan]   ██║   ██╔══╝  ╚════██║██║     ██╔══██║[/]
[bold cyan]   ██║   ███████╗███████║███████╗██║  ██║[/]
[bold cyan]   ╚═╝   ╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝[/]
[dim]Dashcam Multi-Camera Overlay Processor[/]
"""
    console.print(banner)
    console.print(f"[muted]Version {version}[/]\n")


def print_config_summary(
    clip_count: int,
    cameras: set,
    output_file: str,
    overlay_scale: float = 1.0,
    map_style: str = "simple",
    north_up: bool = False,
    layout: str = "grid",
    color_grade: str = None,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    gamma: float = 1.0,
    shadows: float = 0.0,
    highlights: float = 0.0,
) -> None:
    """
    Print a styled configuration summary panel.

    Args:
        clip_count: Number of clips to process
        cameras: Set of camera names
        output_file: Output file path
        overlay_scale: Scale factor for dashboard/map overlays
        map_style: Map background style
        north_up: Whether using north-up map orientation
        layout: Multi-camera layout mode ("grid" or "pip")
        color_grade: Color grading preset name or LUT path
        brightness: Brightness adjustment (-1.0 to 1.0)
        contrast: Contrast adjustment (-1.0 to 1.0)
        saturation: Saturation adjustment (-1.0 to 1.0)
        gamma: Gamma correction (0.1 to 3.0)
        shadows: Shadow adjustment (-1.0 to 1.0)
        highlights: Highlight adjustment (-1.0 to 1.0)
    """
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Clips", f"[highlight]{clip_count}[/]")
    table.add_row("Cameras", f"[camera]{', '.join(sorted(cameras))}[/]")
    table.add_row("Layout", f"{layout}" if layout == "grid" else f"[highlight]{layout}[/] (fullscreen + thumbnails)")
    table.add_row("Output", f"[green]{output_file}[/]")
    table.add_row("Overlay Scale", f"{overlay_scale:.1f}x")
    table.add_row("Map Style", f"{map_style}")
    table.add_row("Map Orientation", "north-up" if north_up else "heading-up")

    # Color grading settings - only show if any are active
    color_parts = []
    if color_grade:
        # Check if it's a LUT file path or a preset name
        if color_grade.endswith('.cube'):
            import os
            color_parts.append(f"LUT: {os.path.basename(color_grade)}")
        else:
            color_parts.append(f"preset: [highlight]{color_grade}[/]")
    if abs(brightness) >= 0.001:
        color_parts.append(f"brightness: {brightness:+.2f}")
    if abs(contrast) >= 0.001:
        color_parts.append(f"contrast: {contrast:+.2f}")
    if abs(saturation) >= 0.001:
        color_parts.append(f"saturation: {saturation:+.2f}")
    if abs(gamma - 1.0) >= 0.001:
        color_parts.append(f"gamma: {gamma:.2f}")
    if abs(shadows) >= 0.001:
        color_parts.append(f"shadows: {shadows:+.2f}")
    if abs(highlights) >= 0.001:
        color_parts.append(f"highlights: {highlights:+.2f}")

    if color_parts:
        table.add_row("Color Grade", ", ".join(color_parts))
    else:
        table.add_row("Color Grade", "[dim]none[/]")

    panel = Panel(
        table,
        title="[bold]Configuration[/]",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


def print_phase(phase_num: int, total_phases: int, description: str) -> None:
    """
    Print a phase header for multi-step processing.

    Args:
        phase_num: Current phase number (1-indexed)
        total_phases: Total number of phases
        description: Description of this phase
    """
    console.print(
        f"\n[bold cyan]Step {phase_num}/{total_phases}:[/] [bold]{description}[/]"
    )


def print_completion_summary(
    output_file: str,
    clip_count: int,
    total_frames: Optional[int] = None,
    gps_points: Optional[int] = None,
) -> None:
    """
    Print a styled completion summary.

    Args:
        output_file: Path to output file
        clip_count: Number of clips processed
        total_frames: Total frames processed (optional)
        gps_points: Total GPS points extracted (optional)
    """
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold green")

    table.add_row("Clips Processed", str(clip_count))
    if total_frames:
        table.add_row("Total Frames", f"{total_frames:,}")
    if gps_points:
        table.add_row("GPS Points", f"{gps_points:,}")
    table.add_row("Output", output_file)

    panel = Panel(
        table,
        title="[bold green]Complete[/]",
        border_style="green",
        padding=(1, 2),
    )
    console.print()
    console.print(panel)


def print_error(message: str, hint: Optional[str] = None) -> None:
    """
    Print a styled error message.

    Args:
        message: Error message
        hint: Optional hint for resolution
    """
    console.print(f"\n[error]Error:[/] {message}")
    if hint:
        console.print(f"[muted]Hint: {hint}[/]")
