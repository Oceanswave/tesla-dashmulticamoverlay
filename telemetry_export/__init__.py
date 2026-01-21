"""
Telemetry export module for Tesla dashcam SEI metadata.

Exports embedded telemetry data to GPX and FIT formats for use with
GPS tracking software, fitness apps, and mapping tools.
"""

from telemetry_export.data_models import (
    GearState,
    AutopilotState,
    TelemetryRecord,
    TelemetryTrack,
)
from telemetry_export.exporter import TelemetryExporter

__all__ = [
    "GearState",
    "AutopilotState",
    "TelemetryRecord",
    "TelemetryTrack",
    "TelemetryExporter",
]
