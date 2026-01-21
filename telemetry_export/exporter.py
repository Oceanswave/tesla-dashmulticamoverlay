"""
Telemetry exporter orchestrator for Tesla dashcam data.

Coordinates extraction of SEI metadata from video clips and export
to various formats (GPX, FIT, JSON).
"""

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Union

from telemetry_export.data_models import TelemetryRecord, TelemetryTrack
from telemetry_export.gpx_writer import write_gpx
from telemetry_export.fit_writer import write_fit, write_fit_with_json_sidecar

logger = logging.getLogger(__name__)

# Tesla dashcam frame rate
TESLA_FPS = 29.97


def parse_tesla_timestamp(timestamp_prefix: str) -> Optional[datetime]:
    """
    Parse Tesla dashcam timestamp prefix to datetime.

    Args:
        timestamp_prefix: Format like '2026-01-09_11-45-38'

    Returns:
        datetime in UTC, or None if parsing fails
    """
    try:
        # Tesla format: YYYY-MM-DD_HH-MM-SS
        if '_' in timestamp_prefix:
            date_part, time_part = timestamp_prefix.split('_')
            time_formatted = time_part.replace('-', ':')
            dt_str = f"{date_part} {time_formatted}"
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        pass
    return None


class TelemetryExporter:
    """
    Extracts and exports Tesla dashcam telemetry to various formats.

    Usage:
        from main import discover_clips
        clips = discover_clips('./dashcam_data')
        exporter = TelemetryExporter(clips)
        track = exporter.extract_all()
        exporter.export_gpx('output.gpx')
        exporter.export_fit('output.fit')
    """

    def __init__(self, clips: List, sample_rate: float = 1.0):
        """
        Initialize exporter with clip list.

        Args:
            clips: List of ClipSet objects from discover_clips()
            sample_rate: Fraction of frames to include (1.0 = all, 0.1 = every 10th)
        """
        self.clips = clips
        self.sample_rate = sample_rate
        self._track: Optional[TelemetryTrack] = None

    def extract_all(self, progress_callback=None) -> TelemetryTrack:
        """
        Extract telemetry from all clips into a single track.

        Args:
            progress_callback: Optional callable(clip_index, total_clips) for progress

        Returns:
            TelemetryTrack containing all telemetry records in order
        """
        from sei_parser import extract_sei_data

        records = []
        source_clips = []
        track_name = None

        for clip_idx, clip in enumerate(self.clips):
            if progress_callback:
                progress_callback(clip_idx, len(self.clips))

            # Parse clip start time from filename
            clip_start = parse_tesla_timestamp(clip.timestamp_prefix)
            if clip_start is None:
                logger.warning(f"Could not parse timestamp from {clip.timestamp_prefix}, using epoch")
                clip_start = datetime(1970, 1, 1, tzinfo=timezone.utc)

            # Use first clip's timestamp as track name
            if track_name is None:
                track_name = clip.timestamp_prefix

            source_clips.append(os.path.basename(clip.front))

            # Extract SEI metadata
            sei_data = extract_sei_data(clip.front)
            if not sei_data:
                logger.warning(f"No SEI data found in {clip.front}")
                continue

            # Convert to TelemetryRecords with timestamps
            frame_indices = sorted(sei_data.keys())

            # Apply sample rate
            if self.sample_rate < 1.0:
                step = int(1.0 / self.sample_rate)
                frame_indices = frame_indices[::step]

            for frame_idx in frame_indices:
                sei_meta = sei_data[frame_idx]

                # Calculate timestamp: clip_start + frame_idx / fps
                frame_offset = timedelta(seconds=frame_idx / TESLA_FPS)
                timestamp = clip_start + frame_offset

                try:
                    record = TelemetryRecord.from_sei_metadata(sei_meta, timestamp)
                    records.append(record)
                except Exception as e:
                    logger.debug(f"Failed to convert frame {frame_idx}: {e}")
                    continue

        self._track = TelemetryTrack(
            name=track_name or "Tesla Dashcam",
            source_clips=source_clips,
            records=records
        )

        logger.info(
            f"Extracted {len(records)} telemetry records from {len(source_clips)} clips "
            f"({self._track.duration_seconds:.1f}s, {self._track.distance_meters:.1f}m)"
        )

        return self._track

    @property
    def track(self) -> TelemetryTrack:
        """Get the extracted track, extracting if not already done."""
        if self._track is None:
            self.extract_all()
        return self._track

    def export_gpx(self, output_path: Union[str, Path]) -> str:
        """
        Export telemetry to GPX format.

        Args:
            output_path: Path for output .gpx file

        Returns:
            Path to the created file
        """
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.gpx')

        with open(output_path, 'w', encoding='utf-8') as f:
            write_gpx(self.track, f)

        logger.info(f"Exported GPX to {output_path}")
        return str(output_path)

    def export_fit(self, output_path: Union[str, Path], include_json_sidecar: bool = False) -> str:
        """
        Export telemetry to FIT format.

        Args:
            output_path: Path for output .fit file
            include_json_sidecar: If True, also create .json file with full telemetry

        Returns:
            Path to the created FIT file
        """
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.fit')

        if include_json_sidecar:
            json_path = output_path.with_suffix('.tesla.json')
            with open(output_path, 'wb') as fit_f, open(json_path, 'wb') as json_f:
                write_fit_with_json_sidecar(self.track, fit_f, json_f)
            logger.info(f"Exported FIT to {output_path} with sidecar {json_path}")
        else:
            with open(output_path, 'wb') as f:
                write_fit(self.track, f)
            logger.info(f"Exported FIT to {output_path}")

        return str(output_path)

    def export_json(self, output_path: Union[str, Path]) -> str:
        """
        Export telemetry to JSON format (all fields).

        Args:
            output_path: Path for output .json file

        Returns:
            Path to the created file
        """
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.json')

        data = {
            "name": self.track.name,
            "source_clips": self.track.source_clips,
            "duration_seconds": self.track.duration_seconds,
            "distance_meters": self.track.distance_meters,
            "record_count": len(self.track.records),
            "records": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "frame_seq_no": r.frame_seq_no,
                    "latitude_deg": r.latitude_deg,
                    "longitude_deg": r.longitude_deg,
                    "heading_deg": r.heading_deg,
                    "vehicle_speed_mps": r.vehicle_speed_mps,
                    "speed_mph": r.speed_mph,
                    "linear_acceleration_x": r.linear_acceleration_x,
                    "linear_acceleration_y": r.linear_acceleration_y,
                    "linear_acceleration_z": r.linear_acceleration_z,
                    "lateral_g": r.lateral_g,
                    "longitudinal_g": r.longitudinal_g,
                    "steering_wheel_angle": r.steering_wheel_angle,
                    "accelerator_pedal_position": r.accelerator_pedal_position,
                    "brake_applied": r.brake_applied,
                    "blinker_on_left": r.blinker_on_left,
                    "blinker_on_right": r.blinker_on_right,
                    "gear_state": r.gear_state.name,
                    "autopilot_state": r.autopilot_state.name,
                }
                for r in self.track.records
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported JSON to {output_path}")
        return str(output_path)

    def export_all(self, output_base: Union[str, Path]) -> List[str]:
        """
        Export to all supported formats (GPX, FIT, JSON).

        Args:
            output_base: Base path without extension (e.g., 'drive_2026-01-09')

        Returns:
            List of paths to created files
        """
        output_base = Path(output_base)
        created = []

        # GPX
        gpx_path = self.export_gpx(output_base.with_suffix('.gpx'))
        created.append(gpx_path)

        # FIT with JSON sidecar
        fit_path = self.export_fit(output_base.with_suffix('.fit'), include_json_sidecar=True)
        created.append(fit_path)
        created.append(str(output_base.with_suffix('.tesla.json')))

        # Full JSON
        json_path = self.export_json(output_base.with_suffix('.json'))
        created.append(json_path)

        return created


def run_export(
    input_path: str,
    output_path: Optional[str] = None,
    export_format: str = "gpx",
    sample_rate: float = 1.0,
    verbose: bool = False
) -> List[str]:
    """
    CLI entry point for telemetry export.

    Args:
        input_path: Path to dashcam file or directory
        output_path: Output file path (auto-generated if None)
        export_format: Format to export: 'gpx', 'fit', 'json', or 'all'
        sample_rate: Fraction of frames to include (1.0 = all)
        verbose: Enable verbose logging

    Returns:
        List of created file paths
    """
    # Import here to avoid circular imports
    from main import discover_clips

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Discover clips
    clips = discover_clips(input_path)
    if not clips:
        raise ValueError(f"No dashcam clips found in {input_path}")

    logger.info(f"Found {len(clips)} clip(s)")

    # Create exporter
    exporter = TelemetryExporter(clips, sample_rate=sample_rate)

    # Extract telemetry
    track = exporter.extract_all()

    # Determine output path
    if output_path is None:
        # Use first clip timestamp as base name
        base_name = clips[0].timestamp_prefix
        output_dir = os.path.dirname(input_path) if os.path.isfile(input_path) else input_path
        output_base = os.path.join(output_dir, base_name)
    else:
        output_base = os.path.splitext(output_path)[0]

    # Export based on format
    created = []
    export_format = export_format.lower()

    if export_format == "all":
        created = exporter.export_all(output_base)
    elif export_format == "gpx":
        created.append(exporter.export_gpx(f"{output_base}.gpx"))
    elif export_format == "fit":
        created.append(exporter.export_fit(f"{output_base}.fit"))
    elif export_format == "json":
        created.append(exporter.export_json(f"{output_base}.json"))
    else:
        raise ValueError(f"Unknown export format: {export_format}. Use 'gpx', 'fit', 'json', or 'all'")

    return created
