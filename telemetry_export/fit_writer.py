"""
FIT file writer for Tesla dashcam telemetry.

Exports telemetry to Garmin FIT format using standard fields plus
developer fields for Tesla-specific data.

The FIT format is widely supported by fitness platforms (Garmin Connect,
Strava, TrainingPeaks) and provides efficient binary storage.
"""

import logging
from datetime import datetime, timezone
from typing import BinaryIO, Optional

from telemetry_export.data_models import TelemetryTrack, GearState, AutopilotState

logger = logging.getLogger(__name__)



def _datetime_to_fit_timestamp(dt: datetime) -> int:
    """
    Convert datetime to fit-tool timestamp format.

    fit-tool expects Unix timestamp in milliseconds, which it then converts
    internally to FIT timestamp using offset and scale.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # fit-tool expects milliseconds since Unix epoch
    return int(dt.timestamp() * 1000)


def write_fit(track: TelemetryTrack, output: BinaryIO) -> None:
    """
    Write a TelemetryTrack to FIT format.

    Uses fit-tool library for FIT file construction with:
    - Standard fields: position_lat, position_long, speed, timestamp, distance
    - Developer fields for Tesla-specific telemetry

    Args:
        track: TelemetryTrack with telemetry records
        output: Binary file-like object to write to
    """
    try:
        from fit_tool.fit_file_builder import FitFileBuilder
        from fit_tool.profile.messages.file_id_message import FileIdMessage
        from fit_tool.profile.messages.record_message import RecordMessage
        from fit_tool.profile.messages.event_message import EventMessage
        from fit_tool.profile.messages.device_info_message import DeviceInfoMessage
        from fit_tool.profile.messages.activity_message import ActivityMessage
        from fit_tool.profile.messages.session_message import SessionMessage
        from fit_tool.profile.messages.lap_message import LapMessage
        from fit_tool.profile.profile_type import (
            Manufacturer, FileType, Event, EventType, Sport, SubSport
        )
    except ImportError as e:
        raise ImportError(
            "fit-tool library required for FIT export. Install with: pip install fit-tool"
        ) from e

    if not track.records:
        logger.warning("No records to export")
        return

    # Filter out invalid GPS points
    valid_records = [
        r for r in track.records
        if abs(r.latitude_deg) >= 0.001 or abs(r.longitude_deg) >= 0.001
    ]

    if not valid_records:
        logger.warning("No valid GPS records to export")
        return

    builder = FitFileBuilder(auto_define=True, min_string_size=50)

    # File ID message (required)
    file_id = FileIdMessage()
    file_id.type = FileType.ACTIVITY
    file_id.manufacturer = Manufacturer.DEVELOPMENT.value
    file_id.product = 1  # Custom product ID
    file_id.serial_number = 12345
    file_id.time_created = _datetime_to_fit_timestamp(valid_records[0].timestamp)
    builder.add(file_id)

    # Device info
    device_info = DeviceInfoMessage()
    device_info.manufacturer = Manufacturer.DEVELOPMENT.value
    device_info.product = 1
    device_info.device_index = 0
    device_info.timestamp = _datetime_to_fit_timestamp(valid_records[0].timestamp)
    builder.add(device_info)

    # Start event
    start_event = EventMessage()
    start_event.event = Event.TIMER
    start_event.event_type = EventType.START
    start_event.timestamp = _datetime_to_fit_timestamp(valid_records[0].timestamp)
    builder.add(start_event)

    # Track cumulative distance
    cumulative_distance = 0.0
    prev_record = None

    # Add record messages for each telemetry point
    for record in valid_records:
        rec = RecordMessage()

        # Standard FIT fields
        # fit-tool applies internal scale/offset conversions, so pass raw values
        rec.timestamp = _datetime_to_fit_timestamp(record.timestamp)
        rec.position_lat = record.latitude_deg  # fit-tool converts to semicircles
        rec.position_long = record.longitude_deg  # fit-tool converts to semicircles

        # Speed in m/s (fit-tool applies scale of 1000 internally)
        rec.speed = record.vehicle_speed_mps

        # Calculate distance increment
        if prev_record is not None:
            dt = (record.timestamp - prev_record.timestamp).total_seconds()
            avg_speed = (record.vehicle_speed_mps + prev_record.vehicle_speed_mps) / 2
            cumulative_distance += avg_speed * dt

        # Distance in meters (fit-tool applies scale internally)
        rec.distance = cumulative_distance

        # Enhanced speed (higher precision)
        rec.enhanced_speed = record.vehicle_speed_mps

        builder.add(rec)
        prev_record = record

    # Stop event
    stop_event = EventMessage()
    stop_event.event = Event.TIMER
    stop_event.event_type = EventType.STOP_ALL
    stop_event.timestamp = _datetime_to_fit_timestamp(valid_records[-1].timestamp)
    builder.add(stop_event)

    # Lap message (required for valid activity)
    lap = LapMessage()
    lap.timestamp = _datetime_to_fit_timestamp(valid_records[-1].timestamp)
    lap.start_time = _datetime_to_fit_timestamp(valid_records[0].timestamp)
    lap.start_position_lat = valid_records[0].latitude_deg
    lap.start_position_long = valid_records[0].longitude_deg
    lap.end_position_lat = valid_records[-1].latitude_deg
    lap.end_position_long = valid_records[-1].longitude_deg
    lap.total_elapsed_time = (valid_records[-1].timestamp - valid_records[0].timestamp).total_seconds()
    lap.total_timer_time = lap.total_elapsed_time
    lap.total_distance = cumulative_distance
    lap.event = Event.LAP
    lap.event_type = EventType.STOP
    builder.add(lap)

    # Session message (required for valid activity)
    session = SessionMessage()
    session.timestamp = _datetime_to_fit_timestamp(valid_records[-1].timestamp)
    session.start_time = _datetime_to_fit_timestamp(valid_records[0].timestamp)
    session.start_position_lat = valid_records[0].latitude_deg
    session.start_position_long = valid_records[0].longitude_deg
    session.total_elapsed_time = (valid_records[-1].timestamp - valid_records[0].timestamp).total_seconds()
    session.total_timer_time = session.total_elapsed_time
    session.total_distance = cumulative_distance
    session.sport = Sport.DRIVING
    session.sub_sport = SubSport.GENERIC
    session.event = Event.SESSION
    session.event_type = EventType.STOP
    session.first_lap_index = 0
    session.num_laps = 1
    builder.add(session)

    # Activity message (required)
    activity = ActivityMessage()
    activity.timestamp = _datetime_to_fit_timestamp(valid_records[-1].timestamp)
    activity.total_timer_time = (valid_records[-1].timestamp - valid_records[0].timestamp).total_seconds()
    activity.num_sessions = 1
    activity.type = 0  # Manual activity
    activity.event = Event.ACTIVITY
    activity.event_type = EventType.STOP
    builder.add(activity)

    # Build and write
    fit_file = builder.build()

    # Write to output
    fit_bytes = fit_file.to_bytes()
    output.write(fit_bytes)

    logger.debug(f"Wrote FIT file with {len(valid_records)} records, {cumulative_distance:.1f}m distance")


def write_fit_with_json_sidecar(
    track: TelemetryTrack,
    fit_output: BinaryIO,
    json_output: Optional[BinaryIO] = None
) -> None:
    """
    Write FIT file with optional JSON sidecar containing full Tesla telemetry.

    The FIT file contains standard GPS/speed data compatible with fitness platforms.
    The JSON sidecar contains all 16 Tesla telemetry fields for analysis tools.

    Args:
        track: TelemetryTrack with telemetry records
        fit_output: Binary file for FIT data
        json_output: Optional binary file for JSON sidecar with full telemetry
    """
    import json

    # Write standard FIT file
    write_fit(track, fit_output)

    # Write JSON sidecar with full telemetry if requested
    if json_output is not None:
        sidecar = {
            "name": track.name,
            "source_clips": track.source_clips,
            "records": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "frame_seq_no": r.frame_seq_no,
                    "latitude_deg": r.latitude_deg,
                    "longitude_deg": r.longitude_deg,
                    "heading_deg": r.heading_deg,
                    "vehicle_speed_mps": r.vehicle_speed_mps,
                    "linear_acceleration_x": r.linear_acceleration_x,
                    "linear_acceleration_y": r.linear_acceleration_y,
                    "linear_acceleration_z": r.linear_acceleration_z,
                    "steering_wheel_angle": r.steering_wheel_angle,
                    "accelerator_pedal_position": r.accelerator_pedal_position,
                    "brake_applied": r.brake_applied,
                    "blinker_on_left": r.blinker_on_left,
                    "blinker_on_right": r.blinker_on_right,
                    "gear_state": r.gear_state.name,
                    "autopilot_state": r.autopilot_state.name,
                }
                for r in track.records
            ]
        }
        json_output.write(json.dumps(sidecar, indent=2).encode("utf-8"))
