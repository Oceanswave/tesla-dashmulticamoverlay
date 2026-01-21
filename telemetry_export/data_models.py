"""
Data models for Tesla dashcam telemetry export.

Pydantic models representing the 16 telemetry fields from Tesla SEI metadata.
"""

from datetime import datetime
from enum import IntEnum
from typing import List, Optional

from pydantic import BaseModel, Field


class GearState(IntEnum):
    """Vehicle gear state matching Tesla protobuf enum."""
    PARK = 0
    DRIVE = 1
    REVERSE = 2
    NEUTRAL = 3


class AutopilotState(IntEnum):
    """Autopilot mode matching Tesla protobuf enum."""
    NONE = 0
    SELF_DRIVING = 1  # FSD
    AUTOSTEER = 2
    TACC = 3  # Traffic-Aware Cruise Control


class TelemetryRecord(BaseModel):
    """
    Single telemetry sample from a video frame.

    Contains all 16 fields from Tesla SEI metadata, plus computed timestamp.
    """
    # Timing
    timestamp: datetime = Field(description="Computed from clip start + frame_idx / fps")
    frame_seq_no: int = Field(description="Frame sequence number from SEI")

    # GPS
    latitude_deg: float = Field(description="WGS84 latitude in degrees")
    longitude_deg: float = Field(description="WGS84 longitude in degrees")
    heading_deg: float = Field(description="Vehicle heading in degrees (0=North, 90=East)")

    # Motion dynamics
    vehicle_speed_mps: float = Field(description="Vehicle speed in meters per second")
    linear_acceleration_x: float = Field(description="Lateral acceleration in m/s^2 (left/right)")
    linear_acceleration_y: float = Field(description="Longitudinal acceleration in m/s^2 (forward/backward)")
    linear_acceleration_z: float = Field(description="Vertical acceleration in m/s^2 (up/down)")

    # Controls
    steering_wheel_angle: float = Field(description="Steering wheel angle in degrees (negative=left)")
    accelerator_pedal_position: float = Field(description="Accelerator pedal position 0.0-1.0")
    brake_applied: bool = Field(description="Whether brake pedal is pressed")

    # Signals and state
    blinker_on_left: bool = Field(description="Left turn signal active")
    blinker_on_right: bool = Field(description="Right turn signal active")
    gear_state: GearState = Field(description="Current gear: P/D/R/N")
    autopilot_state: AutopilotState = Field(description="Autopilot mode")

    @property
    def speed_mph(self) -> float:
        """Convert speed to miles per hour."""
        return self.vehicle_speed_mps * 2.23694

    @property
    def speed_kmh(self) -> float:
        """Convert speed to kilometers per hour."""
        return self.vehicle_speed_mps * 3.6

    @property
    def lateral_g(self) -> float:
        """Lateral acceleration in G-force."""
        return self.linear_acceleration_x / 9.81

    @property
    def longitudinal_g(self) -> float:
        """Longitudinal acceleration in G-force."""
        return self.linear_acceleration_y / 9.81

    @classmethod
    def from_sei_metadata(
        cls,
        sei_meta,
        timestamp: datetime,
    ) -> "TelemetryRecord":
        """
        Create a TelemetryRecord from a protobuf SeiMetadata object.

        Args:
            sei_meta: dashcam_pb2.SeiMetadata protobuf object
            timestamp: Computed timestamp for this frame

        Returns:
            TelemetryRecord with all fields populated
        """
        return cls(
            timestamp=timestamp,
            frame_seq_no=sei_meta.frame_seq_no,
            latitude_deg=sei_meta.latitude_deg,
            longitude_deg=sei_meta.longitude_deg,
            heading_deg=sei_meta.heading_deg,
            vehicle_speed_mps=sei_meta.vehicle_speed_mps,
            linear_acceleration_x=sei_meta.linear_acceleration_mps2_x,
            linear_acceleration_y=sei_meta.linear_acceleration_mps2_y,
            linear_acceleration_z=sei_meta.linear_acceleration_mps2_z,
            steering_wheel_angle=sei_meta.steering_wheel_angle,
            accelerator_pedal_position=sei_meta.accelerator_pedal_position,
            brake_applied=sei_meta.brake_applied,
            blinker_on_left=sei_meta.blinker_on_left,
            blinker_on_right=sei_meta.blinker_on_right,
            gear_state=GearState(sei_meta.gear_state),
            autopilot_state=AutopilotState(sei_meta.autopilot_state),
        )


class TelemetryTrack(BaseModel):
    """
    Complete telemetry track from one or more video clips.

    Represents the full driving session with all telemetry records in order.
    """
    name: str = Field(description="Track name (typically from first clip timestamp)")
    source_clips: List[str] = Field(description="List of source clip filenames")
    records: List[TelemetryRecord] = Field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Total duration of the track in seconds."""
        if len(self.records) < 2:
            return 0.0
        return (self.records[-1].timestamp - self.records[0].timestamp).total_seconds()

    @property
    def distance_meters(self) -> float:
        """
        Approximate total distance traveled in meters.

        Computed by integrating speed over time intervals.
        """
        if len(self.records) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(self.records)):
            dt = (self.records[i].timestamp - self.records[i - 1].timestamp).total_seconds()
            avg_speed = (self.records[i].vehicle_speed_mps + self.records[i - 1].vehicle_speed_mps) / 2
            total += avg_speed * dt
        return total

    @property
    def start_time(self) -> Optional[datetime]:
        """Start time of the track."""
        return self.records[0].timestamp if self.records else None

    @property
    def end_time(self) -> Optional[datetime]:
        """End time of the track."""
        return self.records[-1].timestamp if self.records else None
