"""
GPX 1.1 writer with Garmin and Tesla extensions.

Exports Tesla dashcam telemetry to GPX format with:
- Standard GPX 1.1 trackpoints (lat, lon, time)
- Garmin TrackPointExtension for speed and course
- Custom Tesla extension for all other telemetry fields
"""

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import TextIO

from telemetry_export.data_models import TelemetryTrack, GearState, AutopilotState


# XML namespaces
NS_GPX = "http://www.topografix.com/GPX/1/1"
NS_GARMIN = "http://www.garmin.com/xmlschemas/TrackPointExtension/v2"
NS_TESLA = "http://tesla.com/gpx/telemetry/v1"
NS_XSI = "http://www.w3.org/2001/XMLSchema-instance"


def _gear_name(gear: GearState) -> str:
    """Convert gear enum to readable name."""
    return {
        GearState.PARK: "PARK",
        GearState.DRIVE: "DRIVE",
        GearState.REVERSE: "REVERSE",
        GearState.NEUTRAL: "NEUTRAL",
    }.get(gear, "UNKNOWN")


def _autopilot_name(state: AutopilotState) -> str:
    """Convert autopilot enum to readable name."""
    return {
        AutopilotState.NONE: "MANUAL",
        AutopilotState.SELF_DRIVING: "FSD",
        AutopilotState.AUTOSTEER: "AUTOSTEER",
        AutopilotState.TACC: "TACC",
    }.get(state, "UNKNOWN")


def write_gpx(track: TelemetryTrack, output: TextIO) -> None:
    """
    Write a TelemetryTrack to GPX 1.1 format with extensions.

    Args:
        track: TelemetryTrack with telemetry records
        output: File-like object to write to
    """
    # Register namespaces to avoid ns0/ns1 prefixes
    ET.register_namespace("", NS_GPX)
    ET.register_namespace("gpxtpx", NS_GARMIN)
    ET.register_namespace("tesla", NS_TESLA)
    ET.register_namespace("xsi", NS_XSI)

    # Create root GPX element with namespaces
    # Use nsmap approach to avoid duplicate xmlns attributes
    gpx = ET.Element(
        "{%s}gpx" % NS_GPX,
        attrib={
            "version": "1.1",
            "creator": "Tesla Dashcam Telemetry Exporter",
            "{%s}schemaLocation" % NS_XSI: (
                f"{NS_GPX} http://www.topografix.com/GPX/1/1/gpx.xsd "
                f"{NS_GARMIN} http://www.garmin.com/xmlschemas/TrackPointExtensionv2.xsd"
            ),
        }
    )

    # Metadata
    metadata = ET.SubElement(gpx, "metadata")
    name_elem = ET.SubElement(metadata, "name")
    name_elem.text = track.name

    if track.start_time:
        time_elem = ET.SubElement(metadata, "time")
        time_elem.text = _format_time(track.start_time)

    desc = ET.SubElement(metadata, "desc")
    desc.text = f"Tesla dashcam telemetry from {len(track.source_clips)} clip(s)"

    # Track
    trk = ET.SubElement(gpx, "trk")
    trk_name = ET.SubElement(trk, "name")
    trk_name.text = track.name

    trk_src = ET.SubElement(trk, "src")
    trk_src.text = "Tesla Dashcam SEI Metadata"

    # Track segment
    trkseg = ET.SubElement(trk, "trkseg")

    # Add trackpoints
    for record in track.records:
        # Skip null island (invalid GPS)
        if abs(record.latitude_deg) < 0.001 and abs(record.longitude_deg) < 0.001:
            continue

        trkpt = ET.SubElement(trkseg, "trkpt")
        trkpt.set("lat", f"{record.latitude_deg:.7f}")
        trkpt.set("lon", f"{record.longitude_deg:.7f}")

        # Standard GPX time
        time_elem = ET.SubElement(trkpt, "time")
        time_elem.text = _format_time(record.timestamp)

        # Extensions container
        extensions = ET.SubElement(trkpt, "extensions")

        # Garmin TrackPointExtension (speed and course)
        gpxtpx = ET.SubElement(extensions, f"{{{NS_GARMIN}}}TrackPointExtension")

        # Speed in m/s (Garmin standard)
        speed_elem = ET.SubElement(gpxtpx, f"{{{NS_GARMIN}}}speed")
        speed_elem.text = f"{record.vehicle_speed_mps:.2f}"

        # Course/heading in degrees
        course_elem = ET.SubElement(gpxtpx, f"{{{NS_GARMIN}}}course")
        course_elem.text = f"{record.heading_deg:.1f}"

        # Tesla custom extension for all other telemetry
        tesla_ext = ET.SubElement(extensions, f"{{{NS_TESLA}}}TeslaExtension")

        # Vehicle state
        _add_tesla_elem(tesla_ext, "gear", _gear_name(record.gear_state))
        _add_tesla_elem(tesla_ext, "autopilot", _autopilot_name(record.autopilot_state))

        # Controls
        _add_tesla_elem(tesla_ext, "steering_angle", f"{record.steering_wheel_angle:.1f}")
        _add_tesla_elem(tesla_ext, "accelerator", f"{record.accelerator_pedal_position:.3f}")
        _add_tesla_elem(tesla_ext, "brake", str(record.brake_applied).lower())

        # Signals
        _add_tesla_elem(tesla_ext, "blinker_left", str(record.blinker_on_left).lower())
        _add_tesla_elem(tesla_ext, "blinker_right", str(record.blinker_on_right).lower())

        # 3-axis acceleration
        _add_tesla_elem(tesla_ext, "accel_x", f"{record.linear_acceleration_x:.4f}")
        _add_tesla_elem(tesla_ext, "accel_y", f"{record.linear_acceleration_y:.4f}")
        _add_tesla_elem(tesla_ext, "accel_z", f"{record.linear_acceleration_z:.4f}")

        # Frame sequence for correlation with video
        _add_tesla_elem(tesla_ext, "frame_seq", str(record.frame_seq_no))

    # Write XML with declaration
    tree = ET.ElementTree(gpx)
    output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    tree.write(output, encoding="unicode", xml_declaration=False)


def _add_tesla_elem(parent: ET.Element, name: str, value: str) -> None:
    """Add a Tesla namespace element to the parent."""
    elem = ET.SubElement(parent, f"{{{NS_TESLA}}}{name}")
    elem.text = value


def _format_time(dt: datetime) -> str:
    """Format datetime as ISO 8601 UTC string for GPX."""
    # Ensure UTC timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
