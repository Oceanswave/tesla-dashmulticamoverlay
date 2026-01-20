"""
SEI (Supplemental Enhancement Information) parser for Tesla dashcam videos.

Extracts embedded protobuf telemetry data from H.264 NAL units within MP4 files.
"""

import struct
import logging
from typing import BinaryIO, Generator, Optional, Tuple, Dict
from dataclasses import dataclass
from google.protobuf.message import DecodeError
import dashcam_pb2

logger = logging.getLogger(__name__)


@dataclass
class ParseStats:
    """Statistics from SEI parsing operation."""
    nal_units_scanned: int = 0
    sei_units_found: int = 0
    protobuf_payloads: int = 0
    decode_errors: int = 0
    successful_parses: int = 0


def extract_sei_data(path: str) -> Dict[int, dashcam_pb2.SeiMetadata]:
    """
    Extracts SEI metadata from the video file and returns a dictionary
    mapping frame index (0-based, roughly) to SeiMetadata.

    Note: purely sequential extraction assumes the SEI messages appear
    in order corresponding to frames.

    Args:
        path: Path to the MP4 video file

    Returns:
        Dictionary mapping frame index to SeiMetadata protobuf objects
    """
    sei_data = {}
    stats = ParseStats()

    with open(path, "rb") as fp:
        try:
            offset, size = find_mdat(fp)
            logger.debug(f"Found mdat atom at offset {offset}, size {size}")
        except RuntimeError as e:
            logger.warning(f"Failed to find mdat atom in {path}: {e}")
            return {}

        # We'll just index them sequentially as we find them.
        # This is an approximation if there are gaps, but standard for this format.
        index = 0
        for meta in iter_sei_messages(fp, offset, size, stats):
            sei_data[index] = meta
            index += 1

    # Log parsing summary
    if stats.decode_errors > 0:
        logger.warning(
            f"SEI parsing completed with {stats.decode_errors} decode errors "
            f"({stats.successful_parses} successful)"
        )
    logger.debug(
        f"SEI stats for {path}: "
        f"NALs={stats.nal_units_scanned}, "
        f"SEI={stats.sei_units_found}, "
        f"payloads={stats.protobuf_payloads}, "
        f"parsed={stats.successful_parses}, "
        f"errors={stats.decode_errors}"
    )

    return sei_data


def iter_sei_messages(
    fp: BinaryIO, offset: int, size: int, stats: ParseStats
) -> Generator[dashcam_pb2.SeiMetadata, None, None]:
    """Yield parsed SeiMetadata messages from the MP4 file."""
    for nal in iter_nals(fp, offset, size, stats):
        payload = extract_proto_payload(nal)
        if not payload:
            continue

        stats.protobuf_payloads += 1
        meta = dashcam_pb2.SeiMetadata()
        try:
            meta.ParseFromString(payload)
            stats.successful_parses += 1
        except DecodeError as e:
            stats.decode_errors += 1
            logger.debug(f"Protobuf decode error: {e}")
            continue
        yield meta


def extract_proto_payload(nal: bytes) -> Optional[bytes]:
    """Extract protobuf payload from SEI NAL unit."""
    if not isinstance(nal, bytes) or len(nal) < 2:
        return None
    for i in range(3, len(nal) - 1):
        byte = nal[i]
        if byte == 0x42:
            continue
        if byte == 0x69:
            if i > 2:
                return strip_emulation_prevention_bytes(nal[i + 1:-1])
            break
        break
    return None


def strip_emulation_prevention_bytes(data: bytes) -> bytes:
    """Remove emulation prevention bytes (0x03 following 0x00 0x00)."""
    stripped = bytearray()
    zero_count = 0
    for byte in data:
        if zero_count >= 2 and byte == 0x03:
            zero_count = 0
            continue
        stripped.append(byte)
        zero_count = 0 if byte != 0 else zero_count + 1
    return bytes(stripped)


def iter_nals(
    fp: BinaryIO, offset: int, size: int, stats: ParseStats
) -> Generator[bytes, None, None]:
    """Yield SEI user NAL units from the MP4 mdat atom."""
    NAL_ID_SEI = 6
    NAL_SEI_ID_USER_DATA_UNREGISTERED = 5

    fp.seek(offset)
    consumed = 0
    while size == 0 or consumed < size:
        header = fp.read(4)
        if len(header) < 4:
            break
        nal_size = struct.unpack(">I", header)[0]
        stats.nal_units_scanned += 1

        if nal_size < 2:
            fp.seek(nal_size, 1)
            consumed += 4 + nal_size
            continue

        first_two = fp.read(2)
        if len(first_two) != 2:
            break

        if (first_two[0] & 0x1F) != NAL_ID_SEI or first_two[1] != NAL_SEI_ID_USER_DATA_UNREGISTERED:
            fp.seek(nal_size - 2, 1)
            consumed += 4 + nal_size
            continue  # skip non-SEI NALs

        stats.sei_units_found += 1
        rest = fp.read(nal_size - 2)
        if len(rest) != nal_size - 2:
            break
        payload = first_two + rest
        consumed += 4 + nal_size
        yield payload


def find_mdat(fp: BinaryIO) -> Tuple[int, int]:
    """Return (offset, size) for the first mdat atom."""
    fp.seek(0)
    while True:
        header = fp.read(8)
        if len(header) < 8:
            raise RuntimeError("mdat atom not found")
        size32, atom_type = struct.unpack(">I4s", header)
        if size32 == 1:
            large = fp.read(8)
            if len(large) != 8:
                raise RuntimeError("truncated extended atom size")
            atom_size = struct.unpack(">Q", large)[0]
            header_size = 16
        else:
            atom_size = size32
            header_size = 8

        # Handle special size values per MP4 spec
        if atom_size == 0:
            # Size 0 means "atom extends to end of file" (only valid for last atom)
            if atom_type == b"mdat":
                # Calculate remaining file size as payload
                current_pos = fp.tell()
                fp.seek(0, 2)  # Seek to end
                file_end = fp.tell()
                payload_size = file_end - current_pos
                fp.seek(current_pos)  # Restore position
                return current_pos, payload_size
            else:
                # Non-mdat atom extends to EOF - we're done searching
                break

        if atom_type == b"mdat":
            payload_size = atom_size - header_size
            return fp.tell(), payload_size

        if atom_size < header_size:
            raise RuntimeError("invalid MP4 atom size")
        fp.seek(atom_size - header_size, 1)
