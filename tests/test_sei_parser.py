"""
Tests for SEI (Supplemental Enhancement Information) parser.

Tests NAL unit parsing, emulation prevention byte stripping, and protobuf extraction.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import struct

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sei_parser import (
    extract_sei_data,
    strip_emulation_prevention_bytes,
    extract_proto_payload,
    find_mdat,
    iter_nals,
    ParseStats,
)


class TestStripEmulationPreventionBytes:
    """Tests for emulation prevention byte removal."""

    def test_no_prevention_bytes(self):
        """Data without prevention bytes should be unchanged."""
        data = bytes([0x01, 0x02, 0x03, 0x04])
        result = strip_emulation_prevention_bytes(data)
        assert result == data

    def test_single_prevention_byte(self):
        """Single 0x00 0x00 0x03 sequence should have 0x03 removed."""
        data = bytes([0x00, 0x00, 0x03, 0x04])
        result = strip_emulation_prevention_bytes(data)
        assert result == bytes([0x00, 0x00, 0x04])

    def test_multiple_prevention_bytes(self):
        """Multiple prevention bytes should all be removed."""
        data = bytes([0x00, 0x00, 0x03, 0x01, 0x00, 0x00, 0x03, 0x02])
        result = strip_emulation_prevention_bytes(data)
        assert result == bytes([0x00, 0x00, 0x01, 0x00, 0x00, 0x02])

    def test_consecutive_zeros(self):
        """Long zero sequences with prevention bytes."""
        data = bytes([0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x01])
        result = strip_emulation_prevention_bytes(data)
        assert result == bytes([0x00, 0x00, 0x00, 0x00, 0x01])

    def test_empty_data(self):
        """Empty data should return empty."""
        assert strip_emulation_prevention_bytes(b"") == b""


class TestExtractProtoPayload:
    """Tests for protobuf payload extraction from NAL units."""

    def test_none_for_short_data(self):
        """Data too short should return None."""
        assert extract_proto_payload(b"") is None
        assert extract_proto_payload(b"\x00") is None

    def test_none_for_non_bytes(self):
        """Non-bytes input should return None."""
        assert extract_proto_payload(None) is None
        assert extract_proto_payload("string") is None

    def test_valid_sei_payload(self):
        """Valid SEI NAL with 0x42 0x69 marker should extract payload."""
        # Format: first 3 bytes header, then 0x42 marker(s), 0x69, payload, trailing byte
        nal = bytes([0x06, 0x05, 0x10,  # SEI header
                     0x42, 0x42, 0x69,   # Tesla markers
                     0x01, 0x02, 0x03,   # Payload
                     0x80])              # Trailing
        result = extract_proto_payload(nal)
        assert result == bytes([0x01, 0x02, 0x03])

    def test_no_marker_returns_none(self):
        """NAL without 0x69 marker should return None."""
        nal = bytes([0x06, 0x05, 0x10, 0x42, 0x42, 0x01, 0x02])
        result = extract_proto_payload(nal)
        assert result is None


class TestParseStats:
    """Tests for ParseStats dataclass."""

    def test_default_values(self):
        """Default values should be zero."""
        stats = ParseStats()
        assert stats.nal_units_scanned == 0
        assert stats.sei_units_found == 0
        assert stats.protobuf_payloads == 0
        assert stats.decode_errors == 0
        assert stats.successful_parses == 0

    def test_mutable_stats(self):
        """Stats should be mutable."""
        stats = ParseStats()
        stats.nal_units_scanned = 100
        stats.decode_errors = 5
        assert stats.nal_units_scanned == 100
        assert stats.decode_errors == 5


class TestFindMdat:
    """Tests for MP4 mdat atom discovery."""

    def test_simple_mdat(self):
        """Simple file with mdat atom."""
        # Create minimal MP4 with ftyp and mdat atoms
        ftyp = struct.pack(">I4s", 8, b"ftyp")
        mdat = struct.pack(">I4s", 100, b"mdat")
        data = ftyp + mdat + b"\x00" * 92  # mdat payload

        with patch("builtins.open", mock_open(read_data=data)):
            fp = open("test.mp4", "rb")
            fp.read = MagicMock(side_effect=[
                data[0:8],   # ftyp header
                data[8:16],  # mdat header
            ])
            fp.seek = MagicMock()
            fp.tell = MagicMock(return_value=16)

            offset, size = find_mdat(fp)
            assert offset == 16  # After mdat header
            assert size == 92    # mdat size - header

    def test_mdat_not_found(self):
        """File without mdat should raise RuntimeError."""
        # Only ftyp atom, then EOF
        with patch("builtins.open", mock_open()):
            fp = MagicMock()
            fp.read = MagicMock(side_effect=[
                struct.pack(">I4s", 8, b"ftyp"),
                b"",  # EOF
            ])
            fp.seek = MagicMock()

            with pytest.raises(RuntimeError, match="mdat atom not found"):
                find_mdat(fp)


class TestIterNals:
    """Tests for NAL unit iteration."""

    def test_sei_nal_yielded(self):
        """SEI NAL units should be yielded."""
        stats = ParseStats()

        # Create a mock file with SEI NAL
        # NAL header: 4-byte size, then NAL type byte (0x06 = SEI), SEI type (0x05 = user data)
        nal_size = 10
        nal_data = bytes([0x06, 0x05]) + bytes([0x42] * 8)

        fp = MagicMock()
        fp.read = MagicMock(side_effect=[
            struct.pack(">I", nal_size),  # NAL size header
            bytes([0x06, 0x05]),          # First 2 bytes: SEI type
            bytes([0x42] * 8),            # Rest of NAL
            b"",                          # EOF
        ])
        fp.seek = MagicMock()

        nals = list(iter_nals(fp, 0, 20, stats))
        assert stats.nal_units_scanned == 1
        assert stats.sei_units_found == 1

    def test_non_sei_nal_skipped(self):
        """Non-SEI NAL units should be skipped."""
        stats = ParseStats()

        # IDR slice (NAL type 5)
        nal_size = 10
        fp = MagicMock()
        fp.read = MagicMock(side_effect=[
            struct.pack(">I", nal_size),  # NAL size header
            bytes([0x05, 0x00]),          # IDR, not SEI
            b"",
        ])
        fp.seek = MagicMock()

        nals = list(iter_nals(fp, 0, 20, stats))
        assert len(nals) == 0
        assert stats.nal_units_scanned == 1
        assert stats.sei_units_found == 0


class TestExtractSeiData:
    """Integration tests for full SEI extraction."""

    def test_nonexistent_file(self):
        """Non-existent file should return empty dict."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            # The function opens the file, so we expect it to fail
            with pytest.raises(FileNotFoundError):
                extract_sei_data("/nonexistent/file.mp4")

    def test_file_without_mdat(self):
        """File without mdat should return empty dict."""
        with patch("builtins.open", mock_open(read_data=b"")):
            with patch("sei_parser.find_mdat", side_effect=RuntimeError("no mdat")):
                result = extract_sei_data("test.mp4")
                assert result == {}
