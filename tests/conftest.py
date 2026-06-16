import json
import struct
import zlib
from pathlib import Path

import pytest


def _tiny_png(color: tuple[int, int, int] = (128, 128, 128)) -> bytes:
    """A valid 1x1 PNG, hand-built so tests need no image library."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1, 8-bit RGB
    raw = bytes([0]) + bytes(color)  # one filter byte + one RGB pixel
    idat = zlib.compress(raw)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


@pytest.fixture
def collection_dir(tmp_path: Path) -> Path:
    """A collection of three PNGs with an objects.json sidecar."""
    for i, oid in enumerate(("100", "200", "300")):
        (tmp_path / f"{oid}.png").write_bytes(_tiny_png((i * 40, i * 40, i * 40)))
    sidecar = {
        "collection_id": "test_collection",
        "organelle_type": "mitochondrion",
        "objects": [
            {"object_id": "100", "image": "100.png", "meta": {"voxel_count": 84213}},
            {"object_id": "200", "image": "200.png", "meta": {"at_volume_edge": True}},
            {"object_id": "300", "image": "300.png", "meta": {}},
        ],
    }
    (tmp_path / "objects.json").write_text(json.dumps(sidecar))
    return tmp_path


@pytest.fixture
def scan_dir(tmp_path: Path) -> Path:
    """A collection with no sidecar — must be discovered by scanning."""
    for oid in ("a", "b"):
        (tmp_path / f"{oid}.png").write_bytes(_tiny_png())
    return tmp_path
