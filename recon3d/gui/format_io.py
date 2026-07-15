"""Multi-format point-cloud / model I/O.

Reading is handled by :mod:`recon3d.gui.model_io` (:func:`read_model`), which
supports ~14 formats with pure numpy and routes heavy-lib formats (torch, h5py,
lazrs, gaussforge) through optional extras.  This module keeps the older
``read_point_cloud`` / ``write_point_cloud`` / ``convert_file`` API that the
export path and reconstruction panel use.
"""

from __future__ import annotations

import os

import numpy as np

from .model_io import supported_extensions, read_model, ModelReadError  # noqa: F401
from .ply_io import PlyError  # noqa: F401

# --- Optional gaussforge import (used only for write/convert + export menu) ---
try:
    import gaussforge
    HAS_GAUSSFORGE = True
except ImportError:
    gaussforge = None  # type: ignore[assignment]
    HAS_GAUSSFORGE = False


def supported_formats() -> list[str]:
    """Return list of openable file extensions (including the leading dot)."""
    return supported_extensions()


def read_point_cloud(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read any supported model file and return (positions Nx3, colors Nx3 uint8).

    Thin compatibility wrapper over :func:`recon3d.gui.model_io.read_model` —
    collapses opacity/scale/normals (the full viewer path uses ``ModelLoader``
    directly to keep those).
    """
    data = read_model(path)
    positions = data.xyz
    if data.rgb is not None:
        colors = data.rgb
    else:
        colors = np.full((len(positions), 3), 180, dtype=np.uint8)
    return positions, colors


def write_point_cloud(path: str, positions: np.ndarray, colors: np.ndarray) -> None:
    """Write a point cloud to *path*, format determined by extension.

    For .ply the built-in writer is used.  For other formats, gaussforge
    is required.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".ply":
        from recon3d.io_utils import write_ply
        write_ply(path, positions, colors)
        return

    if not HAS_GAUSSFORGE:
        raise ImportError(
            f"导出 {ext} 格式需要 gaussforge 库。\n请运行: pip install recon3d[formats]"
        )

    cloud = gaussforge.GaussianCloud()
    cloud.positions = np.ascontiguousarray(positions, dtype=np.float32)
    colors_f = np.ascontiguousarray(colors, dtype=np.float32)
    if colors_f.max() > 1.0:
        colors_f = colors_f / 255.0
    cloud.colors = colors_f
    gaussforge.write(cloud, path)


def convert_file(input_path: str, output_path: str) -> None:
    """Convert between supported formats using gaussforge.convert()."""
    if not HAS_GAUSSFORGE:
        raise ImportError("格式转换需要 gaussforge 库。\n请运行: pip install recon3d[formats]")
    gaussforge.convert(input_path, output_path)
