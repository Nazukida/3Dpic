"""Multi-format 3D Gaussian Splatting I/O with optional gaussforge backend.

When the ``gaussforge`` package is installed (``pip install recon3d[formats]``),
this module supports reading and writing PLY, SPLAT, KSPLAT, SPZ, and SOG
formats.  Without it, only PLY is available via the built-in reader.
"""

from __future__ import annotations

import os

import numpy as np

from .ply_io import read_ply, PlyError

# --- Optional gaussforge import ---
try:
    import gaussforge
    HAS_GAUSSFORGE = True
except ImportError:
    gaussforge = None  # type: ignore[assignment]
    HAS_GAUSSFORGE = False

# Extensions that always work (built-in PLY reader)
_BASE_EXTENSIONS = [".ply"]

# Extensions requiring gaussforge
_GAUSSFORGE_EXTENSIONS = [".splat", ".ksplat", ".spz", ".sog"]


def supported_formats() -> list[str]:
    """Return list of openable file extensions (including the leading dot)."""
    exts = list(_BASE_EXTENSIONS)
    if HAS_GAUSSFORGE:
        exts.extend(_GAUSSFORGE_EXTENSIONS)
    return exts


def _require_gaussforge(path: str) -> None:
    """Raise a helpful error if gaussforge is needed but not installed."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in _BASE_EXTENSIONS and not HAS_GAUSSFORGE:
        raise ImportError(
            f"打开 {ext} 格式需要 gaussforge 库。\n"
            f"请运行: pip install recon3d[formats]"
        )


def read_point_cloud(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a point cloud file and return (positions Nx3 float32, colors Nx3 uint8).

    For .ply files the built-in reader is used.  For other formats, gaussforge
    is required.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".ply":
        data = read_ply(path)
        positions = data.xyz
        if data.rgb is not None:
            colors = data.rgb
        else:
            # No colour info — generate grey
            colors = np.full((len(positions), 3), 180, dtype=np.uint8)
        return positions, colors

    # Non-PLY: require gaussforge
    _require_gaussforge(path)

    cloud = gaussforge.read(path)
    positions = np.asarray(cloud.positions, dtype=np.float32)

    if hasattr(cloud, "colors") and cloud.colors is not None:
        colors_raw = np.asarray(cloud.colors)
        # Normalise to uint8 Nx3
        if colors_raw.dtype.kind == "f":
            colors = np.clip(colors_raw * 255.0, 0, 255).astype(np.uint8)
        else:
            colors = colors_raw.astype(np.uint8)
        # Handle RGBA -> RGB
        if colors.ndim == 2 and colors.shape[1] == 4:
            colors = colors[:, :3]
    else:
        colors = np.full((len(positions), 3), 180, dtype=np.uint8)

    return positions, colors


def write_point_cloud(path: str, positions: np.ndarray, colors: np.ndarray) -> None:
    """Write a point cloud to *path*, format determined by extension.

    For .ply the built-in writer is used.  For other formats, gaussforge is
    required.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".ply":
        # Use the project's existing PLY writer
        from recon3d.io_utils import write_ply
        write_ply(path, positions, colors)
        return

    _require_gaussforge(path)

    # Build a GaussianCloud and write via gaussforge
    cloud = gaussforge.GaussianCloud()
    cloud.positions = np.ascontiguousarray(positions, dtype=np.float32)
    # Ensure colours are float32 0-1 for gaussforge
    colors_f = np.ascontiguousarray(colors, dtype=np.float32)
    if colors_f.max() > 1.0:
        colors_f = colors_f / 255.0
    cloud.colors = colors_f
    gaussforge.write(cloud, path)


def convert_file(input_path: str, output_path: str) -> None:
    """Convert between supported formats using gaussforge.convert().

    Both input and output can be any gaussforge-supported format.
    """
    _require_gaussforge(input_path)
    _require_gaussforge(output_path)
    gaussforge.convert(input_path, output_path)
