"""Universal, adaptive 3D-model reader for the viewer.

One entry point — :func:`read_model` — loads almost any file that expresses a
3D model and returns a :class:`~recon3d.gui.ply_io.PlyData` (xyz / rgb /
normals / opacity / scale / is_gaussian).  The viewer only renders points, so
meshes are surface-sampled into a dense point cloud and bag-of-arrays formats
(``.pt/.npy/.npz/.h5``) are scanned heuristically for the xyz/rgb/opacity/scale
arrays.

Format detection is **content-first** (magic bytes), extension-second, so a
misnamed file still loads.  ~14 formats parse with pure numpy (no extra deps);
heavy libs (torch, h5py, lazrs, gaussforge, trimesh) are optional — when a
format needs one that isn't installed, a clear ``pip install`` hint is raised.
"""

from __future__ import annotations

import json
import os
import struct
import zipfile
from dataclasses import replace
from typing import Any

import numpy as np

from .ply_io import (
    PlyData, PlyError, read_ply, _sh_dc_to_rgb, _sigmoid,
)


class ModelReadError(Exception):
    """Raised when a file cannot be interpreted as a 3D model."""


def _hint(pkg: str, fmt: str) -> str:
    return (f"打开 {fmt} 格式需要 {pkg} 库（未安装）。\n"
            f"请运行: pip install {pkg}\n"
            f"或: pip install recon3d[models]")


# --------------------------------------------------------------------------- #
# Small normalisation helpers
# --------------------------------------------------------------------------- #
def _rgb_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalise an (N,3) or (N,4) colour array to uint8 (N,3)."""
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ModelReadError("colour array is not (N,>=3)")
    arr = arr[:, :3]
    if arr.dtype.kind == "f":
        mx = float(arr.max()) if arr.size else 0.0
        if mx <= 1.0 + 1e-6:
            arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _norm_opacity(arr: np.ndarray) -> np.ndarray:
    """Opacity -> float32 in [0,1].

    Integer 0..255 -> /255; float 0..1 -> as-is; anything else (e.g. raw
    3DGS logits) -> sigmoid.
    """
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if arr.dtype.kind in "iu" and mn >= 0 and mx <= 255:
        arr = arr / 255.0
    elif mn < -0.001 or mx > 1.001:          # outside [0,1] -> treat as logit
        arr = _sigmoid(arr)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _norm_scale(arr: np.ndarray, is_log: bool = False) -> np.ndarray:
    """Scale -> (N,) float32. If stored as log-scale, exp it. Average 3 axes.

    If ``exp`` produces inf/NaN or absurd values, the array was probably already
    linear — fall back to the raw values.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] >= 3:
        arr = arr[:, :3].mean(axis=1)
    arr = arr.ravel()
    if is_log:
        exp = np.exp(arr)
        if (np.isfinite(exp).all() and float(exp.max()) < 1e6
                and float(exp.max()) > 1e-12):
            arr = exp
    return np.maximum(arr, 1e-6).astype(np.float32)


def _finalize(xyz, rgb=None, normals=None, opacity=None, scale=None,
              is_gaussian=False, source="") -> PlyData:
    """Assemble a PlyData with contiguous, correctly-typed arrays."""
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    n = len(xyz)
    if rgb is not None and len(rgb) == n:
        rgb = np.ascontiguousarray(_rgb_to_uint8(rgb) if rgb.dtype.kind == "f"
                                   else np.clip(rgb[:, :3], 0, 255).astype(np.uint8))
    else:
        rgb = None
    if normals is not None and len(normals) == n:
        normals = np.ascontiguousarray(normals, dtype=np.float32)
    else:
        normals = None
    if opacity is not None and len(opacity) == n:
        opacity = np.ascontiguousarray(opacity, dtype=np.float32)
    else:
        opacity = None
    if scale is not None and len(scale) == n:
        scale = np.ascontiguousarray(scale, dtype=np.float32)
    else:
        scale = None
    return PlyData(xyz=xyz, rgb=rgb, normals=normals, opacity=opacity,
                   scale=scale, count=n, is_gaussian=is_gaussian, source=source)


def _apply_max_points(data: PlyData, max_points: int | None) -> PlyData:
    n = len(data.xyz)
    if max_points is None or not (0 < max_points < n):
        return data
    step = int(np.ceil(n / max_points))
    idx = np.arange(0, n, step)
    return replace(
        data,
        xyz=np.ascontiguousarray(data.xyz[idx]),
        rgb=data.rgb[idx] if data.rgb is not None else None,
        normals=data.normals[idx] if data.normals is not None else None,
        opacity=data.opacity[idx] if data.opacity is not None else None,
        scale=data.scale[idx] if data.scale is not None else None,
        count=len(idx),
    )


# --------------------------------------------------------------------------- #
# Format detection (content-first, extension-second)
# --------------------------------------------------------------------------- #
def _detect(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "rb") as f:
            head = f.read(512)
    except OSError:
        head = b""
    h = head.lstrip()

    if h[:3] in (b"ply", b"PLY"):
        return "ply"
    if h[:6] == b"\x93NUMPY":
        return "npy"
    if h[:8] == b"\x89HDF\r\n\x1a\n":
        return "h5"
    if h[:4] == b"VOX ":
        return "vox"
    if h[:4] == b"glTF":
        return "glb"
    if h[:1] == b"{":
        return "gltf"
    if h[:4] == b"LASF":
        return "las"
    if h[:4] == b"PK\x03\x04":
        # zip container: torch checkpoint, npz, or gltf-in-zip (rare)
        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
            if any(n.endswith(".npy") for n in names):
                return "npz"
            if any("data.pkl" in n or n.endswith(".pth") for n in names):
                return "pt"
            if any(n.endswith(".gltf") or n.endswith(".glb") for n in names):
                return "glb"
        except zipfile.BadZipFile:
            pass
        return "pt" if ext in (".pt", ".pth") else "npz"

    # ASCII-ish formats — sniff first non-comment token.
    txt = head.decode("latin-1", "replace")
    stripped = "\n".join(l for l in txt.splitlines()
                         if l.strip() and not l.lstrip().startswith("#"))
    first = stripped.split(None, 1)[0] if stripped.split() else ""

    if first in ("OFF", "COFF", "NOFF", "4OFF", "STOFF"):
        return "off"
    if first == "solid" and b"facet" in head:
        return "stl_ascii"
    if first.startswith("VERSION") and "FIELDS" in txt:
        return "pcd"
    if first == "v" or first == "vn" or first == "f" or first == "vt":
        return "obj"
    if first == "#obj" or first == "obj":
        return "obj"

    # binary STL: 80-byte header + uint32 count + 50*ntri. A binary STL may
    # legally start with "solid" in its header, so verify by the count formula
    # (count@80 * 50 + 84 == filesize) — that's authoritative — before falling
    # back to the ascii "solid"/"facet" keywords.
    if ext == ".stl":
        sz = os.path.getsize(path)
        if sz >= 84 and len(head) >= 84:
            ntri = struct.unpack_from("<I", head, 80)[0]
            if ntri < 100_000_000 and ntri * 50 + 84 == sz:
                return "stl_bin"
        if first == "solid" or b"facet" in head:
            return "stl_ascii"
        return "stl_bin"

    # Extension fallback for magic-less / hard-to-sniff formats.
    return {
        ".ply": "ply", ".pcd": "pcd",
        ".xyz": "text", ".xyzrgb": "text", ".xyzn": "text",
        ".txt": "text", ".pts": "text", ".csv": "text", ".asc": "text",
        ".obj": "obj", ".stl": "stl_bin", ".off": "off",
        ".las": "las", ".laz": "laz",
        ".splat": "splat", ".ksplat": "ksplat", ".spz": "spz", ".sog": "sog",
        ".npy": "npy", ".npz": "npz",
        ".pt": "pt", ".pth": "pt", ".h5": "h5", ".hdf5": "h5",
        ".vox": "vox", ".glb": "glb", ".gltf": "gltf",
        ".bin": "colmap",           # COLMAP points3D.bin (sniffed inside)
    }.get(ext, "text")               # last resort: try as a text table


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def read_model(path: str, max_points: int | None = None) -> PlyData:
    """Load any supported 3D-model file into a :class:`PlyData`.

    Content-sniffs the format (magic bytes) and falls back to the extension.
    ``max_points`` caps the point count via uniform striding (applied once,
    centrally, after the reader returns the full cloud).
    """
    fmt = _detect(path)
    reader = _READERS.get(fmt, _read_text)
    try:
        data = reader(path)
    except (ModelReadError, PlyError):
        raise
    except Exception as e:                       # noqa: BLE001 - surface a clear msg
        raise ModelReadError(f"无法解析 {os.path.basename(path)} ({fmt}): {e}") from e
    if max_points:
        data = _apply_max_points(data, max_points)
    if not data.source:
        data = replace(data, source=_FORMAT_LABELS.get(fmt, fmt))
    return data


_FORMAT_LABELS = {
    "ply": "PLY", "text": "点云文本", "pcd": "PCD", "obj": "OBJ",
    "stl_bin": "STL", "stl_ascii": "STL", "off": "OFF", "las": "LAS",
    "splat": "SPLAT", "colmap_bin": "COLMAP", "colmap_txt": "COLMAP",
    "npy": "NumPy", "npz": "NumPy", "vox": "MagicaVoxel",
    "glb": "glTF", "gltf": "glTF", "pt": "PyTorch", "h5": "HDF5",
    "ksplat": "KSPLAT", "spz": "SPZ", "sog": "SOG", "laz": "LAZ",
}


# --------------------------------------------------------------------------- #
# PLY (delegate to the fast built-in reader)
# --------------------------------------------------------------------------- #
def _read_ply(path, max_points=None):
    data = read_ply(path)
    return replace(data, source="PLY" if not data.is_gaussian else "3DGS-PLY")


# --------------------------------------------------------------------------- #
# Text tables: .xyz/.xyzrgb/.xyzn/.pts/.csv/.asc/.txt
# --------------------------------------------------------------------------- #
def _sniff_delimiter(sample: str) -> str:
    first = sample.splitlines()[0]
    counts = {d: first.count(d) for d in (",", "\t", ";")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else None     # None -> whitespace


def _looks_like_colmap_txt(first_data: str) -> bool:
    toks = first_data.split()
    if len(toks) < 7:
        return False
    try:
        i0 = float(toks[0])
        x, y, z = float(toks[1]), float(toks[2]), float(toks[3])
        r, g, b = int(toks[4]), int(toks[5]), int(toks[6])
    except ValueError:
        return False
    return (i0 == int(i0) and 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255
            and (abs(x) > 2 or abs(y) > 2 or abs(z) > 2))


def _read_text(path, max_points=None):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        rows = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not rows:
        raise ModelReadError("文件中没有数值数据")
    first_data = rows[0]

    # COLMAP points3D.txt looks like:  ID X Y Z R G B ERROR track...
    # (integer ID, then 3 floats, then 3 bytes 0..255). Detect & dispatch so the
    # ID isn't misread as X.
    if _looks_like_colmap_txt(first_data):
        return _read_colmap_txt(path)

    delim = _sniff_delimiter(first_data)

    def split_row(r):
        return r.split(delim) if delim else r.split()

    # If the first row's first token isn't numeric, treat it as a column header.
    header_names = None
    try:
        float(split_row(first_data)[0])
    except (ValueError, IndexError):
        header_names = [t.strip().lower() for t in split_row(first_data)]
        rows = rows[1:]

    # PTS-style count header: a lone integer on the first line, real data after.
    ncols = len(header_names) if header_names else len(split_row(rows[0]))
    if ncols < 3 and len(rows) > 1:
        ncols = len(split_row(rows[1]))
        if ncols >= 3:
            rows = rows[1:]

    # Join all data rows into one whitespace-separated buffer and parse once
    # (far faster than line-by-line, and tolerant of variable trailing cols).
    sep = " "
    buf = sep.join(r.replace(delim, sep) if delim else r for r in rows)
    flat = np.array(buf.split(), dtype=np.float64)
    if flat.size < ncols or flat.size % ncols != 0:
        # fall back: infer ncols from the first data row
        ncols = len(split_row(rows[0]))
        if flat.size % ncols != 0:
            raise ModelReadError("文本点云列数不一致，无法解析")
    arr = flat.reshape(-1, ncols)
    n, ncols = arr.shape
    if ncols < 3:
        raise ModelReadError(f"文本点云至少需要 3 列 (x,y,z)，实际 {ncols} 列")

    def col(candidates):
        if header_names:
            for c in candidates:
                if c in header_names:
                    return header_names.index(c)
        return None

    xi = col(["x", "px", "pos_x", "position_x"]) or 0
    yi = col(["y", "py", "pos_y", "position_y"]) or 1
    zi = col(["z", "pz", "pos_z", "position_z"]) or 2
    xyz = arr[:, [xi, yi, zi]].astype(np.float32)

    rgb = None
    ri = col(["r", "red", "diffuse_red"])
    gi = col(["g", "green", "diffuse_green"])
    bi = col(["b", "blue", "diffuse_blue"])
    if not (ri is None or gi is None or bi is None):
        rgb = arr[:, [ri, gi, bi]]
    elif ncols >= 6:
        cand = arr[:, 3:6]
        mn, mx = float(cand.min()), float(cand.max())
        # colour-like if all in [0, 255] (covers 0..1 float, 0..255 float/int).
        if mn >= -1e-3 and mx <= 255.0 + 1e-3:
            rgb = cand
    elif ncols == 4:
        rgb = np.repeat(arr[:, 3:4], 3, axis=1)     # intensity -> grey

    normals = None
    nxi = col(["nx", "normal_x", "normalx"])
    nyi = col(["ny", "normal_y", "normaly"])
    nzi = col(["nz", "normal_z", "normalz"])
    if not (nxi is None or nyi is None or nzi is None):
        normals = arr[:, [nxi, nyi, nzi]].astype(np.float32)
    elif ncols >= 9 and rgb is not None:
        normals = arr[:, 6:9].astype(np.float32)

    return _finalize(xyz, rgb, normals, source="点云文本")


# --------------------------------------------------------------------------- #
# PCD (PCL point cloud data)
# --------------------------------------------------------------------------- #
_PCD_FIELD_NP = {"x": ("x", "<f4"), "y": ("y", "<f4"), "z": ("z", "<f4"),
                 "intensity": ("intensity", "<f4"),
                 "normal_x": ("nx", "<f4"), "normal_y": ("ny", "<f4"),
                 "normal_z": ("nz", "<f4"),
                 "rgb": ("rgb", "<u4"), "rgba": ("rgba", "<u4")}

_PCD_TYPE = {1: "u1", 2: "u2", 4: "u4", 8: "u8"}
# PCD 'F' type (4-byte) maps to float32 for x/y/z but uint32 for rgb — decide later.


def _read_pcd(path, max_points=None):
    header = {}
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii", "replace")
            if not line:
                raise ModelReadError("PCD: 未找到 DATA 行")
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0]
            if key == "DATA":
                header["DATA"] = parts[1]
                data_offset = f.tell()
                break
            header[key] = " ".join(parts[1:])
    fields = header["FIELDS"].split()
    sizes = [int(s) for s in header["SIZE"].split()]
    types = header["TYPE"].split()
    counts = [int(c) for c in header["COUNT"].split()] if "COUNT" in header else [1] * len(fields)
    n = int(header.get("POINTS", 0)) or (int(header.get("WIDTH", 1))
                                         * int(header.get("HEIGHT", 1)))
    mode = header["DATA"]
    if mode == "binary_compressed":
        raise ModelReadError("PCD binary_compressed (LZF) 暂不支持；请用 ascii 或 binary 格式，"
                             "或 pip install python-lzf")
    name_idx = {fields[i].lower(): i for i in range(len(fields))}

    if mode == "ascii":
        # data_offset points just past the "DATA ascii" line; parse from there.
        with open(path, "r", encoding="utf-8", errors="replace") as tf:
            tf.seek(data_offset)
            flat = np.array(tf.read().split(), dtype=np.float64)
        ncols = len(fields)
        if flat.size == 0 or flat.size % ncols != 0:
            raise ModelReadError("PCD ascii 数据列数与 FIELDS 不符")
        arr = flat.reshape(-1, ncols)
        xi = name_idx.get("x", 0); yi = name_idx.get("y", 1); zi = name_idx.get("z", 2)
        xyz = arr[:, [xi, yi, zi]].astype(np.float32)
        rgb = None
        if "rgb" in name_idx:
            v = arr[:, name_idx["rgb"]]
            # PCL writes rgb in ascii as the float whose BIT PATTERN is the
            # packed (r,g,b) uint32 (a tiny denormal); other tools write the
            # integer value. Reinterpret bits when tiny, else treat as integer.
            if np.abs(v).max() < 1e-3:
                packed = v.astype("<f4").view(np.uint32).astype(np.int64)
            else:
                packed = v.astype(np.int64)
            rgb = np.stack([((packed >> 16) & 0xFF), ((packed >> 8) & 0xFF),
                            (packed & 0xFF)], axis=1).astype(np.uint8)
        return _finalize(xyz, rgb, source="PCD")

    # binary: build an offset-dtype (itemsize = full record length) so we read
    # only the fields we keep, with correct strides even with extra properties.
    keep = []
    off = 0
    for i, name in enumerate(fields):
        npname = _PCD_FIELD_NP.get(name.lower())
        field_len = sizes[i] * counts[i]
        if npname is not None and counts[i] == 1:
            t = types[i]
            if name.lower() in ("rgb", "rgba"):
                btype = "<u4"
            elif t == "F":
                btype = "<f4" if sizes[i] == 4 else "<f8"
            else:
                btype = "<" + _PCD_TYPE[sizes[i]]
            keep.append((npname[0], btype, off))
        off += field_len
    rec_size = off
    dt = np.dtype({"names": [k[0] for k in keep],
                   "formats": [k[1] for k in keep],
                   "offsets": [k[2] for k in keep],
                   "itemsize": rec_size})
    raw = np.memmap(path, dtype=dt, mode="r", offset=data_offset, shape=(n,))
    xyz = np.empty((n, 3), np.float32)
    xyz[:, 0] = raw["x"]; xyz[:, 1] = raw["y"]; xyz[:, 2] = raw["z"]
    rgb = None
    if "rgb" in dt.names:
        packed = np.asarray(raw["rgb"]).astype(np.int64)
        rgb = np.stack([((packed >> 16) & 0xFF), ((packed >> 8) & 0xFF), (packed & 0xFF)],
                       axis=1).astype(np.uint8)
    return _finalize(xyz, rgb, source="PCD")


# --------------------------------------------------------------------------- #
# Mesh surface sampler (shared by OBJ/STL/OFF/GLB)
# --------------------------------------------------------------------------- #
def _sample_mesh(verts, faces, colors=None, n=300_000, seed=1234):
    """Uniformly sample points on a triangle mesh, proportional to face area."""
    verts = np.asarray(verts, dtype=np.float32)
    if faces is None or len(faces) == 0:
        col = _rgb_to_uint8(colors) if colors is not None else None
        return verts, col
    faces = np.asarray(faces, dtype=np.int64)
    v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    area = np.maximum(area, 1e-12)
    p = area / area.sum()
    n = int(min(n, 1_000_000, max(1, len(faces) * 200)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(faces), size=n, p=p)
    r1 = rng.random(n); r2 = rng.random(n)
    flip = r1 + r2 > 1.0
    r1[flip] = 1.0 - r1[flip]; r2[flip] = 1.0 - r2[flip]
    pts = ((1 - r1 - r2)[:, None] * v0[idx]
           + r1[:, None] * v1[idx] + r2[:, None] * v2[idx]).astype(np.float32)
    col = None
    if colors is not None and len(colors) == len(verts):
        c0 = colors[faces[idx, 0]]; c1 = colors[faces[idx, 1]]; c2 = colors[faces[idx, 2]]
        interp = (1 - r1 - r2)[:, None] * c0 + r1[:, None] * c1 + r2[:, None] * c2
        col = _rgb_to_uint8(interp)      # handles float 0..1 AND 0..255
    return pts, col


# --------------------------------------------------------------------------- #
# OBJ
# --------------------------------------------------------------------------- #
def _read_obj(path, max_points=None):
    verts, vcolors, faces = [], [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
                if len(p) >= 7:
                    try:
                        vcolors.append([float(p[4]), float(p[5]), float(p[6])])
                    except ValueError:
                        vcolors.append([0.8, 0.8, 0.8])
                else:
                    vcolors.append([0.8, 0.8, 0.8])
            elif line.startswith("f "):
                fidx = []
                for tok in line.split()[1:]:
                    vi = tok.split("/")[0]
                    try:
                        iv = int(vi)
                        # OBJ indices are 1-based; negative ones count from the end
                        # (-1 == last vertex), so they map straight to numpy idx.
                        fidx.append(iv - 1 if iv > 0 else iv)
                    except ValueError:
                        pass
                # triangulate a polygon face as a fan
                for k in range(1, len(fidx) - 1):
                    faces.append([fidx[0], fidx[k], fidx[k + 1]])
    if not verts:
        raise ModelReadError("OBJ 文件没有顶点 (v)")
    verts = np.asarray(verts, dtype=np.float32)
    colors = (np.asarray(vcolors, dtype=np.float32) if len(vcolors) == len(verts) else None)
    faces = np.asarray(faces, dtype=np.int64) if faces else None
    pts, col = _sample_mesh(verts, faces, colors)
    return _finalize(pts, col, source="OBJ")


# --------------------------------------------------------------------------- #
# OFF
# --------------------------------------------------------------------------- #
def _read_off(path, max_points=None):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()
        if first.startswith("#"):
            first = f.readline().strip()
        toks = first.split()
        # OFF magic may sit alone ("OFF\n8 6 0") or share its line ("OFF 8 6 0").
        if toks and toks[0] in ("OFF", "COFF", "NOFF", "4OFF", "STOFF"):
            toks = toks[1:]
            if not toks:
                toks = f.readline().split()
        counts = toks
        nv, nf = int(counts[0]), int(counts[1])
        verts = np.array([f.readline().split()[:3] for _ in range(nv)], dtype=np.float32)
        faces = []
        for _ in range(nf):
            p = f.readline().split()
            k = int(p[0])
            idx = [int(x) for x in p[1:1 + k]]
            for j in range(1, k - 1):
                faces.append([idx[0], idx[j], idx[j + 1]])
    faces = np.asarray(faces, dtype=np.int64) if faces else None
    pts, col = _sample_mesh(verts, faces)
    return _finalize(pts, col, source="OFF")


# --------------------------------------------------------------------------- #
# STL (binary + ascii)
# --------------------------------------------------------------------------- #
def _read_stl_bin(path, max_points=None):
    with open(path, "rb") as f:
        f.read(80)
        ntri = struct.unpack("<I", f.read(4))[0]
    # Guard against a garbage/oversized count: cap by what the file can hold.
    expected = (os.path.getsize(path) - 84) // 50
    if ntri > expected:
        ntri = max(0, expected)
    if ntri <= 0:
        raise ModelReadError("STL 文件没有三角形")
    dt = np.dtype([("normal", "<f4", 3),
                   ("v0", "<f4", 3), ("v1", "<f4", 3), ("v2", "<f4", 3),
                   ("attr", "<u2")])
    with open(path, "rb") as f:
        f.seek(84)
        recs = np.fromfile(f, dtype=dt, count=ntri)
    # Interleave each triangle's 3 verts (v0,v1,v2 of tri i are contiguous) so
    # faces [[0,1,2],[3,4,5],...] map to the correct triangles.
    verts = np.stack([recs["v0"], recs["v1"], recs["v2"]], axis=1).reshape(-1, 3)
    faces = np.arange(len(verts)).reshape(-1, 3)
    pts, _ = _sample_mesh(verts, faces)
    return _finalize(pts, source="STL")


def _read_stl_ascii(path, max_points=None):
    verts, faces = [], []
    cur = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "vertex" and len(t) >= 4:
                cur.append([float(t[1]), float(t[2]), float(t[3])])
                if len(cur) == 3:
                    base = len(verts)
                    verts.extend(cur)
                    faces.append([base, base + 1, base + 2])
                    cur = []
    if not verts:
        raise ModelReadError("STL 文件没有顶点")
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    pts, _ = _sample_mesh(verts, faces)
    return _finalize(pts, source="STL")


# --------------------------------------------------------------------------- #
# SPLAT (antimatter15 .splat — 32 bytes/splat, pure numpy)
# --------------------------------------------------------------------------- #
def _read_splat(path, max_points=None):
    rec = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1"),
                    ("sx", "<f4"), ("sy", "<f4"), ("sz", "<f4"),
                    ("rx", "u1"), ("ry", "u1"), ("rz", "u1"), ("rw", "u1")])
    raw = np.memmap(path, dtype=rec, mode="r")
    n = len(raw)
    if n == 0:
        raise ModelReadError(".splat 文件为空或格式不符")
    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
    rgb = np.stack([raw["r"], raw["g"], raw["b"]], axis=1).astype(np.uint8)
    opacity = (raw["a"].astype(np.float32) / 255.0)
    scale = _norm_scale(np.stack([raw["sx"], raw["sy"], raw["sz"]], axis=1))
    return _finalize(xyz, rgb, opacity=opacity, scale=scale, is_gaussian=True,
                     source="SPLAT")


# --------------------------------------------------------------------------- #
# COLMAP points3D (.txt / .bin)
# --------------------------------------------------------------------------- #
def _read_colmap(path, max_points=None):
    # Decide by content, not extension: a generic .bin that isn't a COLMAP
    # points3D.bin must not be fed to the binary reader (it would alloc a
    # garbage-sized offset array).
    if _is_colmap_bin(path):
        return _read_colmap_bin(path)
    return _read_colmap_txt(path)


def _is_colmap_bin(path):
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        if len(head) < 8:
            return False
        n = struct.unpack("<Q", head)[0]
        return 0 < n < 200_000_000
    except (OSError, struct.error):
        return False


def _read_colmap_txt(path):
    arr = np.loadtxt(path, comments="#", usecols=(1, 2, 3, 4, 5, 6))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    xyz = arr[:, 0:3].astype(np.float32)
    rgb = arr[:, 3:6].astype(np.uint8)
    return _finalize(xyz, rgb, source="COLMAP")


def _read_colmap_bin(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        blob = f.read()
    # Pass 1: find each point's fixed-header offset (variable track length).
    # fixed part: id(i8,8) + xyz(f8,24) + rgb(u1,3) + err(f8,8) + tracklen(u8,8) = 51
    offs = np.empty(n, dtype=np.int64)
    cur = 0
    for i in range(n):
        offs[i] = cur
        tlen = struct.unpack_from("<Q", blob, cur + 43)[0]
        cur += 51 + 8 * tlen
    # Vectorised extraction of xyz (at off+8) and rgb (at off+32).
    xyz = np.empty((n, 3), dtype=np.float32)
    rgb = np.empty((n, 3), dtype=np.uint8)
    for i in range(n):
        o = offs[i]
        xyz[i] = struct.unpack_from("<3d", blob, o + 8)
        rgb[i] = struct.unpack_from("<3B", blob, o + 32)
    return _finalize(xyz, rgb, source="COLMAP")


# --------------------------------------------------------------------------- #
# LAS / LAZ (pure-numpy LAS; LAZ needs lazrs)
# --------------------------------------------------------------------------- #
def _read_las(path, max_points=None):
    with open(path, "rb") as f:
        head = f.read(375)
    if len(head) < 104:
        raise ModelReadError("LAS 文件头过短")
    ver_minor = head[21]
    # ASPRS Public Header Block field offsets (fixed by spec):
    # offset_to_point_data @96, point_format_id @104, point_record_length @105,
    # num_point_records @107, scales @131/139/147, offsets @155/163/171.
    offset_to_points = struct.unpack_from("<I", head, 96)[0]
    point_format = head[104]
    prl = struct.unpack_from("<H", head, 105)[0]
    nrec = struct.unpack_from("<I", head, 107)[0]
    xs, ys, zs = struct.unpack_from("<3d", head, 131)
    xo, yo, zo = struct.unpack_from("<3d", head, 155)
    if prl <= 0:
        raise ModelReadError(f"LAS point record length {prl} 无效")

    file_size = os.path.getsize(path)
    n_by_size = (file_size - offset_to_points) // prl
    n = n_by_size if nrec == 0 else min(nrec, n_by_size)
    if n <= 0:
        raise ModelReadError("LAS: 没有点记录")

    fields = [("X", "<i4", 0), ("Y", "<i4", 4), ("Z", "<i4", 8),
              ("intensity", "<u2", 12)]
    if point_format in (2,):
        fields += [("r", "<u2", 20), ("g", "<u2", 22), ("b", "<u2", 24)]
    elif point_format in (3, 5):
        fields += [("r", "<u2", 28), ("g", "<u2", 30), ("b", "<u2", 32)]
    dt = np.dtype({"names": [f[0] for f in fields],
                   "formats": [f[1] for f in fields],
                   "offsets": [f[2] for f in fields],
                   "itemsize": prl})
    raw = np.memmap(path, dtype=dt, mode="r", offset=offset_to_points, shape=(n,))
    xyz = np.empty((n, 3), dtype=np.float32)
    xyz[:, 0] = raw["X"] * xs + xo
    xyz[:, 1] = raw["Y"] * ys + yo
    xyz[:, 2] = raw["Z"] * zs + zo
    rgb = None
    if "r" in dt.names:
        rgb = np.stack([raw["r"], raw["g"], raw["b"]], axis=1)
        rgb = np.clip(rgb // 256, 0, 255).astype(np.uint8)   # 16-bit -> 8-bit
    else:
        inten = np.asarray(raw["intensity"], dtype=np.float32)
        if inten.max() > 0:
            g = np.clip(inten / max(inten.max(), 1.0) * 255.0, 0, 255).astype(np.uint8)
            rgb = np.stack([g, g, g], axis=1)
    return _finalize(xyz, rgb, source="LAS")


def _read_laz(path, max_points=None):
    try:
        import lazrs       # noqa: F401
    except ImportError:
        raise ModelReadError(_hint("lazrs", ".laz"))
    import lazrs
    las = lazrs.LasReader(path).read()
    xyz = np.stack([las.x, las.y, las.z], axis=1).astype(np.float32)
    rgb = None
    if hasattr(las, "rgb") and las.rgb is not None:
        rgb = np.stack([las.rgb[:, 0], las.rgb[:, 1], las.rgb[:, 2]], axis=1)
        rgb = np.clip(rgb // 256, 0, 255).astype(np.uint8)
    return _finalize(xyz, rgb, source="LAZ")


# --------------------------------------------------------------------------- #
# MagicaVoxel .vox
# --------------------------------------------------------------------------- #
def _read_vox(path, max_points=None):
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:4] != b"VOX ":
        raise ModelReadError("不是合法的 .vox 文件")
    palette = None
    xyzi = None

    # Chunks nest (SIZE/XYZI/RGBA are children of MAIN, possibly deep inside
    # nTRN/nGRP transforms), so walk recursively.
    def walk(start, end):
        nonlocal palette, xyzi
        i = start
        while i + 12 <= end:
            cid = blob[i:i + 4]
            csize, chsize = struct.unpack_from("<II", blob, i + 4)
            cstart = i + 12
            if cid == b"XYZI":
                num = struct.unpack_from("<I", blob, cstart)[0]
                xyzi = np.frombuffer(blob, dtype=np.uint8,
                                     count=num * 4, offset=cstart + 4).reshape(num, 4)
            elif cid == b"RGBA":
                palette = np.frombuffer(blob, dtype=np.uint8,
                                        count=256 * 4, offset=cstart).reshape(256, 4)
            walk(cstart + csize, cstart + csize + chsize)
            i = cstart + csize + chsize

    walk(8, len(blob))
    if xyzi is None:
        raise ModelReadError(".vox 文件没有体素数据 (XYZI)")
    xyz = xyzi[:, 0:3].astype(np.float32) + 0.5
    idx = xyzi[:, 3]
    if palette is not None:
        rgb = palette[np.clip(idx.astype(np.int32) - 1, 0, 255)][:, 0:3].copy()
    else:
        rgb = np.full((len(xyz), 3), 200, dtype=np.uint8)
    return _finalize(xyz, rgb, source="MagicaVoxel")


# --------------------------------------------------------------------------- #
# glTF / GLB (minimal: extract mesh POSITION, surface-sample)
# --------------------------------------------------------------------------- #
def _read_glb(path, max_points=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gltf":
        return _read_gltf(path)
    with open(path, "rb") as f:
        magic, version, length = struct.unpack_from("<III", f.read(12))
    if magic != 0x46546C67:
        raise ModelReadError("不是合法的 GLB 文件")
    json_chunk = None
    bin_buf = None
    off = 12
    with open(path, "rb") as f:
        f.seek(off)
        while off < length:
            clen, ctype = struct.unpack("<II", f.read(8))
            data = f.read(clen)
            if ctype == 0x4E4F534A:        # JSON
                json_chunk = json.loads(data.decode("utf-8"))
            elif ctype == 0x004E4942:      # BIN
                bin_buf = data
            # chunks are 4-byte padded; chunkLength excludes the padding.
            pad = (4 - (clen % 4)) % 4
            if pad:
                f.read(pad)
            off += 8 + clen + pad
    if json_chunk is None:
        raise ModelReadError("GLB 缺少 JSON 块")
    return _gltf_extract(json_chunk, {"binary": bin_buf}, os.path.dirname(path))


def _read_gltf(path):
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    buffers = {}
    base = os.path.dirname(path)
    for i, b in enumerate(spec.get("buffers", [])):
        uri = b.get("uri", "")
        if uri.startswith("data:"):
            buffers[i] = _decode_data_uri(uri)
        else:
            with open(os.path.join(base, uri), "rb") as bf:
                buffers[i] = bf.read()
    return _gltf_extract(spec, buffers, base)


_COMP = {5120: "i1", 5121: "u1", 5122: "i2", 5123: "u2", 5125: "u4", 5126: "f4"}
_COMPSIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _gltf_accessor_data(spec, buffers, acc_idx):
    acc = spec["accessors"][acc_idx]
    bv = spec["bufferViews"][acc["bufferView"]]
    buf = buffers.get(bv.get("buffer", 0)) or buffers.get("binary")
    comp = _COMP[acc["componentType"]]
    csize = _COMPSIZE[acc["componentType"]]
    ncomp = _NCOMP[acc["type"]]
    count = acc["count"]
    base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride", 0)
    esize = ncomp * csize
    if stride and stride != esize:
        # interleaved vertex attributes: gather each element at its stride
        out = np.empty((count, ncomp), dtype=comp)
        for i in range(count):
            out[i] = np.frombuffer(buf, dtype=comp, count=ncomp,
                                   offset=base + i * stride)
        arr = out
    else:
        arr = np.frombuffer(buf, dtype=comp, count=count * ncomp, offset=base)
        if ncomp > 1:
            arr = arr.reshape(count, ncomp)
    if acc.get("normalized") and comp not in ("f4", "f8"):
        info = np.iinfo(np.dtype(comp))
        denom = float(info.max) if info.kind == "u" else float(-info.min)
        arr = arr.astype(np.float32) / denom
    return arr


def _triangulate(idx: np.ndarray, mode: int):
    """glTF index array -> triangle indices (M,3) for TRIANGLES/STRIP/FAN."""
    idx = np.asarray(idx)
    if mode == 4:                                # TRIANGLES
        return idx.reshape(-1, 3)
    if mode == 5 and len(idx) >= 3:              # TRIANGLE_STRIP
        return np.stack([idx[:-2], idx[1:-1], idx[2:]], axis=1)
    if mode == 6 and len(idx) >= 3:              # TRIANGLE_FAN
        return np.stack([np.full(len(idx) - 2, idx[0]), idx[1:-1], idx[2:]], axis=1)
    return None                                  # POINTS/LINES -> no triangles


def _gltf_extract(spec, buffers, base):
    verts_all, faces_all, col_all = [], [], []
    voff = 0
    for mesh in spec.get("meshes", []):
        for prim in mesh["primitives"]:
            pos_i = prim["attributes"].get("POSITION")
            if pos_i is None:
                continue
            mode = prim.get("mode", 4)
            verts = _gltf_accessor_data(spec, buffers, pos_i).astype(np.float32)
            col = None
            if "COLOR_0" in prim["attributes"]:
                col = _gltf_accessor_data(spec, buffers,
                                          prim["attributes"]["COLOR_0"]).astype(np.float32)
            if "indices" in prim:
                idx = _gltf_accessor_data(spec, buffers, prim["indices"]).ravel()
            else:
                idx = np.arange(len(verts))     # non-indexed primitive
            try:
                tri = _triangulate(idx, mode) if mode in (4, 5, 6) else None
            except ValueError:
                tri = None                       # malformed index count -> verts as points
            faces = (tri + voff) if tri is not None and len(tri) else None
            verts_all.append(verts)
            if col is not None:
                col_all.append(col)
            if faces is not None:
                faces_all.append(faces)
            voff += len(verts)
    if not verts_all:
        raise ModelReadError("glTF 中没有网格 POSITION 属性")
    verts = np.concatenate(verts_all, axis=0)
    colors = np.concatenate(col_all, axis=0) if col_all else None
    faces = np.concatenate(faces_all, axis=0) if faces_all else None
    pts, col = _sample_mesh(verts, faces, colors)
    return _finalize(pts, col, source="glTF")


def _decode_data_uri(uri):
    import base64
    head, b64 = uri.split(",", 1)
    return base64.b64decode(b64)


# --------------------------------------------------------------------------- #
# NumPy .npy / .npz  +  adaptive array extractor (the "pt 等" core)
# --------------------------------------------------------------------------- #
def _read_npy(path, max_points=None):
    arr = np.load(path, allow_pickle=False)
    return _array_to_cloud(arr, source="NumPy")


def _read_npz(path, max_points=None):
    with np.load(path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    return _extract_cloud(d, source="NumPy")


def _array_to_cloud(arr, source="NumPy"):
    arr = np.asarray(arr)
    if arr.dtype == object:
        # allow_pickle single dict/array
        if isinstance(arr.item(), dict):
            return _extract_cloud(arr.item(), source=source)
        arr = np.asarray(arr.item())
    if arr.ndim == 1 and arr.size % 3 == 0:
        arr = arr.reshape(-1, 3)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        # could be an RGB image or a (D,H,3) — not a point cloud.
        raise ModelReadError("NumPy 数组形状 (H,W,3) 像图像而非点云；请提供 (N,3) 点云或体素网格")
    if arr.ndim == 3 and arr.shape[-1] != 3:
        # voxel grid: occupied where > threshold
        return _voxel_grid_to_cloud(arr, source=source)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ModelReadError(f"NumPy 数组形状 {arr.shape} 无法解释为点云")
    xyz = arr[:, 0:3].astype(np.float32)
    rgb = None
    if arr.shape[1] >= 6:
        cand = arr[:, 3:6]
        if cand.dtype.kind == "f" and float(cand.min()) >= -1e-3 and float(cand.max()) <= 1.0 + 1e-6:
            rgb = cand
        elif cand.dtype.kind in "iu" and float(cand.min()) >= 0 and float(cand.max()) <= 255:
            rgb = cand
    return _finalize(xyz, rgb, source=source)


def _voxel_grid_to_cloud(grid, source="NumPy"):
    grid = np.asarray(grid)
    thr = max(np.median(grid) + 0.5 * grid.std(), 1e-9)
    zz, yy, xx = np.where(grid > thr)
    if len(xx) == 0:
        raise ModelReadError("体素网格中没有占据的体素")
    xyz = np.stack([xx, yy, zz], axis=1).astype(np.float32)
    return _finalize(xyz, source=source)


# ---- adaptive extractor for a bag of named arrays (.pt/.npz/.h5) ---- #
def _flatten_arrays(obj, prefix=""):
    """Yield (name, ndarray) for every numpy array nested in obj."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten_arrays(v, f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(_flatten_arrays(v, f"{prefix}[{i}]"))
    else:
        arr = np.asarray(obj)
        if arr.dtype != object and arr.size > 0:
            out[prefix] = arr
    return out


def _find_by_name(arrays, patterns, shape_ndim=None, shape_last=None):
    """Return the array whose (lowercased) name matches any regex in patterns."""
    import re
    rx = [re.compile(p, re.I) for p in patterns]
    for name, arr in arrays.items():
        low = name.lower()
        if any(r.search(low) for r in rx):
            if shape_ndim is not None and arr.ndim != shape_ndim:
                continue
            if shape_last is not None and (arr.ndim < 2 or arr.shape[-1] != shape_last):
                continue
            return name, arr
    return None, None


def _extract_cloud(d, source="tensor") -> PlyData:
    arrays = _flatten_arrays(d)
    if not arrays:
        raise ModelReadError("文件中没有数组数据")

    # ---- 3D Gaussian Splatting checkpoint? ---- #
    xyz_key, xyz = _find_by_name(arrays, [r"(^|[_.])means3d$", r"_xyz$", r"(^|[_.])positions?$",
                                     r"(^|[_.])means$", r"(^|[_.])location$"], 2, 3)
    op_key, op = _find_by_name(arrays, [r"opacity", r"_opacity$", r"alphas?$"], None, None)
    sc_key, sc = _find_by_name(arrays, [r"log_scales", r"_scales$", r"scales$", r"_scaling",
                                        r"scaling"], None, None)
    if xyz_key is not None and op_key is not None and sc_key is not None:
        xyz = xyz.astype(np.float32)
        opacity = _norm_opacity(op)
        # 3DGS stores scale as log-scale (gaussian-splatting `_scales`/`_scaling`,
        # `log_scales`); exp it. _norm_scale falls back to linear if exp blows up.
        scale = _norm_scale(sc, is_log=True)
        # colour: SH DC (features_dc / f_dc) or precomputed colours
        dc_key, dc = _find_by_name(arrays, [r"features_dc", r"f_dc"], 3, None)
        col_key, col = _find_by_name(arrays, [r"colors_precomp", r"colou?rs?$", r"rgb$",
                                              r"features_precomp"], 2, 3)
        rgb = None
        if dc is not None:
            rgb = _sh_dc_to_rgb(_sh_chan(dc, 0), _sh_chan(dc, 1), _sh_chan(dc, 2))
        elif col is not None:
            rgb = _rgb_to_uint8(col)
        return _finalize(xyz, rgb, opacity=opacity, scale=scale, is_gaussian=True,
                         source=source + "/3DGS")

    # ---- generic: pick the best (N,3) point array ---- #
    xyz_key, xyz = _find_by_name(arrays, [r"xyz$", r"points?$", r"positions?$",
                                          r"vertices?$", r"coords?$", r"location$",
                                          r"pos$", r"means$"], 2, 3)
    if xyz is None:
        # fall back to the largest (N,3) array
        cands = [(n, a) for n, a in arrays.items() if a.ndim == 2 and a.shape[-1] == 3]
        if not cands:
            # try three 1-D x/y/z arrays
            xk, xv = _find_by_name(arrays, [r"(^|[_.])x$"], 1, None)
            yk, yv = _find_by_name(arrays, [r"(^|[_.])y$"], 1, None)
            zk, zv = _find_by_name(arrays, [r"(^|[_.])z$"], 1, None)
            if xv is not None and yv is not None and zv is not None:
                xyz = np.stack([xv, yv, zv], axis=1)
            else:
                raise ModelReadError("未能在数组中找到 (N,3) 点坐标（xyz/points/positions…）")
        else:
            xyz_key, xyz = max(cands, key=lambda kv: kv[1].shape[0])
    n = xyz.shape[0]
    xyz = xyz.astype(np.float32)

    rgb = None
    ck, col = _find_by_name(arrays, [r"rgb$", r"colou?rs?$", r"colou?r$", r"albedo$"], 2, 3)
    if col is not None and col.shape[0] == n:
        rgb = _rgb_to_uint8(col)
    nk, nrm = _find_by_name(arrays, [r"normals?$"], 2, 3)
    normals = nrm.astype(np.float32) if (nrm is not None and nrm.shape[0] == n) else None
    ok, op = _find_by_name(arrays, [r"opacity", r"alphas?$"], None, None)
    opacity = _norm_opacity(op) if (op is not None and op.shape[0] == n) else None
    sk, sc = _find_by_name(arrays, [r"scales?$", r"radii$"], None, None)
    scale = None
    if sc is not None and sc.shape[0] == n:
        scale = _norm_scale(sc, is_log=("log" in (sk or "").lower()))
    return _finalize(xyz, rgb, normals=normals, opacity=opacity, scale=scale,
                     source=source)


def _sh_chan(dc, i):
    """Return channel i of an SH-DC array shaped (N,3,1) / (N,1,3) / (N,3)."""
    dc = np.asarray(dc, dtype=np.float64)
    if dc.ndim == 3:
        if dc.shape[1] == 3:
            return dc[:, i, 0]
        if dc.shape[2] == 3:
            return dc[:, 0, i]
    if dc.ndim == 2 and dc.shape[1] == 3:
        return dc[:, i]
    raise ModelReadError("无法解释 SH DC 系数数组的形状")


# --------------------------------------------------------------------------- #
# Optional-lib readers
# --------------------------------------------------------------------------- #
def _read_torch(path, max_points=None):
    try:
        import torch
    except ImportError:
        raise ModelReadError(_hint("torch", os.path.splitext(path)[1]))
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    elif hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    # convert tensors to numpy
    def to_np(o):
        if isinstance(o, dict):
            return {k: to_np(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [to_np(v) for v in o]
        if hasattr(o, "detach"):
            return o.detach().cpu().numpy()
        return o
    return _extract_cloud(to_np(obj), source="PyTorch")


def _read_h5(path, max_points=None):
    try:
        import h5py
    except ImportError:
        raise ModelReadError(_hint("h5py", ".h5/.hdf5"))
    d = {}

    def visit(name, obj):
        if hasattr(obj, "shape"):
            d[name] = np.asarray(obj)
    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return _extract_cloud(d, source="HDF5")


def _read_gaussforge(path, fmt_label):
    try:
        import gaussforge
    except ImportError:
        raise ModelReadError(_hint("gaussforge", os.path.splitext(path)[1]))
    cloud = gaussforge.read(path)
    xyz = np.asarray(cloud.positions, dtype=np.float32)
    rgb = None
    if getattr(cloud, "colors", None) is not None:
        rgb = _rgb_to_uint8(np.asarray(cloud.colors))
    opacity = getattr(cloud, "opacities", None)
    opacity = np.asarray(opacity, dtype=np.float32) if opacity is not None else None
    scale = getattr(cloud, "scales", None)
    scale = _norm_scale(np.asarray(scale)) if scale is not None else None
    return _finalize(xyz, rgb, opacity=opacity, scale=scale, is_gaussian=True,
                     source=fmt_label)


# --------------------------------------------------------------------------- #
# Reader registry
# --------------------------------------------------------------------------- #
_READERS = {
    "ply": _read_ply,
    "text": _read_text,
    "pcd": _read_pcd,
    "obj": _read_obj,
    "stl_bin": _read_stl_bin,
    "stl_ascii": _read_stl_ascii,
    "off": _read_off,
    "splat": _read_splat,
    "colmap": _read_colmap,
    "colmap_bin": _read_colmap_bin,
    "colmap_txt": _read_colmap_txt,
    "las": _read_las,
    "laz": _read_laz,
    "vox": _read_vox,
    "glb": _read_glb,
    "gltf": _read_gltf,
    "npy": _read_npy,
    "npz": _read_npz,
    "pt": _read_torch,
    "h5": _read_h5,
    "ksplat": lambda p, **kw: _read_gaussforge(p, "KSPLAT"),
    "spz": lambda p, **kw: _read_gaussforge(p, "SPZ"),
    "sog": lambda p, **kw: _read_gaussforge(p, "SOG"),
}


# --------------------------------------------------------------------------- #
# Supported-extension registry (for the file dialog)
# --------------------------------------------------------------------------- #
ALWAYS_EXTENSIONS = [
    ".ply", ".xyz", ".xyzrgb", ".xyzn", ".pts", ".csv", ".asc", ".txt",
    ".pcd", ".obj", ".stl", ".off", ".las", ".splat",
    ".npy", ".npz", ".vox", ".glb", ".gltf", ".bin",
]
OPTIONAL_EXTENSIONS = {
    "torch": [".pt", ".pth"],
    "h5py": [".h5", ".hdf5"],
    "lazrs": [".laz"],
    "gaussforge": [".ksplat", ".spz", ".sog"],
}


def _have(mod: str) -> bool:
    try:
        import importlib
        importlib.import_module(mod)
        return True
    except ImportError:
        return False


def supported_extensions() -> list[str]:
    exts = list(ALWAYS_EXTENSIONS)
    if _have("torch"):
        exts += OPTIONAL_EXTENSIONS["torch"]
    if _have("h5py"):
        exts += OPTIONAL_EXTENSIONS["h5py"]
    if _have("lazrs"):
        exts += OPTIONAL_EXTENSIONS["lazrs"]
    if _have("gaussforge"):
        exts += OPTIONAL_EXTENSIONS["gaussforge"]
    return exts


def format_groups() -> list[tuple[str, list[str]]]:
    """Return (label, extensions) groups for a grouped file dialog."""
    groups = [
        ("点云", [".ply", ".pcd", ".las", ".xyz", ".xyzrgb", ".xyzn",
                  ".pts", ".csv", ".asc", ".txt"]),
        ("网格（自动曲面采样）", [".obj", ".stl", ".off", ".glb", ".gltf"]),
        ("Gaussian Splatting", [".splat"]),
        ("张量 / 数组", [".npy", ".npz"]),
        ("体素", [".vox"]),
        ("COLMAP", [".bin"]),
    ]
    if _have("torch"):
        groups.append(("PyTorch 检查点", [".pt", ".pth"]))
    if _have("h5py"):
        groups.append(("HDF5", [".h5", ".hdf5"]))
    if _have("lazrs"):
        groups.append(("LAZ", [".laz"]))
    if _have("gaussforge"):
        groups.append(("GaussForge", [".ksplat", ".spz", ".sog"]))
    return groups


def is_supported(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in supported_extensions() or ext == ".bin"
