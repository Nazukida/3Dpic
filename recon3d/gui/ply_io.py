"""Fast, dependency-light PLY reader tuned for the viewer.

Design goals:

* **Speed** - binary bodies are read in a single ``np.fromfile`` using a
  structured dtype that mirrors the file's property layout, then the columns
  we care about (xyz / rgb / normals) are sliced out.  No per-vertex Python.
* **Tolerance** - handles ``binary_little_endian``, ``binary_big_endian`` and
  ``ascii`` formats, arbitrary extra properties, colours as ``uchar`` or
  ``float``, and the common ``diffuse_red`` naming.
* **Predictable output** - always returns contiguous arrays:
  ``xyz`` float32 (N,3); ``rgb`` uint8 (N,3) or ``None``; ``normals`` float32
  (N,3) or ``None``.

Only vertex elements are decoded (faces are skipped) - this is a point-cloud
viewer.  Writing is handled by :func:`recon3d.io_utils.write_ply`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# PLY scalar type -> numpy base type (endianness applied later).
_PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


@dataclass
class PlyData:
    xyz: np.ndarray                    # (N, 3) float32
    rgb: np.ndarray | None            # (N, 3) uint8 or None
    normals: np.ndarray | None        # (N, 3) float32 or None
    count: int
    opacity: np.ndarray | None = None  # (N,) float32 in [0,1] or None
    scale: np.ndarray | None = None    # (N,) float32 (avg radius) or None
    is_gaussian: bool = False          # True if loaded from a 3DGS source
    source: str = ""                   # detected format name (for UI status)

    @property
    def has_color(self) -> bool:
        return self.rgb is not None

    @property
    def has_normals(self) -> bool:
        return self.normals is not None


class PlyError(Exception):
    pass


def _read_header(f) -> tuple[str, int, list[tuple[str, str]], list[str]]:
    """Parse the ASCII header.

    Returns ``(fmt, vertex_count, vertex_props, other_element_lines)`` where
    ``vertex_props`` is a list of ``(name, ply_type)`` in file order.
    """
    magic = f.readline()
    if magic.strip() not in (b"ply", b"PLY"):
        raise PlyError("not a PLY file (missing 'ply' magic)")

    fmt = None
    elements: list[tuple[str, int]] = []       # (element_name, count)
    props: dict[str, list[tuple[str, str]]] = {}
    cur_elem = None

    while True:
        line = f.readline()
        if not line:
            raise PlyError("unexpected EOF in header")
        toks = line.split()
        if not toks:
            continue
        key = toks[0]
        if key == b"format":
            fmt = toks[1].decode("ascii", "replace")
        elif key == b"comment" or key == b"obj_info":
            continue
        elif key == b"element":
            cur_elem = toks[1].decode("ascii", "replace")
            elements.append((cur_elem, int(toks[2])))
            props[cur_elem] = []
        elif key == b"property":
            if cur_elem is None:
                continue
            if toks[1] == b"list":
                # e.g. face: property list uchar int vertex_indices
                # encode count/value scalar types into the type string so the
                # body-skip logic can advance past it correctly.
                ctype = toks[2].decode("ascii", "replace") if len(toks) > 2 else "uchar"
                vtype = toks[3].decode("ascii", "replace") if len(toks) > 3 else "int"
                pname = toks[4].decode("ascii", "replace") if len(toks) > 4 else "__list__"
                props[cur_elem].append((pname, f"list:{ctype}:{vtype}"))
            else:
                ptype = toks[1].decode("ascii", "replace")
                pname = toks[2].decode("ascii", "replace")
                props[cur_elem].append((pname, ptype))
        elif key == b"end_header":
            break

    if fmt is None:
        raise PlyError("missing 'format' line in header")

    # Ordered element metadata: (name, count, props). Body order follows header
    # declaration order per the PLY spec — vertex need not be first.
    elements_full = [(name, cnt, props.get(name, [])) for name, cnt in elements]

    vcount = 0
    vprops: list[tuple[str, str]] = []
    for name, cnt in elements:
        if name == "vertex":
            vcount = cnt
            vprops = props.get("vertex", [])
            break
    if not vprops:
        raise PlyError("PLY has no 'vertex' element")

    return fmt, vcount, vprops, elements_full


def _color_scale(dtype_char: str) -> float:
    """Colours stored as float are usually 0..1; ints are already 0..255."""
    return 255.0 if dtype_char in ("f4", "f8") else 1.0


def _extract(struct_arr: np.ndarray, names_present: set[str],
             candidates: list[str]) -> str | None:
    for c in candidates:
        if c in names_present:
            return c
    return None


def _is_gaussian_splatting(vprops: list[tuple[str, str]]) -> bool:
    """Detect if the PLY is a 3D Gaussian Splatting export.

    3DGS PLYs store colour as spherical harmonics DC coefficients (f_dc_0/1/2)
    and have opacity + scale properties, rather than conventional red/green/blue.
    """
    names = {p[0] for p in vprops}
    return "f_dc_0" in names and "opacity" in names


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for opacity activation."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _sh_dc_to_rgb(dc0: np.ndarray, dc1: np.ndarray, dc2: np.ndarray) -> np.ndarray:
    """Convert SH degree-0 (DC) coefficients to sRGB [0,255] uint8.

    The SH DC basis constant is C0 = 0.28209479... The stored value is the
    coefficient; the actual colour is ``coeff * C0 + 0.5``.
    """
    C0 = 0.28209479177387814
    r = dc0 * C0 + 0.5
    g = dc1 * C0 + 0.5
    b = dc2 * C0 + 0.5
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb


def read_ply(path: str, max_points: int | None = None) -> PlyData:
    """Load a PLY point cloud.

    Automatically detects 3D Gaussian Splatting format (spherical harmonics
    colour, opacity, scale) and converts to a renderable representation.

    ``max_points`` optionally caps the number of points via uniform striding
    (cheap, order-preserving) - useful as a guard for enormous files.

    Binary bodies are read via a memory-mapped view (see :func:`_read_binary`)
    so a 3.8M-vertex / 200-byte-record file never materialises its full ~750 MB
    structured array in RAM — only the xyz/rgb/normals columns we keep are
    allocated (~60 MB). ASCII falls back to a buffered read.
    """
    with open(path, "rb") as f:
        fmt, vcount, vprops, elements = _read_header(f)
        is_ascii = (fmt == "ascii")
        big_endian = "big" in fmt

        # Body order follows header declaration order. Skip every element
        # declared before 'vertex' so we don't misread its bytes/lines as
        # vertex data (vertex is not required to be first per the PLY spec).
        for name, cnt, eprops in elements:
            if name == "vertex":
                break
            _skip_element(f, cnt, eprops, is_ascii, big_endian)

        if is_ascii:
            xyz, rgb, nrm = _read_ascii(f, vcount, vprops)
        else:
            # Byte offset of the vertex body within the file. The file handle
            # is positioned there after skipping preceding elements.
            vertex_offset = f.tell()
            xyz, rgb, nrm = _read_binary(path, vertex_offset, vcount, vprops,
                                         big_endian)

    # --- 3D Gaussian Splatting detection & conversion ---
    is_gaussian = _is_gaussian_splatting(vprops)
    opacity = None
    scale = None

    if is_gaussian:
        # Re-read the structured data for GS-specific properties.
        # We need the raw field values for f_dc, opacity, scale.
        with open(path, "rb") as f:
            _fmt, _vc, _vp, _elems = _read_header(f)
            for name, cnt, eprops in _elems:
                if name == "vertex":
                    break
                _skip_element(f, cnt, eprops, is_ascii, big_endian)

            if not is_ascii:
                gs_data = _read_gaussian_binary(f, vcount, vprops, big_endian)
            else:
                gs_data = _read_gaussian_ascii(f, vcount, vprops)

        rgb = gs_data["rgb"]
        opacity = gs_data["opacity"]
        scale = gs_data["scale"]

    n = len(xyz)
    if max_points is not None and 0 < max_points < n:
        step = int(np.ceil(n / max_points))
        idx = np.arange(0, n, step)
        xyz = xyz[idx]
        rgb = rgb[idx] if rgb is not None else None
        nrm = nrm[idx] if nrm is not None else None
        opacity = opacity[idx] if opacity is not None else None
        scale = scale[idx] if scale is not None else None

    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    if rgb is not None:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if nrm is not None:
        nrm = np.ascontiguousarray(nrm, dtype=np.float32)
    if opacity is not None:
        opacity = np.ascontiguousarray(opacity, dtype=np.float32)
    if scale is not None:
        scale = np.ascontiguousarray(scale, dtype=np.float32)
    return PlyData(xyz=xyz, rgb=rgb, normals=nrm, count=len(xyz),
                   opacity=opacity, scale=scale, is_gaussian=is_gaussian)


def _read_gaussian_binary(f, vcount, vprops, big_endian) -> dict:
    """Extract 3DGS-specific fields (SH colour, opacity, scale) from binary."""
    order = ">" if big_endian else "<"
    fields = []
    for name, ptype in vprops:
        base = _PLY_TYPES.get(ptype)
        if base is None:
            raise PlyError(f"unknown property type '{ptype}'")
        fields.append((name, order + base))
    dt = np.dtype(fields)
    data = np.fromfile(f, dtype=dt, count=vcount)
    if len(data) < vcount:
        vcount = len(data)

    names = {p[0] for p in vprops}

    # Colour from SH DC coefficients
    rgb = None
    if "f_dc_0" in names and "f_dc_1" in names and "f_dc_2" in names:
        dc0 = data["f_dc_0"].astype(np.float64)
        dc1 = data["f_dc_1"].astype(np.float64)
        dc2 = data["f_dc_2"].astype(np.float64)
        rgb = _sh_dc_to_rgb(dc0, dc1, dc2)

    # Opacity (stored as inverse sigmoid)
    opacity = None
    if "opacity" in names:
        raw_opacity = data["opacity"].astype(np.float64)
        opacity = _sigmoid(raw_opacity).astype(np.float32)

    # Scale (stored as log): average of the 3 axes gives a representative radius
    scale = None
    if "scale_0" in names and "scale_1" in names and "scale_2" in names:
        s0 = np.exp(data["scale_0"].astype(np.float64))
        s1 = np.exp(data["scale_1"].astype(np.float64))
        s2 = np.exp(data["scale_2"].astype(np.float64))
        scale = ((s0 + s1 + s2) / 3.0).astype(np.float32)

    return {"rgb": rgb, "opacity": opacity, "scale": scale}


def _read_gaussian_ascii(f, vcount, vprops) -> dict:
    """Extract 3DGS-specific fields from ASCII format."""
    names = [p[0] for p in vprops]
    col = {name: i for i, name in enumerate(names)}
    ncols = len(names)
    name_set = set(names)

    rows = []
    read = 0
    for line in f:
        if not line.strip():
            continue
        rows.append(line)
        read += 1
        if read >= vcount:
            break
    if not rows:
        raise PlyError("no vertex data found")

    buf = b" ".join(r.strip() for r in rows)
    flat = np.array(buf.split(), dtype=np.float64)
    if flat.size < read * ncols:
        raise PlyError("ASCII vertex data has fewer values than declared")
    flat = flat[: read * ncols].reshape(read, ncols)

    # Colour from SH DC
    rgb = None
    if "f_dc_0" in name_set and "f_dc_1" in name_set and "f_dc_2" in name_set:
        dc0 = flat[:, col["f_dc_0"]]
        dc1 = flat[:, col["f_dc_1"]]
        dc2 = flat[:, col["f_dc_2"]]
        rgb = _sh_dc_to_rgb(dc0, dc1, dc2)

    # Opacity
    opacity = None
    if "opacity" in name_set:
        raw_opacity = flat[:, col["opacity"]]
        opacity = _sigmoid(raw_opacity).astype(np.float32)

    # Scale
    scale = None
    if "scale_0" in name_set and "scale_1" in name_set and "scale_2" in name_set:
        s0 = np.exp(flat[:, col["scale_0"]])
        s1 = np.exp(flat[:, col["scale_1"]])
        s2 = np.exp(flat[:, col["scale_2"]])
        scale = ((s0 + s1 + s2) / 3.0).astype(np.float32)

    return {"rgb": rgb, "opacity": opacity, "scale": scale}


def _resolve_columns(vprops: list[tuple[str, str]]):
    names_present = {p[0] for p in vprops}
    x = _extract(None, names_present, ["x"])
    y = _extract(None, names_present, ["y"])
    z = _extract(None, names_present, ["z"])
    if not (x and y and z):
        raise PlyError("vertex element missing x/y/z properties")
    r = _extract(None, names_present, ["red", "r", "diffuse_red"])
    g = _extract(None, names_present, ["green", "g", "diffuse_green"])
    b = _extract(None, names_present, ["blue", "b", "diffuse_blue"])
    nx = _extract(None, names_present, ["nx", "normal_x"])
    ny = _extract(None, names_present, ["ny", "normal_y"])
    nz = _extract(None, names_present, ["nz", "normal_z"])
    has_rgb = bool(r and g and b)
    has_nrm = bool(nx and ny and nz)
    return (x, y, z), (r, g, b) if has_rgb else None, (nx, ny, nz) if has_nrm else None


def _skip_element(f, count, eprops, is_ascii, big_endian):
    """Advance the file cursor past an element body we don't decode.

    Handles fixed-size scalar layouts (fast seek) and list properties (e.g.
    faces: ``property list uchar int vertex_indices``), for both ASCII and
    binary bodies, so an element declared before ``vertex`` never bleeds into
    the vertex records.
    """
    if count <= 0:
        return
    if is_ascii:
        # one element instance per line; list properties are still one line
        read = 0
        while read < count:
            line = f.readline()
            if not line:
                break
            if line.strip():
                read += 1
        return

    # binary: if the element has only fixed-size scalars, seek past it in one
    # shot; otherwise (list properties) walk each record field by field.
    order = ">" if big_endian else "<"
    endian = "big" if big_endian else "little"
    has_list = any(str(ptype).startswith("list") for _, ptype in eprops)

    if not has_list:
        scalar = [order + _PLY_TYPES[ptype] for _, ptype in eprops
                  if ptype in _PLY_TYPES]
        if len(scalar) != len(eprops):
            has_list = True   # unknown scalar type -> fall through to safe path
        else:
            rec_size = int(np.dtype([(f"f{i}", t) for i, t in enumerate(scalar)]).itemsize)
            f.seek(rec_size * count, 1)
            return

    for _ in range(count):
        for _, ptype in eprops:
            if str(ptype).startswith("list"):
                # "list:<count_type>:<value_type>": read the count, skip values
                _, ctype, vtype = ptype.split(":")
                cn = np.dtype(_PLY_TYPES.get(ctype, "u1")).itemsize
                vn = np.dtype(_PLY_TYPES.get(vtype, "i4")).itemsize
                n = int.from_bytes(f.read(cn), endian)
                f.seek(vn * n, 1)
            else:
                f.seek(np.dtype(_PLY_TYPES.get(ptype, "f4")).itemsize, 1)


def _read_binary(path: str, offset: int, vcount: int,
                 vprops: list[tuple[str, str]], big_endian: bool):
    """Decode a binary vertex body via a memory-mapped, copy-free view.

    ``path``/``offset`` locate the first vertex record in the file. The body is
    exposed as a structured ``np.memmap`` view (Windows allocation-granularity
    handled by mapping the aligned base and slicing the view by ``delta``), so
    the multi-hundred-MB structured array is **never allocated in RAM** — only
    the xyz/rgb/normals columns we copy out are. Falls back to a plain
    ``np.fromfile`` if memory-mapping is unavailable (e.g. on a pipe).
    """
    if any(str(t).startswith("list") for _, t in vprops):
        # vertex element with a list property is pathological; fall back slow.
        raise PlyError("list properties in vertex element are not supported")

    order = ">" if big_endian else "<"
    fields = []
    ptype_by_name = {}
    for name, ptype in vprops:
        base = _PLY_TYPES.get(ptype)
        if base is None:
            raise PlyError(f"unknown property type '{ptype}'")
        fields.append((name, order + base))
        ptype_by_name[name] = base
    dt = np.dtype(fields)
    rec_size = dt.itemsize

    data = _memmap_struct(path, dt, offset, vcount)
    if data is None:
        # mmap unavailable (unusual path/pipe): classic buffered read.
        with open(path, "rb") as f:
            f.seek(offset)
            data = np.fromfile(f, dtype=dt, count=vcount)

    got = len(data)
    if got < vcount:
        vcount = got  # truncated file - use what we got rather than crashing

    (xn, yn, zn), rgb_names, nrm_names = _resolve_columns(vprops)
    # Copy the columns we keep out of the (memmap-backed) structured view into
    # ordinary contiguous arrays. The assignment gathers strided data into a
    # dense buffer, so the memmap pages are read once and then released.
    xyz = np.empty((vcount, 3), dtype=np.float32)
    xyz[:, 0] = data[xn][:vcount]
    xyz[:, 1] = data[yn][:vcount]
    xyz[:, 2] = data[zn][:vcount]

    rgb = None
    if rgb_names:
        rn, gn, bn = rgb_names
        scale = _color_scale(ptype_by_name[rn])
        rgb = np.empty((vcount, 3), dtype=np.float32)
        rgb[:, 0] = data[rn][:vcount]
        rgb[:, 1] = data[gn][:vcount]
        rgb[:, 2] = data[bn][:vcount]
        if scale != 1.0:
            rgb *= scale
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    nrm = None
    if nrm_names:
        nxn, nyn, nzn = nrm_names
        nrm = np.empty((vcount, 3), dtype=np.float32)
        nrm[:, 0] = data[nxn][:vcount]
        nrm[:, 1] = data[nyn][:vcount]
        nrm[:, 2] = data[nzn][:vcount]

    del data  # drop the memmap view; the returned arrays are independent copies
    return xyz, rgb, nrm


def _memmap_struct(path: str, dt: np.dtype, offset: int, count: int):
    """Return a read-only structured view of ``count`` records at ``offset``.

    Windows requires mmap offsets to be a multiple of the allocation
    granularity (64 KiB); a PLY header puts the vertex body at an arbitrary
    byte offset, so we map the aligned base and create a strided ``np.ndarray``
    view that starts ``delta`` bytes in. The view shares the mapped pages — no
    copy of the body is made. Returns ``None`` if the file cannot be mapped.
    """
    import mmap
    rec_size = dt.itemsize
    try:
        ag = getattr(mmap, "ALLOCATIONGRANULARITY", 65536) or 65536
        base = (offset // ag) * ag
        delta = offset - base
        raw = np.memmap(path, dtype=np.uint8, mode="r", offset=base,
                        shape=(delta + count * rec_size,))
        # Zero-copy structured view starting delta bytes into the mapped region.
        return np.ndarray((count,), dtype=dt, buffer=raw,
                          offset=delta, strides=(rec_size,))
    except (ValueError, OSError, BufferError):
        return None


def _read_ascii(f, vcount, vprops):
    (xn, yn, zn), rgb_names, nrm_names = _resolve_columns(vprops)
    names = [p[0] for p in vprops]
    col = {name: i for i, name in enumerate(names)}
    ncols = len(names)

    # Read exactly vcount vertex lines, then hand to np.loadtxt-style parsing.
    # We parse manually with fromstring for speed and to ignore trailing face
    # data that np.loadtxt would choke on.
    rows = []
    read = 0
    for line in f:
        if not line.strip():
            continue
        rows.append(line)
        read += 1
        if read >= vcount:
            break
    if not rows:
        raise PlyError("no vertex data found")

    # Join and parse in one shot. buf.split() on the whole block is far faster
    # than parsing line-by-line in Python and sidesteps np.fromstring (removed
    # in newer numpy). Tokens are converted by np.array once.
    buf = b" ".join(r.strip() for r in rows)
    flat = np.array(buf.split(), dtype=np.float64)
    if flat.size < read * ncols:
        raise PlyError("ASCII vertex data has fewer values than declared")
    flat = flat[: read * ncols].reshape(read, ncols)

    xyz = np.empty((read, 3), dtype=np.float32)
    xyz[:, 0] = flat[:, col[xn]]
    xyz[:, 1] = flat[:, col[yn]]
    xyz[:, 2] = flat[:, col[zn]]

    rgb = None
    if rgb_names:
        rn, gn, bn = rgb_names
        rtype = dict(vprops)[rn]
        base = _PLY_TYPES.get(rtype, "u1")
        scale = _color_scale(base)
        rgb = np.empty((read, 3), dtype=np.float32)
        rgb[:, 0] = flat[:, col[rn]]
        rgb[:, 1] = flat[:, col[gn]]
        rgb[:, 2] = flat[:, col[bn]]
        if scale != 1.0:
            rgb *= scale
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    nrm = None
    if nrm_names:
        nxn, nyn, nzn = nrm_names
        nrm = np.empty((read, 3), dtype=np.float32)
        nrm[:, 0] = flat[:, col[nxn]]
        nrm[:, 1] = flat[:, col[nyn]]
        nrm[:, 2] = flat[:, col[nzn]]

    return xyz, rgb, nrm


def quick_count(path: str) -> int:
    """Read only the header and return the vertex count (no body decode)."""
    with open(path, "rb") as f:
        _, vcount, _, _ = _read_header(f)
    return vcount
