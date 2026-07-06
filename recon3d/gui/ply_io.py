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


def read_ply(path: str, max_points: int | None = None) -> PlyData:
    """Load a PLY point cloud.

    ``max_points`` optionally caps the number of points via uniform striding
    (cheap, order-preserving) - useful as a guard for enormous files.
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
            xyz, rgb, nrm = _read_binary(f, vcount, vprops, big_endian)

    n = len(xyz)
    if max_points is not None and 0 < max_points < n:
        step = int(np.ceil(n / max_points))
        idx = np.arange(0, n, step)
        xyz = xyz[idx]
        rgb = rgb[idx] if rgb is not None else None
        nrm = nrm[idx] if nrm is not None else None

    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    if rgb is not None:
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if nrm is not None:
        nrm = np.ascontiguousarray(nrm, dtype=np.float32)
    return PlyData(xyz=xyz, rgb=rgb, normals=nrm, count=len(xyz))


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


def _read_binary(f, vcount, vprops, big_endian):
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

    data = np.fromfile(f, dtype=dt, count=vcount)
    if len(data) < vcount:
        # truncated file - use what we got rather than crashing
        vcount = len(data)

    (xn, yn, zn), rgb_names, nrm_names = _resolve_columns(vprops)
    xyz = np.empty((vcount, 3), dtype=np.float32)
    xyz[:, 0] = data[xn]
    xyz[:, 1] = data[yn]
    xyz[:, 2] = data[zn]

    rgb = None
    if rgb_names:
        rn, gn, bn = rgb_names
        scale = _color_scale(ptype_by_name[rn])
        rgb = np.empty((vcount, 3), dtype=np.float32)
        rgb[:, 0] = data[rn]
        rgb[:, 1] = data[gn]
        rgb[:, 2] = data[bn]
        if scale != 1.0:
            rgb *= scale
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    nrm = None
    if nrm_names:
        nxn, nyn, nzn = nrm_names
        nrm = np.empty((vcount, 3), dtype=np.float32)
        nrm[:, 0] = data[nxn]
        nrm[:, 1] = data[nyn]
        nrm[:, 2] = data[nzn]

    return xyz, rgb, nrm


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
