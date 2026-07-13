from __future__ import annotations

import glob
import os
import struct
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image, ExifTags

IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', ".webp"]

@dataclass
class ImageInfo:
    idx: int
    path: str
    name: str
    width: int
    height: int
    orig_width: int
    orig_height: int
    scale: float
    exif_focal_px: float | None
    
def list_images(image_dir: str) -> list[str]:
    files: list[str] = []
    for ext in IMAGE_EXTS:
        files += glob.glob(os.path.join(image_dir, f"*{ext}"))
        files += glob.glob(os.path.join(image_dir, f"*{ext.upper()}"))
    seen, uniq = set(), []
    for f in sorted(files):
        key = os.path.normcase(os.path.abspath(f))
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq

def _exif_focal_px(pil_img: Image.Image, work_w: int, work_h: int, orig_w: int, orig_h: int) -> float | None:
    try:
        exif = pil_img.getexif()
    except Exception:
        return None
    if not exif:
        return None
    
    tag = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    try:
        ifd = exif.get_ifd(0x8769)
        for k, v in ifd.items():
            tag[ExifTags.TAGS.get(k, k)] = v
    except Exception:
        pass
    
    def as_float(x):
        try:
            if isinstance(x, tuple) and len(x) == 2:
                return float(x[0]) / float(x[1]) if x[1] else None
            return float(x)
        except Exception:
            return None
        
    longest_orig = max(orig_w, orig_h)
    work_scale = max(work_w, work_h) / longest_orig if longest_orig else 1.0
    f_mm = as_float(tag.get("FocalLength"))
    fp_xres = as_float(tag.get("FocalPlaneXResolution"))
    fp_unit = tag.get("FocalPlaneResolutionUnit", 2)
    if f_mm and fp_xres:
        unit_mm = 25.4 if fp_unit != 3 else 10.0
        px_per_mm = fp_xres / unit_mm
        focal_px_orig = f_mm * px_per_mm
        if focal_px_orig > 0:
            return focal_px_orig * work_scale
        
    f35 = as_float(tag.get("FocalLengthIn35mmFilm"))
    if f35 and f35 > 0:
        return (f35 / 36.0) * max(work_w, work_h)
    return None

def load_and_prepare(path: str, idx: int, max_size: int) -> tuple[ImageInfo, np.ndarray]:
    pil = Image.open(path)
    pil = _apply_exif_orientation(pil)
    orig_w, orig_h = pil.size
    longest = max(orig_w, orig_h)
    scale = 1.0
    if max_size > 0 and longest > max_size:
        scale = max_size / longest
    work_w = max(1, int(round(orig_w * scale)))
    work_h = max(1, int(round(orig_h * scale)))
    
    focal_px = _exif_focal_px(pil, work_w, work_h, orig_w, orig_h)
    rgb = pil.convert("RGB")
    if (work_w, work_h) != (orig_w, orig_h):
        rgb = rgb.resize((work_w, work_h), Image.Resampling.LANCZOS)
    arr = np.asarray(rgb)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    
    info = ImageInfo(
        idx=idx, path=path, name=os.path.basename(path),
        width=work_w, height=work_h, orig_width=orig_w, orig_height=orig_h,
        scale=scale, exif_focal_px=focal_px
    )
    return info, bgr

def _apply_exif_orientation(pil: Image.Image) -> Image.Image:
    try:
        exif = pil.getexif()
        orient = exif.get(0x0112, 1)
    except Exception:
        return pil
    ops = {
        2: [Image.FLIP_LEFT_RIGHT],
        3: [Image.ROTATE_180],
        4: [Image.FLIP_TOP_BOTTOM],
        5: [Image.FLIP_LEFT_RIGHT, Image.ROTATE_270],
        6: [Image.ROTATE_270],
        7: [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],
        8: [Image.ROTATE_90]
    }
    for op in ops.get(orient, []):
        pil = pil.transpose(op)
    return pil
def guess_intrinsics(info: ImageInfo, focal_factor: float) -> np.ndarray:
    if info.exif_focal_px and info.exif_focal_px > 0:
        f = float(info.exif_focal_px)
    else:
        f = focal_factor * max(info.width, info.height)
    cx = info.width / 2.0
    cy = info.height / 2.0
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

def write_ply(path: str, points: np.ndarray, colors: np.ndarray | None = None, normals: np.ndarray | None = None, binary: bool = True) -> None:
    points = np.ascontiguousarray(points, dtype = np.float32)
    n = len(points)
    has_color = colors is not None and len(colors) == n
    has_normal = normals is not None and len(normals) == n
    if has_color:
        colors = np.ascontiguousarray(np.clip(colors, 0, 255).astype(np.uint8))
    if has_normal:
        normals = np.ascontiguousarray(normals, dtype=np.float32)
        
    header = ["ply"]
    header.append("format binary_little_endian 1.0" if binary else "format ascii 1.0")
    header.append(f"element vertex {n}")
    header += ["property float x", "property float y", "property float z"]
    if has_normal:
        header += ["property float nx", "property float ny", "property float nz"]
    if has_color:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok = True)
    if binary:
        with open(path, "wb") as f:
            f.write(("\n".join(header) + "\n").encode("ascii"))
            fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
            if has_normal:
                fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
            if has_color:
                fields += [("r", "u1"), ("g", "u1"), ("b", "u1")]
            rec = np.empty(n, dtype = fields)
            rec["x"], rec["y"], rec["z"] = points[:, 0], points[:, 1], points[:, 2]
            if has_normal:
                rec["nx"], rec["ny"], rec["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
            if has_color:
                rec["r"], rec["g"], rec["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
            f.write(rec.tobytes())
    else:
        with open(path, "w") as f:
            f.write("\n".join(header) + "\n")
            for i in range(n):
                row = [f"{points[i, 0]:.6f}", f"{points[i, 1]:.6f}", f"{points[i, 2]:.6f}"]
                if has_normal:
                    row += [f"{normals[i, 0]:.6f}", f"{normals[i, 1]:.6f}", f"{normals[i, 2]:.6f}"]
                if has_color:
                    row += [str(colors[i, 0]), str(colors[i, 1]), str(colors[i, 2])]
                f.write(" ".join(row) + "\n")