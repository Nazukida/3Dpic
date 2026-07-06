from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Callable
import cv2
import numpy as np
from .config import Config
from .scene import Reconstruction, Camera, Point3D
from .io_utils import ImageInfo

@dataclass
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: list[float]
    
@dataclass
class ColmapImage:
    id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str
    points2d: np.ndarray
    
def parse_cameras(path: str) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[cam_id] = ColmapCamera(id = cam_id, model = model, width = width, height = height, params = params)
    return cameras

def parse_images(path: str) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    i = 0
    while i < len(lines) - 1:
        parts = lines[i].split()
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = parts[9]
        obs_parts = lines[i + 1].split()
        n_obs = len(obs_parts) // 3
        if n_obs > 0:
            pts2d = np.array([[float(obs_parts[j * 3]), float(obs_parts[j * 3 + 1])], [float(obs_parts[j * 3 + 2])]] for j in range(n_obs)], dtype=np.float64)
        else:
            pts2d = np.empty((0, 3), dtype=np.float64)
        images[img_id] = ColmapImage(id = img_id, qw = qw, qx = qx, qy = qy, qz = qz, tx = tx, ty = ty, tz = tz, camera_id = camera_id, name = name, points2d = pts2d)
        i += 2
    return images

def parse_points3d(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[tuple[int, int]]]]:
    xyz_list: list[list[float]] = []