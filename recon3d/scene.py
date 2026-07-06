from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from.geometry import camera_center, projection_matrix

@dataclass
class Camera:
    image_idx: int
    R: np.ndarray
    t: np.ndarray
    K: np.ndarray
    @property
    def center(self) -> np.ndarray:
        return camera_center(self.R, self.t)
    def P(self) -> np.ndarray:
        return projection_matrix(self.K, self.R, self.t)
    
@dataclass
class Point3D:
    id: int
    xyz: np.ndarray
    color:np.ndarray
    track_id: int
    obs: dict[int, int] = field(default_factory=dict)
    error: float = 0.0
    
@dataclass
class Reconstruction:
    cameras: dict[int, Camera] = field(default_factory=dict)
    points: dict[int, Point3D] = field(default_factory=dict)
    track_to_point: dict[int, int] = field(default_factory=dict)
    _next_pid: int = 0
    
    def add_camera(self, image_idx: int, R, t, K) -> Camera:
        cam = Camera(image_idx = image_idx, R = np.asarray(R, float), t = np.asarray(t, float).reshape(3), K = np.asarray(K, float))
        self.cameras[image_idx] = cam
        return cam

    def is_registered(self, image_idx: int) -> bool:
        return image_idx in self.cameras
    
    @property
    def registered_images(self) -> list[int]:
        return sorted(self.cameras.keys())
    
    def add_point(self, xyz, color, track_id, obs) -> Point3D:
        pid = self._next_pid
        self._next_pid += 1
        pt = Point3D(id = pid, xyz = np.asarray(xyz, float).reshape(3), color = np.asarray(color, int).reshape(3), track_id = track_id, obs = dict(obs))
        self.points[pid] = pt
        self.track_to_point[track_id] = pid
        return pt
    
    def remove_point(self, pid: int) -> None:
        p = self.points.pop(pid, None)
        if p is not None:
            self.track_to_point.pop(p.track_id, None)
            
    def point_for_track(self, track_id: int) -> Point3D | None:
        pid = self.track_to_point.get(track_id)
        return self.points.get(pid) if pid is not None else None
    
    def point_array(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.points:
            return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
        xyz = np.array([p.xyz for p in self.points.values()], float)
        rgb = np.array([p.color for p in self.points.values()], np.uint8)
        return xyz, rgb
    
    def stats(self) -> str:
        errs = [p.error for p in self.points.values() if np.isfinite(p.error)]
        me = float(np.mean(errs)) if errs else float('nan')
        return (f"camera={len(self.cameras)} points={len(self.points)} "
                f"mean_reproj={me:.3f}px")