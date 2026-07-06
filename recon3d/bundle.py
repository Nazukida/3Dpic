from __future__ import annotations
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .scene import Reconstruction

def _rotate(rvecs: np.ndarray, pts: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rvecs, axis = 1, keepdims = True)
    small = theta[:, 0] < 1e-12
    k = np.zeros_like(rvecs)
    nz = ~small
    k[nz] = rvecs[nz] / theta[nz]
    cos = np.cos(theta)
    sin = np.sin(theta)
    kdotv = np.sum(k * pts, axis = 1, keepdims = True)
    kcrossv = np.cross(k, pts)
    rotated = cos * pts + sin * kcrossv + (1 - cos) * kdotv * k
    rotated[small] = pts[small]
    return rotated

class BundleProblem:
    def __init__(self, recon: Reconstruction, refine_focal: bool):
        self.refine_focal = refine_focal
        self.cam_ids = recon.registered_images
        self.cam_index = {c: i for i, c in enumerate(self.cam_ids)}
        self.pt_ids = list(recon.points.keys())
        self.pt_index = {p: i for i, p in enumerate(self.pt_ids)}
        self.n_cams = len(self.cam_ids)
        self.n_pts = len(self.pt_ids)
        
        cam_obs, pt_obs, uv = [], [], []
        self.cx = np.zeros(self.n_cams)
        self.cy = np.zeros(self.n_cams)
        self.fixed_focal = np.zeros(self.n_cams)
        for ci, cid in enumerate(self.cam_ids):
            cam = recon.cameras[cid]
            self.cx[ci] = cam.K[0, 2]
            self.cy[ci] = cam.K[1, 2]
            self.fixed_focal[ci] = cam.K[0, 0]
        self.long_side = 2.0 * np.maximum(self.cx, self.cy)
        self._recon = recon
        self.cam_obs = None
        self.pt_obs = None
        self.uv = None
        
    def build_observations(self, coords) -> None:
        cam_obs, pt_obs, uv = [], [], []
        for pid in self.pt_ids:
            p = self._recon.points[pid]
            pli = self.pt_index[pid]
            for img_idx, feat_idx in p.obs.items():
                if img_idx not in self.cam_index:
                    continue
                u, v = coords(img_idx, feat_idx)
                cam_obs.append(self.cam_index[img_idx])
                pt_obs.append(pli)
                uv.append((u, v))
        self.cam_obs = np.asarray(cam_obs, np.int64)
        self.pt_obs = np.asarray(pt_obs, np.int64)
        self.uv = np.asarray(uv, np.float64)
        
    def pack(self) -> np.ndarray:
        recon = self._recon
        cam_params = np.zeros((self.n_cams, 6))
        for ci, cid in enumerate(self.cam_ids):
            cam = recon.cameras[cid]
            rvec = cv2.Rodrigues(cam.R)[0].ravel()
            cam_params[ci, :3] = rvec
            cam_params[ci, 3:6] = cam.t
        pts = np.array([recon.points[p].xyz for p in self.pt_ids])
        x = np.concatenate([cam_params.ravel(), pts.ravel()])
        if self.refine_focal:
            nf0 = float(np.median(self.fixed_focal / self.long_side))
            x = np.concatenate([x, [nf0]])
        return x
    
    def unpack_to_recon(self, x: np.ndarray) -> None:
        recon = self._recon
        cam_params = x[:self.n_cams * 6].reshape(self.n_cams, 6)
        off = self.n_cams * 6
        pts = x[off:off + self.n_pts * 3].reshape(self.n_pts, 3)
        nf = None
        if self.refine_focal:
            nf = x[-1]
        for ci, cid in enumerate(self.cam_ids):
            cam = recon.cameras[cid]
            cam.R = cv2.Rodrigues(cam_params[ci, :3])[0]
            cam.t = cam_params[ci, 3:6].copy()
            if nf is not None:
                f_px = nf * self.long_side[ci]
                cam.K[0, 0] = f_px
                cam.K[1, 1] = f_px
        for pi, pid in enumerate(self.pt_ids):
            recon.points[pid].xyz = pts[pi].copy()
            
    def residuals(self, x: np.ndarray) -> np.ndarray:
        cam_params = x[:self.n_cams * 6].reshape(self.n_cams, 6)
        off = self.n_cams * 6
        pts = x[off:off + self.n_pts * 3].reshape(self.n_pts, 3)
        if self.refine_focal:
            nf = x[-1]
        rvecs = cam_params[self.cam_obs, :3]
        tvecs = cam_params[self.cam_obs, 3:6]
        X = pts[self.pt_obs]
        Xc = _rotate(rvecs, X) + tvecs
        z = Xc[:, 2]
        z_safe = np.where(np.abs(z) < 1e-8, 1e-8, z)
        x_n = Xc[:, 0] / z_safe
        y_n = Xc[:, 1] / z_safe
        
        if self.refine_focal:
            f = nf * self.long_side[self.cam_obs]
        else:
            f = self.fixed_focal[self.cam_obs]
        u = f * x_n + self.cx[self.cam_obs]
        v = f * y_n + self.cy[self.cam_obs]
        res = np.empty(len(self.cam_obs) * 2)
        res[0::2] = u - self.uv[:, 0]
        res[1::2] = v - self.uv[:, 1]
        behind = z <= 1e-8
        if np.any(behind):
            res[np.repeat(behind, 2)] = 0.0
        return res
    
    def jac_sparsity(self) ->lil_matrix:
        m = len(self.cam_obs) * 2
        n = self.n_cams * 6 + self.n_pts * 3 + (1 if self.refine_focal else 0)
        A = lil_matrix((m, n), dtype=int)
        i = np.arange(len(self.cam_obs))
        for s in range(6):
            A[2 * i, self.cam_obs * 6 + s] = 1
            A[2 * i + 1, self.cam_obs * 6 + s] = 1
        pt_base = self.n_cams * 6
        for s in range(3):
            A[2 * i, pt_base + self.pt_obs * 3 + s] = 1
            A[2 * i + 1, pt_base + self.pt_obs * 3 + s] = 1
        if self.refine_focal:
            A[:, -1] = 1
        return A
    
def run_bundle_adjustment(recon: Reconstruction, coords, refine_focal: bool = True,max_iters: int = 30, ftol: float = 1e-4, verbose: bool = False, log = print, label: str = "local") -> float:
    if len(recon.cameras) < 2 or len(recon.points) < 1:
        return float('nan')
    prob = BundleProblem(recon, refine_focal = refine_focal)
    prob.build_observations(coords)
    if prob.uv is None or len(prob.uv) < 1:
        return float('nan')
    x0 = prob.pack()
    A = prob.jac_sparsity()
    r0 = prob.residuals(x0)
    rms0 = float(np.sqrt(np.mean(r0 ** 2)))
    
    import time as _time
    t0 = _time.time()
    log(f"[BA] {label}: solving {prob.n_cams} cams, {prob.n_pts} pts, "
        f"{len(prob.uv)} obs (start RMS {rms0:.3f}px, max {max_iters} iters)...")
    res = least_squares(
        prob.residuals, x0, jac_sparsity=A,
        verbose=2 if verbose else 0, x_scale='jac', ftol=ftol,
        method='trf', loss='huber', f_scale=2.0, max_nfev=max_iters,
    )
    prob.unpack_to_recon(res.x)
    rms1 = float(np.sqrt(np.mean(res.fun ** 2)))
    log(f"[BA] {label}: RMS {rms0:.3f} -> {rms1:.3f}px "
        f"({len(res.fun)//2} residual pairs, {_time.time()-t0:.2f}s)")
    return rms1