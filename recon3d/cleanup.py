from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

def statistical_outlier_removal(points: np.ndarray, colors = None, normals = None, k: int = 16, std_ratio: float = 2.0, log = print, chunk: int = 2_000_000):
    _log = log or (lambda *a, **k: None)
    import time
    n = len(points)
    if n <= k:
        return points, colors, normals
    _log(f"[clean] SOR: computing {k}-NN distances for {n:,} points...")
    t0 = time.time()
    tree = cKDTree(points)
    # Query in chunks so peak memory stays bounded: a single tree.query over
    # all points allocates an (n, k+1) int64 index array AND an (n, k+1) float
    # distance array — for tens of millions of points that is many GiB and
    # blows up. We only need the per-point mean neighbour distance, so we can
    # accumulate it chunk by chunk and never hold more than (chunk, k+1) at once.
    mean_d = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dist, _ = tree.query(points[start:end], k=k + 1, workers=-1)
        # column 0 is the point itself (distance 0); average the k neighbours
        mean_d[start:end] = dist[:, 1:].mean(axis=1)
    mu = mean_d.mean()
    sigma = mean_d.std()
    thresh = mu + std_ratio * sigma
    keep = mean_d <= thresh
    elapsed = time.time() - t0
    _log(f"[clean] SOR: kept {int(keep.sum()):,}/{n:,} "
         f"(thresh {thresh:.4f}, mu {mu:.4f}, sigma {sigma:.4f}) [{elapsed:.2f}s]")
    return (points[keep],
            None if colors is None else colors[keep],
            None if normals is None else normals[keep])
    
def voxel_downsample(points: np.ndarray, colors = None, normals = None, voxel: float = 0.0, log = print):
    _log = log or (lambda *a, **k: None)
    if voxel <= 0.0 or len(points) == 0:
        return points, colors, normals
    keys = np.floor(points / voxel).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    ks = keys[order]
    pts = points[order]
    uniq_mask = np.ones(len(ks), bool)
    uniq_mask[1:] = np.any(ks[1:] != ks[:-1], axis=1)
    group_ids = np.cumsum(uniq_mask) - 1
    n_groups = group_ids[-1] + 1
    
    def avg(arr):
        out = np.zeros((n_groups, arr.shape[1]), np.float64)
        np.add.at(out, group_ids, arr[order].astype(np.float64))
        counts = np.bincount(group_ids, minlength = n_groups).reshape(-1, 1)
        return out / np.maximum(counts, 1)
    
    new_pts = np.zeros((n_groups, 3))
    np.add.at(new_pts, group_ids, pts.astype(np.float64))
    counts = np.bincount(group_ids, minlength = n_groups).reshape(-1, 1)
    new_pts /= np.maximum(counts, 1)
    new_cols = None
    if colors is not None:
        new_cols = np.clip(avg(colors), 0, 255).astype(np.uint8)
    new_nrm = None
    if normals is not None:
        nn = avg(normals)
        norm = np.linalg.norm(nn, axis=1, keepdims=True)
        new_nrm = nn / np.maximum(norm, 1e-12)
    _log(f"[clean] voxel {voxel}: {len(points)} -> {n_groups} points")
    return new_pts, new_cols, new_nrm

def decimal_points(points: np.ndarray, colors = None, normals = None, max_points: int = 1_500_000, log = None) -> tuple:
    _log = log or (lambda *a, **k: None)
    n = len(points)
    if n <= max_points or n <= max_points:
        return points, colors, normals
    target = int(max_points)
    pts = points
    cols = colors
    nrms = normals
    pre_cap = target * 2
    if n > pre_cap:
        step = int (np.ceil(n / pre_cap))
        idx = np.arange(0, n, step)
        pts = pts[idx]
        if cols is not None:
            cols = cols[idx]
        if nrms is not None:
            nrms = nrms[idx]
        _log(f"[decimate] stride {n:,} -> {len(pts):,} (step {step})")
        
    lo = np.percentile(pts, 1, axis = 0)
    hi = np.percentile(pts, 99, axis = 0)
    diag = float((hi - lo).max())
    if diag <= 0:
        diag = float((pts.max(axis=0) - pts.min(axis=0)).max()) or 1.0
    base_pts, base_cols, base_nrms = pts, cols, nrms
    voxel = diag / (target ** 0.5)
    if voxel > 0:
        pts, cols, nrms = voxel_downsample(base_pts, base_cols, base_nrms, voxel, log=log)
        if 0 < len(pts) < 0.75 * target:
            voxel *=(len(pts) / target) ** 0.5
            pts, cols, nrms = voxel_downsample(base_pts, base_cols, base_nrms, voxel, log=None)
    
    m = len(pts)
    if m > target:
        step = int(np.ceil(m / target))
        idx = np.arange(0, m, step)
        pts = pts[idx]
        if cols is not None:
            cols = cols[idx]
        if nrms is not None:
            nrms = nrms[idx]
    
    _log(f"[decimal] {n:,} -> {len(pts):,} (cap {target:,})")
    return pts, cols, nrms