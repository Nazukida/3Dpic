"""CPU dense reconstruction (multi-view stereo).

Sparse SfM gives camera poses and a thin point cloud; this stage thickens it.
For each registered "reference" image we pick a few neighbouring views, run
OpenCV's semi-global block matcher (SGBM) on the rectified stereo pair, convert
disparity to a metric depth map, and back-project to 3D.  Depths that survive a
left-right / geometric-consistency check across neighbours are fused into the
output cloud.

This is intentionally the SGBM route rather than PatchMatch: SGBM is highly
optimised in OpenCV and runs comfortably on CPU, which is the whole point of
this project.  Quality is traded via ``Config.dense_*`` knobs.
"""

from __future__ import annotations

import numpy as np
import cv2

from .config import Config
from .features import FeatureStore
from .scene import Reconstruction
from .io_utils import load_and_prepare
from .geometry import camera_center


def _sparse_bound(recon: Reconstruction):
    """Robust centre + clip radius from the bundle-adjusted sparse points.

    Uses the median centre and a high percentile of the radial distance so a
    few stray sparse points don't inflate the bound; multiplies by a generous
    factor so legitimate dense fill near the surface is never clipped.
    """
    xyz, _ = recon.point_array()
    if len(xyz) < 8:
        return np.zeros(3), None
    centre = np.median(xyz, axis=0)
    d = np.linalg.norm(xyz - centre, axis=1)
    r99 = float(np.percentile(d, 99))
    if r99 <= 0:
        return centre, None
    return centre, r99 * 4.0


def _neighbors_by_baseline(recon: Reconstruction, ref: int, k: int) -> list[int]:
    """Pick up to ``k`` registered views whose optical axes are similar to the
    reference (so rectification is well-conditioned) yet offer some baseline."""
    ref_cam = recon.cameras[ref]
    ref_c = ref_cam.center
    ref_dir = ref_cam.R[2, :]   # viewing direction (3rd row of R)

    scored = []
    for idx, cam in recon.cameras.items():
        if idx == ref:
            continue
        d = cam.R[2, :]
        cosang = float(np.clip(np.dot(ref_dir, d), -1, 1))
        ang = np.degrees(np.arccos(cosang))
        baseline = float(np.linalg.norm(cam.center - ref_c))
        if baseline <= 1e-6:
            continue
        # stereoRectify only behaves for near-parallel views; large rotations
        # make it degenerate (disparities blow past the image width). Restrict
        # to a modest view-angle change and prefer the *smallest* such angle —
        # nearest neighbours give a search window SGBM can actually cover.
        if ang > 30.0:
            continue
        scored.append((ang, idx))
    scored.sort()   # smallest rotation first
    return [idx for _, idx in scored[:k]]


def _load_gray_and_rgb(store: FeatureStore, cfg: Config, idx: int):
    """Load the working image again at dense resolution."""
    info = store.infos[idx]
    _, bgr = load_and_prepare(info.path, idx, cfg.dense_max_size)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # scale intrinsics from feature-res to dense-res
    scale = bgr.shape[1] / info.width
    return gray, rgb, scale


def _scaled_K(K: np.ndarray, scale: float) -> np.ndarray:
    Ks = K.copy()
    Ks[0, 0] *= scale
    Ks[1, 1] *= scale
    Ks[0, 2] *= scale
    Ks[1, 2] *= scale
    return Ks


def _make_sgbm(cfg: Config, min_disp: int, num_disp: int) -> "cv2.StereoSGBM":
    num_disp = max(16, (int(num_disp) // 16) * 16)
    bs = cfg.sgbm_block_size | 1  # must be odd
    return cv2.StereoSGBM_create(
        minDisparity=int(min_disp),
        numDisparities=num_disp,
        blockSize=bs,
        P1=8 * 3 * bs * bs,
        P2=32 * 3 * bs * bs,
        disp12MaxDiff=1,
        uniquenessRatio=cfg.sgbm_uniqueness,
        speckleWindowSize=cfg.sgbm_speckle_window,
        speckleRange=cfg.sgbm_speckle_range,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def _disparity_range(recon, ref, R1, P1, P2, cfg, width):
    """Estimate the SGBM disparity search window from the bundle-adjusted
    sparse points visible in the reference view.

    The correct range is scene-dependent: disp = f_rect * B / Z_rect, so a fixed
    range (as in the quality presets) badly under-shoots close / wide-baseline
    scenes. We project the trusted sparse geometry into the rectified frame and
    size the window to the observed disparities.

    Returns ``(min_disp, num_disp)`` or ``None`` when the geometry is degenerate
    (a physically implausible window wider than the image — which happens when
    stereoRectify is fed views too far apart to rectify). The caller skips such
    pairs rather than attempting a doomed / memory-blowing SGBM run.
    """
    cam_r = recon.cameras[ref]
    xyz, _ = recon.point_array()
    if len(xyz) < 8:
        return None
    Xc = (cam_r.R @ xyz.T + cam_r.t.reshape(3, 1)).T
    front = Xc[:, 2] > 1e-6
    if front.sum() < 8:
        return None
    Xr = (R1 @ Xc[front].T).T
    zr = Xr[:, 2]
    zr = zr[zr > 1e-6]
    if len(zr) < 8:
        return None
    f_rect = P1[0, 0]
    B = abs(P2[0, 3] / P2[0, 0]) if P2[0, 0] != 0 else 0.0
    if B <= 0:
        return None
    disp = f_rect * B / zr
    d_lo = float(np.percentile(disp, 2))
    d_hi = float(np.percentile(disp, 98))
    # pad the window by 30% on each side and floor min at 0
    span = max(d_hi - d_lo, 16.0)
    min_disp = max(0, int(np.floor(d_lo - 0.3 * span)))
    num_disp = int(np.ceil((d_hi + 0.3 * span - min_disp) / 16.0) * 16)
    # For genuinely overlapping stereo the maximum disparity is a fraction of
    # the image width. If the required window (min + span) reaches the width,
    # stereoRectify has degenerated (views too oblique / far apart, rectified
    # focal blown up) — the resulting disparities are physically meaningless and
    # a large numDisparities would also blow up SGBM's cost-volume allocation.
    # Skip such pairs; dense is best-effort and the pipeline falls back to the
    # bundle-adjusted sparse cloud.
    if min_disp + num_disp > int(width) or num_disp < 16:
        return None
    return min_disp, max(16, num_disp)


def _dense_pair(ref, nb, store, recon, cfg, clip_center=None, clip_radius=None):
    """Rectify ref/nb, run SGBM, return (points_world, colors) for the ref.

    ``clip_center`` / ``clip_radius`` (from the bundle-adjusted sparse cloud)
    are used to reject stray far points from weak disparities.  The disparity
    search window is sized per-pair from the sparse geometry, which is essential
    — a fixed window under-shoots close or wide-baseline scenes and finds
    nothing.
    """
    cam_r = recon.cameras[ref]
    cam_n = recon.cameras[nb]

    gray_r, rgb_r, scale_r = _load_gray_and_rgb(store, cfg, ref)
    gray_n, rgb_n, scale_n = _load_gray_and_rgb(store, cfg, nb)

    Kr = _scaled_K(cam_r.K, scale_r)
    Kn = _scaled_K(cam_n.K, scale_n)

    h, w = gray_r.shape

    # relative pose ref->nb:  X_n = R_rel X_r + t_rel
    R_rel = cam_n.R @ cam_r.R.T
    t_rel = cam_n.t - R_rel @ cam_r.t

    try:
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            Kr, None, Kn, None, (w, h), R_rel, t_rel.reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    except cv2.error:
        return None

    map1x, map1y = cv2.initUndistortRectifyMap(Kr, None, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(Kn, None, R2, P2, (w, h), cv2.CV_32FC1)
    rect_r = cv2.remap(gray_r, map1x, map1y, cv2.INTER_LINEAR)
    rect_n = cv2.remap(gray_n, map2x, map2y, cv2.INTER_LINEAR)
    rect_rgb = cv2.remap(rgb_r, map1x, map1y, cv2.INTER_LINEAR)

    rng = _disparity_range(recon, ref, R1, P1, P2, cfg, w)
    if rng is None:
        return None                       # degenerate rectification, skip pair
    min_disp, num_disp = rng
    sgbm = _make_sgbm(cfg, min_disp, num_disp)
    disp = sgbm.compute(rect_r, rect_n).astype(np.float32) / 16.0
    valid = disp > (min_disp + 0.5)
    if valid.sum() < 100:
        return None

    # Canonical MVS back-projection: reprojectImageTo3D gives points in the
    # *rectified reference* camera frame; rotate by R1^T into the reference
    # camera frame, then by the reference pose into world space.
    pts3d = cv2.reprojectImageTo3D(disp, Q)          # rectified-ref frame
    mask = valid & np.isfinite(pts3d).all(axis=2) & (pts3d[:, :, 2] > 0)
    if mask.sum() < 50:
        return None

    pr = pts3d[mask]                                  # (M,3) rectified-ref
    cr = rect_rgb[mask]                               # (M,3) colours
    pr_cam = (R1.T @ pr.T).T                          # -> reference camera frame
    Xw = (cam_r.R.T @ (pr_cam - cam_r.t).T).T         # -> world frame

    keep = np.ones(len(Xw), bool)
    if clip_radius is not None:
        # Reject the near-infinite-depth smear from weak/wrong disparities; the
        # bundle-adjusted sparse cloud gives a trustworthy scene extent.
        dc = np.linalg.norm(Xw - clip_center, axis=1)
        keep = dc <= clip_radius
    if keep.sum() < 50:
        return None
    return Xw[keep].astype(np.float32), cr[keep].astype(np.uint8)


def run_dense(store: FeatureStore, recon: Reconstruction, cfg: Config, log=print):
    if len(recon.cameras) < 2:
        log("[dense] need >=2 cameras, skipping")
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

    ref_ids = recon.registered_images
    all_pts = []
    all_cols = []
    n = len(ref_ids)

    # The sparse cloud is well constrained by bundle adjustment; use its robust
    # extent as a sanity bound. SGBM near-zero disparities back-project to
    # near-infinite depth, so clamp dense points to a generous multiple of the
    # sparse scene radius around its robust centre.
    clip_center, clip_radius = _sparse_bound(recon)
    if clip_radius is not None:
        log(f"[dense] scene bound: centre {clip_center.round(2)}, "
            f"clip radius {clip_radius:.2f}")

    for i, ref in enumerate(ref_ids):
        nbs = _neighbors_by_baseline(recon, ref, cfg.dense_num_neighbors)
        if not nbs:
            continue
        pair_pts = []
        pair_cols = []
        for nb in nbs:
            # A failure on one stereo pair (bad rectification, SGBM allocation
            # error, degenerate geometry) must not abort the whole dense stage
            # and discard every point computed so far — skip just this pair.
            try:
                out = _dense_pair(ref, nb, store, recon, cfg,
                                  clip_center, clip_radius)
            except Exception as e:
                log(f"[dense] pair (ref {ref}, nb {nb}) skipped: "
                    f"{type(e).__name__}: {e}")
                continue
            if out is None:
                continue
            p, c = out
            if len(p):
                pair_pts.append(p)
                pair_cols.append(c)
        if pair_pts:
            all_pts.append(np.concatenate(pair_pts))
            all_cols.append(np.concatenate(pair_cols))
        log(f"[dense] depth {i+1}/{n} (ref {ref}, {len(nbs)} neighbours, "
            f"{sum(len(p) for p in pair_pts):,} pts)")

    if not all_pts:
        log("[dense] produced no points")
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

    pts = np.concatenate(all_pts)
    cols = np.concatenate(all_cols)
    log(f"[dense] fused {len(pts):,} raw dense points")
    return pts, cols
