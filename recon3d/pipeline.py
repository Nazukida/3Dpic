"""End-to-end reconstruction orchestrator.

Wires the stages together:

    images -> features -> matches -> tracks -> incremental SfM
           -> (optional) dense MVS -> cleanup -> PLY files + scene.json

Progress is emitted through a ``log`` callback so both the CLI (prints) and the
GUI (streamed to a panel) can consume it.  Stage banners use the ``[stage]``
prefixes the GUI's progress parser looks for.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from .config import Config
from .io_utils import list_images, write_ply
from .features import extract_features
from .matching import match_all
from .tracks import build_tracks
from .sfm import SfM
from .dense import run_dense
from .cleanup import statistical_outlier_removal, voxel_downsample, decimal_points
from .scene import Reconstruction


def _camera_ply(recon: Reconstruction):
    """A little frustum-ish marker cloud so cameras are visible in the viewer."""
    pts = []
    cols = []
    for cam in recon.cameras.values():
        c = cam.center
        pts.append(c)
        cols.append((255, 220, 60))
    if not pts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    return np.array(pts, np.float32), np.array(cols, np.uint8)


def _scene_summary(recon: Reconstruction) -> dict:
    xyz, _ = recon.point_array()
    return {
        "num_cameras": len(recon.cameras),
        "num_sparse_points": len(recon.points),
        "registered_images": recon.registered_images,
    }


def reconstruct(cfg: Config, log=print) -> dict:
    t_start = time.time()
    # Seed OpenCV's RANSAC and numpy for run-to-run stability. Combined with
    # per-pair seeding in matching and deterministic ordering in SfM, results
    # are near-identical between runs (a fraction of a percent of points may
    # differ due to FLANN's approximate nearest-neighbour search, which the
    # OpenCV RNG seed does not govern).
    import cv2
    cv2.setRNGSeed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.ensure_dirs()
    paths = list_images(cfg.image_dir)
    if len(paths) < 2:
        raise SystemExit(f"[pipeline] need >=2 images in {cfg.image_dir}, found {len(paths)}")
    log(f"[pipeline] {len(paths)} images from {cfg.image_dir}")

    # 1. features
    store = extract_features(paths, cfg, log=log)

    # 2. matches
    graph = match_all(store, cfg, log=log)
    if not graph.pairs:
        raise SystemExit("[pipeline] no image pairs matched; cannot reconstruct")

    # 3. tracks
    tracks = build_tracks(graph, min_len=2, log=log)

    # 4. sparse SfM
    log("[sfm] starting incremental reconstruction")
    recon = SfM(store, graph, tracks, cfg, log=log).run()
    if len(recon.cameras) < 2:
        raise SystemExit("[pipeline] SfM failed to register >=2 cameras")

    # write sparse outputs
    sparse_xyz, sparse_rgb = recon.point_array()
    sparse_path = os.path.join(cfg.work_dir, "sparse.ply")
    write_ply(sparse_path, sparse_xyz, sparse_rgb)
    log(f"[pipeline] wrote {sparse_path} ({len(sparse_xyz):,} points)")

    cam_xyz, cam_rgb = _camera_ply(recon)
    write_ply(os.path.join(cfg.work_dir, "cameras.ply"), cam_xyz, cam_rgb)

    result = {
        "sparse_ply": sparse_path,
        "num_cameras": len(recon.cameras),
        "num_sparse_points": len(recon.points),
    }

    # 5. dense (optional)
    if cfg.do_dense:
        log("[dense] starting dense reconstruction (MVS)")
        dpts, dcols = run_dense(store, recon, cfg, log=log)
        if len(dpts):
            # 6. cleanup — ORDER MATTERS for memory. The dense stage can fuse
            # tens of millions of points; the statistical-outlier step builds a
            # KD-tree and does a k-NN search whose cost scales with the point
            # count, so we CAP the cloud first (cheap striding/voxel decimation)
            # and only then run the expensive neighbour-based filtering on the
            # reduced set. Doing SOR first on the raw cloud is what exhausted
            # memory. The whole block is guarded so a cleanup failure still
            # saves the (valuable, slow-to-compute) dense points.
            log(f"[clean] cleaning dense cloud ({len(dpts):,} raw points)")
            try:
                if len(dpts) > cfg.dense_max_points:
                    dpts, dcols, _ = decimal_points(
                        dpts, dcols, None, cfg.dense_max_points, log=log)
                dpts, dcols, _ = statistical_outlier_removal(
                    dpts, dcols, None, k=cfg.sor_neighbors,
                    std_ratio=cfg.sor_std_ratio, log=log)
                if cfg.voxel_downsample > 0:
                    dpts, dcols, _ = voxel_downsample(
                        dpts, dcols, None, cfg.voxel_downsample, log=log)
            except MemoryError:
                log("[clean] out of memory during cleanup; "
                    "writing the un-cleaned dense cloud instead")
            except Exception as e:
                log(f"[clean] cleanup failed ({type(e).__name__}: {e}); "
                    f"writing the un-cleaned dense cloud instead")
            dense_path = os.path.join(cfg.work_dir, "dense.ply")
            write_ply(dense_path, dpts, dcols)
            log(f"[pipeline] wrote {dense_path} ({len(dpts):,} points)")
            result["dense_ply"] = dense_path
            result["num_dense_points"] = int(len(dpts))
    else:
        log("[dense] skipped (--no-dense)")

    # scene.json
    summary = _scene_summary(recon)
    summary["elapsed_sec"] = round(time.time() - t_start, 1)
    summary.update({k: v for k, v in result.items() if k.startswith("num_")})
    with open(os.path.join(cfg.work_dir, "scene.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"[pipeline] Done in {summary['elapsed_sec']}s — {recon.stats()}")
    result["scene"] = summary
    result["work_dir"] = cfg.work_dir
    # pick the best cloud to show
    result["primary_ply"] = result.get("dense_ply", sparse_path)
    return result
