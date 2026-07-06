"""Command-line entry point: ``recon3d --images DIR --out DIR``.

Also importable as ``python -m recon3d.run``.  Quality presets map onto the
fine-grained :class:`Config` knobs so users don't have to think about SGBM
parameters unless they want to.
"""

from __future__ import annotations

import argparse
import sys

from .config import Config
from .pipeline import reconstruct


# quality presets: (max_image_size, sift_n_features, dense_max_size, sgbm_num_disp, do_dense)
_PRESETS = {
    "fast":     dict(max_image_size=1000, sift_n_features=4000,
                     dense_max_size=800,  sgbm_num_disp=128, dense_num_neighbors=2),
    "balanced": dict(max_image_size=1600, sift_n_features=8000,
                     dense_max_size=1200, sgbm_num_disp=192, dense_num_neighbors=4),
    "high":     dict(max_image_size=2400, sift_n_features=12000,
                     dense_max_size=1800, sgbm_num_disp=256, dense_num_neighbors=6),
    "ultra":    dict(max_image_size=3200, sift_n_features=20000,
                     dense_max_size=2600, sgbm_num_disp=320, dense_num_neighbors=8),
}


def build_config(args) -> Config:
    cfg = Config()
    preset = _PRESETS.get(args.quality, _PRESETS["balanced"])
    for k, v in preset.items():
        setattr(cfg, k, v)
    cfg.image_dir = args.images
    cfg.work_dir = args.out
    cfg.do_dense = not args.no_dense
    if args.workers and args.workers > 0:
        cfg.num_workers = args.workers
    if args.voxel and args.voxel > 0:
        cfg.voxel_downsample = args.voxel
    if args.max_size and args.max_size > 0:
        cfg.max_image_size = args.max_size
    return cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="recon3d",
        description="CPU-only 3D reconstruction from images to a point cloud.")
    p.add_argument("--images", required=True, help="directory of input images")
    p.add_argument("--out", default="output", help="output/work directory")
    p.add_argument("--quality", default="balanced", choices=list(_PRESETS),
                   help="quality preset")
    p.add_argument("--no-dense", action="store_true",
                   help="skip dense MVS (sparse point cloud only)")
    p.add_argument("--workers", type=int, default=0, help="worker processes (0=auto)")
    p.add_argument("--voxel", type=float, default=0.0,
                   help="voxel downsample size for the dense cloud (0=off)")
    p.add_argument("--max-size", type=int, default=0,
                   help="override max image size (px, long edge)")
    args = p.parse_args(argv)

    cfg = build_config(args)
    try:
        result = reconstruct(cfg, log=lambda m: print(m, flush=True))
    except SystemExit as e:
        # raised by the pipeline for expected, actionable conditions
        print(str(e), flush=True)
        return 1
    except KeyboardInterrupt:
        print("[pipeline] interrupted", flush=True)
        return 130
    except Exception as e:
        # Last-resort guard: an unexpected error in any stage must produce a
        # readable message and a clean non-zero exit, never a raw crash that
        # looks like the app "quit by itself". The full traceback still goes to
        # stderr for debugging.
        import traceback
        traceback.print_exc()
        print(f"[pipeline] 意外错误，重建已中止：{type(e).__name__}: {e}", flush=True)
        return 2
    primary = result.get("primary_ply")
    print(f"[pipeline] primary cloud: {primary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
