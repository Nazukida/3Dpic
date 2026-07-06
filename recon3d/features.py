"""SIFT feature extraction with on-disk caching.

Features are the front of the pipeline and are embarrassingly parallel across
images, so extraction is farmed out to a process pool.  Results are cached as
``.npz`` next to the work dir keyed by image path + mtime + parameters, so
re-running a reconstruction (e.g. after tweaking SfM settings) skips this stage
entirely.

Each image yields:
* ``keypoints`` - (M, 2) float32 pixel coordinates (x, y)
* ``sizes``     - (M,) float32 keypoint scale (kept for optional weighting)
* ``descriptors`` - (M, 128) float32 SIFT descriptors
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

from .config import Config
from .io_utils import ImageInfo, load_and_prepare, guess_intrinsics


class FeatureSet:
    __slots__ = ("keypoints", "sizes", "descriptors")

    def __init__(self, keypoints, sizes, descriptors):
        self.keypoints = keypoints
        self.sizes = sizes
        self.descriptors = descriptors

    def __len__(self):
        return len(self.keypoints)


def _cache_key(path: str, cfg: Config) -> str:
    try:
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path)
    except OSError:
        mtime = size = 0
    raw = f"{os.path.abspath(path)}|{mtime}|{size}|{cfg.max_image_size}|" \
          f"{cfg.sift_n_features}|{cfg.sift_contrast_threshold}|{cfg.sift_edge_threshold}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _extract_one(args):
    """Worker: load image, run SIFT, return arrays + ImageInfo fields.

    Runs in a subprocess so it must be importable and self-contained. It must
    NOT raise: an unreadable/corrupt image or a SIFT failure in one worker
    would otherwise propagate through ``fut.result()`` and abort the whole
    extraction stage. On failure it returns an ``("error", ...)`` sentinel that
    the parent logs and skips.
    """
    idx, path, cfg_dict = args
    try:
        cfg = Config(**cfg_dict)
        info, bgr = load_and_prepare(path, idx, cfg.max_image_size)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create(
            nfeatures=cfg.sift_n_features,
            contrastThreshold=cfg.sift_contrast_threshold,
            edgeThreshold=cfg.sift_edge_threshold,
        )
        kps, desc = sift.detectAndCompute(gray, None)
        if kps is None or len(kps) == 0:
            pts = np.zeros((0, 2), np.float32)
            sizes = np.zeros((0,), np.float32)
            desc = np.zeros((0, 128), np.float32)
        else:
            pts = np.array([k.pt for k in kps], np.float32)
            sizes = np.array([k.size for k in kps], np.float32)
            desc = np.asarray(desc, np.float32)

        # sample the colour at each keypoint (for point-cloud tinting later)
        if len(pts):
            xi = np.clip(pts[:, 0].astype(np.int32), 0, bgr.shape[1] - 1)
            yi = np.clip(pts[:, 1].astype(np.int32), 0, bgr.shape[0] - 1)
            bgr_samples = bgr[yi, xi]
            rgb = bgr_samples[:, ::-1].copy()   # BGR -> RGB
        else:
            rgb = np.zeros((0, 3), np.uint8)

        return idx, info, pts, sizes, desc, rgb
    except Exception as e:
        return ("error", idx, path, f"{type(e).__name__}: {e}")


class FeatureStore:
    """Holds per-image features + intrinsics for the whole dataset."""

    def __init__(self):
        self.infos: dict[int, ImageInfo] = {}
        self.features: dict[int, FeatureSet] = {}
        self.colors: dict[int, np.ndarray] = {}
        self.K: dict[int, np.ndarray] = {}

    def coords(self, img_idx: int, feat_idx: int):
        kp = self.features[img_idx].keypoints[feat_idx]
        return float(kp[0]), float(kp[1])


def extract_features(paths: list[str], cfg: Config, log=print) -> FeatureStore:
    store = FeatureStore()
    os.makedirs(cfg.cache_dir, exist_ok=True)

    # figure out which images are cached vs need extraction
    todo = []
    for idx, path in enumerate(paths):
        key = _cache_key(path, cfg)
        cache_path = os.path.join(cfg.cache_dir, f"feat_{key}.npz")
        if os.path.isfile(cache_path):
            try:
                _load_cached(store, idx, path, cache_path, cfg)
                continue
            except Exception:
                pass  # corrupt cache -> re-extract
        todo.append((idx, path))

    n = len(paths)
    log(f"[features] {n} images ({n - len(todo)} cached, {len(todo)} to extract)")

    if todo:
        cfg_dict = _cfg_to_dict(cfg)
        workers = max(1, min(cfg.num_workers, len(todo)))
        state = {"done": n - len(todo), "failed": 0}
        processed: set[int] = set()

        def absorb(res, path):
            """Fold one worker result in, tolerating the error sentinel."""
            if isinstance(res, tuple) and res and res[0] == "error":
                _, bad_idx, bad_path, msg = res
                state["failed"] += 1
                processed.add(bad_idx)
                log(f"[features] skipped {os.path.basename(bad_path)}: {msg}")
                return
            idx = res[0]
            processed.add(idx)
            _absorb(store, res, cfg)
            _write_cache(store, idx, path, cfg)
            state["done"] += 1
            log(f"[features] {state['done']}/{n} {os.path.basename(path)} "
                f"({len(res[4])} kps)")

        if workers == 1:
            for idx, path in todo:
                absorb(_extract_one((idx, path, cfg_dict)), path)
        else:
            try:
                from concurrent.futures.process import BrokenProcessPool
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futs = {ex.submit(_extract_one, (idx, path, cfg_dict)): (idx, path)
                            for idx, path in todo}
                    for fut in as_completed(futs):
                        idx, path = futs[fut]
                        absorb(fut.result(), path)
            except (BrokenProcessPool, OSError, MemoryError) as e:
                log(f"[features] worker pool failed ({type(e).__name__}: {e}); "
                    f"falling back to single-process extraction")
                for idx, path in todo:
                    if idx in processed:
                        continue
                    absorb(_extract_one((idx, path, cfg_dict)), path)

        if state["failed"]:
            log(f"[features] {state['failed']} image(s) skipped due to errors")

    total = sum(len(f) for f in store.features.values())
    log(f"[features] done: {total:,} keypoints across {len(store.features)} images")
    return store


def _cfg_to_dict(cfg: Config) -> dict:
    from dataclasses import asdict
    return asdict(cfg)


def _absorb(store: FeatureStore, res, cfg: Config):
    idx, info, pts, sizes, desc, rgb = res
    store.infos[idx] = info
    store.features[idx] = FeatureSet(pts, sizes, desc)
    store.colors[idx] = rgb
    store.K[idx] = guess_intrinsics(info, cfg.focal_factor)


def _write_cache(store: FeatureStore, idx: int, path: str, cfg: Config):
    key = _cache_key(path, cfg)
    cache_path = os.path.join(cfg.cache_dir, f"feat_{key}.npz")
    info = store.infos[idx]
    fs = store.features[idx]
    try:
        np.savez_compressed(
            cache_path,
            kp=fs.keypoints, sizes=fs.sizes, desc=fs.descriptors,
            rgb=store.colors[idx],
            width=info.width, height=info.height,
            orig_width=info.orig_width, orig_height=info.orig_height,
            scale=info.scale,
            exif_focal_px=-1.0 if info.exif_focal_px is None else info.exif_focal_px,
        )
    except Exception:
        pass


def _load_cached(store: FeatureStore, idx: int, path: str, cache_path: str, cfg: Config):
    d = np.load(cache_path)
    focal = float(d["exif_focal_px"])
    info = ImageInfo(
        idx=idx, path=path, name=os.path.basename(path),
        width=int(d["width"]), height=int(d["height"]),
        orig_width=int(d["orig_width"]), orig_height=int(d["orig_height"]),
        scale=float(d["scale"]),
        exif_focal_px=None if focal < 0 else focal,
    )
    store.infos[idx] = info
    store.features[idx] = FeatureSet(
        d["kp"].astype(np.float32), d["sizes"].astype(np.float32),
        d["desc"].astype(np.float32))
    store.colors[idx] = d["rgb"].astype(np.uint8)
    store.K[idx] = guess_intrinsics(info, cfg.focal_factor)
