"""Pairwise feature matching with geometric verification.

For each candidate image pair we:
1. kNN-match descriptors (FLANN) both directions,
2. apply Lowe's ratio test + mutual-consistency,
3. verify with a fundamental-matrix RANSAC to drop outliers.

Matching is O(pairs) so it dominates runtime on big sets; we parallelise across
pairs and keep only the surviving inlier index pairs (tiny) in memory.

By default every pair is considered (exhaustive), which is robust for the small
/ medium sets this CPU pipeline targets.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

import cv2
import numpy as np

from .config import Config
from .features import FeatureStore


# module-level handle so pool workers can see descriptors without re-pickling
# the whole store for every task (set in match_all before submitting).
_DESC: dict[int, np.ndarray] = {}
_KP: dict[int, np.ndarray] = {}


def _init_worker(desc, kp):
    global _DESC, _KP
    _DESC = desc
    _KP = kp


def _ratio_match(d1: np.ndarray, d2: np.ndarray, ratio: float) -> np.ndarray:
    """Return (K,2) int array of mutually-consistent ratio-test matches."""
    if len(d1) < 2 or len(d2) < 2:
        return np.zeros((0, 2), np.int32)

    # FLANN KD-tree on SIFT (float) descriptors.
    index_params = dict(algorithm=1, trees=4)   # FLANN_INDEX_KDTREE
    search_params = dict(checks=32)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    def one_way(a, b):
        raw = flann.knnMatch(a, b, k=2)
        good = {}
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good[m.queryIdx] = m.trainIdx
        return good

    fwd = one_way(d1, d2)
    bwd = one_way(d2, d1)
    # mutual consistency: i->j and j->i agree
    matches = [(i, j) for i, j in fwd.items() if bwd.get(j, -1) == i]
    if not matches:
        return np.zeros((0, 2), np.int32)
    return np.array(matches, np.int32)


_EMPTY = np.zeros((0, 2), np.int32)


def _verify_pair(args):
    """Worker: match + geometric verify one image pair.

    Must NEVER raise: it runs in a process-pool worker and any exception here
    would propagate through ``fut.result()`` and abort the entire matching
    stage (and thus the whole reconstruction). Degenerate inputs — collinear /
    coincident points, too few matches, an unstable RANSAC solve — are all
    treated as "this pair has no usable matches" and return an empty result.
    """
    i, j, ratio, thresh_px, seed = args
    try:
        # Seed OpenCV's RNG deterministically from the pair identity so the
        # RANSAC result is identical no matter which worker runs the pair or in
        # what order (workers don't inherit the parent process RNG state).
        cv2.setRNGSeed(seed + i * 100003 + j)
        d1, d2 = _DESC[i], _DESC[j]
        m = _ratio_match(d1, d2, ratio)
        if len(m) < 8:
            return i, j, _EMPTY

        p1 = _KP[i][m[:, 0]]
        p2 = _KP[j][m[:, 1]]
        F, mask = cv2.findFundamentalMat(
            p1, p2, method=cv2.USAC_MAGSAC, ransacReprojThreshold=thresh_px,
            confidence=0.9999, maxIters=10000)
        if F is None or mask is None:
            return i, j, _EMPTY
        mask = mask.ravel().astype(bool)
        # Guard: a degenerate USAC solve can return a mask whose length doesn't
        # match the input correspondences. Indexing m with it would raise.
        if mask.shape[0] != len(m):
            return i, j, _EMPTY
        inliers = m[mask]
        return i, j, inliers
    except Exception as e:
        # Report the failure code back to the parent (which logs it) rather
        # than crashing the worker.
        return i, j, ("error", f"{type(e).__name__}: {e}")


class MatchGraph:
    """Symmetric store of inlier matches between image pairs."""

    def __init__(self):
        self.pairs: dict[tuple[int, int], np.ndarray] = {}

    def add(self, i: int, j: int, matches: np.ndarray):
        if len(matches) == 0:
            return
        a, b = (i, j) if i < j else (j, i)
        if i < j:
            self.pairs[(a, b)] = matches
        else:
            self.pairs[(a, b)] = matches[:, ::-1].copy()

    def get(self, i: int, j: int) -> np.ndarray:
        a, b = (i, j) if i < j else (j, i)
        m = self.pairs.get((a, b))
        if m is None:
            return np.zeros((0, 2), np.int32)
        return m if i < j else m[:, ::-1]

    def neighbors(self, i: int) -> list[int]:
        out = []
        for (a, b) in self.pairs:
            if a == i:
                out.append(b)
            elif b == i:
                out.append(a)
        return out


def match_all(store: FeatureStore, cfg: Config, log=print) -> MatchGraph:
    img_ids = sorted(store.features.keys())
    all_pairs = list(combinations(img_ids, 2))
    graph = MatchGraph()

    ratio = 1.0 / cfg.ratio_test if cfg.ratio_test > 1.0 else cfg.ratio_test
    # Config stores ratio_test as e.g. 2.0 meaning "d1 < d2/2" style; normalise
    # to Lowe ratio in (0,1). If someone sets 0.75 directly, respect it.
    if ratio >= 1.0:
        ratio = 0.75

    desc = {i: store.features[i].descriptors for i in img_ids}
    kp = {i: store.features[i].keypoints for i in img_ids}

    log(f"[match] {len(all_pairs)} pairs, ratio={ratio:.2f}, "
        f"thresh={cfg.max_reproj_error_px}px")

    tasks = [(i, j, ratio, cfg.max_reproj_error_px, cfg.seed) for i, j in all_pairs]
    workers = max(1, min(cfg.num_workers, len(tasks)))

    stats = {"inliers": 0, "failed": 0}
    completed: set[tuple[int, int]] = set()   # (min,max) keys actually processed

    def absorb(i, j, inl):
        """Fold one worker result into the graph, tolerating error sentinels."""
        key = (min(i, j), max(i, j))
        if key in completed:
            return                            # guard against any double-report
        completed.add(key)
        if isinstance(inl, tuple) and len(inl) == 2 and inl[0] == "error":
            stats["failed"] += 1
            log(f"[match] pair ({i},{j}) skipped: {inl[1]}")
        else:
            graph.add(i, j, inl)
            stats["inliers"] += len(inl)
        n = len(completed)
        if n % 20 == 0 or n == len(tasks):
            log(f"[match] {n}/{len(tasks)} pairs ({stats['inliers']:,} inliers)")

    if workers == 1:
        _init_worker(desc, kp)
        for t in tasks:
            i, j, inl = _verify_pair(t)
            absorb(i, j, inl)
    else:
        try:
            from concurrent.futures.process import BrokenProcessPool
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_init_worker,
                                     initargs=(desc, kp)) as ex:
                futs = [ex.submit(_verify_pair, t) for t in tasks]
                for fut in as_completed(futs):
                    i, j, inl = fut.result()
                    absorb(i, j, inl)
        except (BrokenProcessPool, OSError, MemoryError) as e:
            # A worker died hard (segfault, OOM in spawn, etc). Don't abort the
            # whole reconstruction — fall back to single-process matching for
            # any pairs not yet completed.
            log(f"[match] worker pool failed ({type(e).__name__}: {e}); "
                f"falling back to single-process matching")
            _init_worker(desc, kp)
            for (i, j, r, th, sd) in tasks:
                if (min(i, j), max(i, j)) in completed:
                    continue
                ri, rj, inl = _verify_pair((i, j, r, th, sd))
                absorb(ri, rj, inl)

    tail = (f", {stats['failed']} pairs skipped" if stats["failed"] else "")
    log(f"[match] done: {len(graph.pairs)} verified pairs, "
        f"{stats['inliers']:,} total inlier matches{tail}")
    return graph
