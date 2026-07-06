"""Incremental Structure-from-Motion.

Classic COLMAP-style incremental pipeline built on the project's ``geometry``
and ``bundle`` helpers:

1. **Seed** - choose an initial image pair with enough inliers *and* a wide
   enough triangulation angle (a small-baseline pair triangulates badly), then
   recover relative pose from the essential matrix and triangulate.
2. **Grow** - repeatedly pick the unregistered image with the most 2D-3D
   correspondences (via shared tracks), solve its pose with PnP-RANSAC,
   triangulate the tracks it newly connects, and periodically run bundle
   adjustment + outlier filtering.

The result is a :class:`Reconstruction` (cameras + coloured 3D points).
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .features import FeatureStore
from .matching import MatchGraph
from .tracks import TrackTable
from .scene import Reconstruction
from .geometry import (
    recover_pose_from_E, triangulate_point, reprojection_errors,
    triangulation_angles, camera_center, projection_matrix,
)
from .bundle import run_bundle_adjustment


class SfM:
    def __init__(self, store: FeatureStore, graph: MatchGraph,
                 tracks: TrackTable, cfg: Config, log=print):
        self.store = store
        self.graph = graph
        self.tracks = tracks
        self.cfg = cfg
        self.log = log
        self.recon = Reconstruction()

    # ------------------------------------------------------------------ #
    def _coords(self, img_idx: int, feat_idx: int):
        kp = self.store.features[img_idx].keypoints[feat_idx]
        return float(kp[0]), float(kp[1])

    def _pts(self, img_idx: int, feat_indices: np.ndarray) -> np.ndarray:
        return self.store.features[img_idx].keypoints[feat_indices]

    # ------------------------------------------------------------------ #
    # Seed selection
    # ------------------------------------------------------------------ #
    def _choose_initial_pair(self) -> tuple[int, int] | None:
        best = None
        best_score = -1.0
        # rank pairs by inlier count, then check the triangulation angle of the
        # recovered geometry; prefer many inliers AND a healthy median angle.
        # Tie-break on (i, j) so the ranking is independent of the order in
        # which parallel matching workers happened to populate graph.pairs.
        ranked = sorted(self.graph.pairs.items(),
                        key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1]))
        for (i, j), matches in ranked[:40]:   # only probe the top pairs
            if len(matches) < self.cfg.init_min_inliers:
                continue
            Ki, Kj = self.store.K[i], self.store.K[j]
            p1 = self._pts(i, matches[:, 0])
            p2 = self._pts(j, matches[:, 1])
            R, t, mask = recover_pose_from_E(Ki, Kj, p1, p2,
                                             thresh_px=self.cfg.pnp_ransac_thresh_px)
            if R is None:
                continue
            inl = int(mask.sum())
            if inl < self.cfg.init_min_inliers:
                continue
            # triangulate the inliers to measure the parallax angle
            P1 = projection_matrix(Ki, np.eye(3), np.zeros(3))
            P2 = projection_matrix(Kj, R, t)
            X = triangulate_point(P1, P2, p1[mask], p2[mask])
            C1 = np.zeros(3)
            C2 = camera_center(R, t)
            angles = triangulation_angles(C1, C2, X)
            med_ang = float(np.median(angles)) if len(angles) else 0.0
            # score rewards inliers but needs a minimum parallax
            if med_ang < 2.0:
                continue
            score = inl * min(med_ang, 15.0)
            if score > best_score:
                best_score = score
                best = (i, j, R, t, mask, X, med_ang)
        if best is None:
            return None
        i, j, R, t, mask, X, med_ang = best
        self.log(f"[sfm] initial pair ({i},{j}): {int(mask.sum())} inliers, "
                 f"median angle {med_ang:.1f}deg")
        self._init_from_pair(i, j, R, t, mask)
        return i, j

    def _init_from_pair(self, i, j, R, t, mask):
        Ki, Kj = self.store.K[i], self.store.K[j]
        self.recon.add_camera(i, np.eye(3), np.zeros(3), Ki.copy())
        self.recon.add_camera(j, R, t, Kj.copy())

        matches = self.graph.get(i, j)[mask]
        p1 = self._pts(i, matches[:, 0])
        p2 = self._pts(j, matches[:, 1])
        P1 = projection_matrix(Ki, np.eye(3), np.zeros(3))
        P2 = projection_matrix(Kj, R, t)
        X = triangulate_point(P1, P2, p1, p2)

        # keep only points in front of both cameras with small reproj error
        e1 = reprojection_errors(Ki, np.eye(3), np.zeros(3), X, p1)
        e2 = reprojection_errors(Kj, R, t, X, p2)
        good = (e1 < self.cfg.max_reproj_error_px) & (e2 < self.cfg.max_reproj_error_px)

        added = 0
        for k in np.where(good)[0]:
            fi, fj = int(matches[k, 0]), int(matches[k, 1])
            tid = self.tracks.obs_to_track.get((i, fi))
            if tid is None:
                continue
            if self.recon.point_for_track(tid) is not None:
                continue
            color = self.store.colors[i][fi]
            obs = self.tracks.tracks[tid]
            self.recon.add_point(X[k], color, tid, obs)
            added += 1
        self.log(f"[sfm] seeded {added} points from initial pair")

    # ------------------------------------------------------------------ #
    # Growing
    # ------------------------------------------------------------------ #
    def _find_2d3d(self, img_idx: int):
        """Collect 2D-3D correspondences for an unregistered image."""
        obj_pts, img_pts, feat_ids, track_ids = [], [], [], []
        feats = self.store.features[img_idx]
        for feat_idx in range(len(feats.keypoints)):
            tid = self.tracks.obs_to_track.get((img_idx, feat_idx))
            if tid is None:
                continue
            pt = self.recon.point_for_track(tid)
            if pt is None:
                continue
            obj_pts.append(pt.xyz)
            img_pts.append(feats.keypoints[feat_idx])
            feat_ids.append(feat_idx)
            track_ids.append(tid)
        if not obj_pts:
            return None
        return (np.array(obj_pts, np.float64),
                np.array(img_pts, np.float64),
                feat_ids, track_ids)

    def _register_next(self) -> int | None:
        import cv2
        registered = set(self.recon.cameras.keys())
        # sorted() so the candidate scan order (and thus the correspondence
        # tie-break below) is independent of feature-extraction worker order.
        candidates = sorted(i for i in self.store.features
                            if i not in registered and i not in self._blacklist)

        # Rank every candidate by available 2D-3D correspondences, then try PnP
        # from best to worst. Crucially, a PnP failure on one candidate must NOT
        # abort registration of the others: we blacklist the failure and fall
        # through to the next. Only after ALL candidates are exhausted do we
        # return None, which the growth loop reads as "nothing left".
        ranked = []
        for img_idx in candidates:
            corr = self._find_2d3d(img_idx)
            if corr is None:
                continue
            n = len(corr[2])
            if n >= 6:
                ranked.append((n, img_idx, corr))
        ranked.sort(key=lambda r: (-r[0], r[1]))
        if not ranked:
            return None

        for best_n, best, best_corr in ranked:
            obj_pts, img_pts, feat_ids, track_ids = best_corr
            K = self.store.K[best]
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts.reshape(-1, 1, 3), img_pts.reshape(-1, 1, 2), K, None,
                reprojectionError=self.cfg.pnp_ransac_thresh_px,
                confidence=0.9999, iterationsCount=1000,
                flags=cv2.SOLVEPNP_EPNP)
            if not ok or inliers is None or len(inliers) < 6:
                self.log(f"[sfm] PnP failed for image {best} ({best_n} corr), "
                         f"trying next candidate")
                self._blacklist.add(best)
                continue

            R = cv2.Rodrigues(rvec)[0]
            t = tvec.ravel()
            self.recon.add_camera(best, R, t, K.copy())
            self.log(f"[sfm] registered image {best}: {len(inliers)}/{best_n} "
                     f"PnP inliers ({len(self.recon.cameras)} cams total)")
            return best

        # every remaining candidate failed PnP this round
        return None

    def _triangulate_new(self, new_img: int):
        """Triangulate tracks connecting the new image to registered ones."""
        cam_new = self.recon.cameras[new_img]
        P_new = cam_new.P()
        C_new = cam_new.center
        feats_new = self.store.features[new_img]
        added = 0

        for other in self.recon.registered_images:
            if other == new_img:
                continue
            matches = self.graph.get(new_img, other)
            if len(matches) == 0:
                continue
            cam_o = self.recon.cameras[other]
            P_o = cam_o.P()
            C_o = cam_o.center

            p_new = self._pts(new_img, matches[:, 0])
            p_o = self._pts(other, matches[:, 1])
            X = triangulate_point(P_new, P_o, p_new, p_o)

            e_new = reprojection_errors(cam_new.K, cam_new.R, cam_new.t, X, p_new)
            e_o = reprojection_errors(cam_o.K, cam_o.R, cam_o.t, X, p_o)
            ang = triangulation_angles(C_new, C_o, X)
            good = ((e_new < self.cfg.max_reproj_error_px) &
                    (e_o < self.cfg.max_reproj_error_px) &
                    (ang > 1.5) & (ang < 178.0))

            for k in np.where(good)[0]:
                fi = int(matches[k, 0])
                tid = self.tracks.obs_to_track.get((new_img, fi))
                if tid is None:
                    continue
                if self.recon.point_for_track(tid) is not None:
                    continue
                color = self.store.colors[new_img][fi]
                obs = self.tracks.tracks[tid]
                self.recon.add_point(X[k], color, tid, obs)
                added += 1
        if added:
            self.log(f"[sfm] triangulated {added} new points")
        return added

    def _filter_outliers(self):
        """Drop points whose mean reprojection error is too large."""
        thresh = self.cfg.max_reproj_error_px * 2.0
        to_remove = []
        for pid, pt in self.recon.points.items():
            errs = []
            for img_idx, feat_idx in pt.obs.items():
                cam = self.recon.cameras.get(img_idx)
                if cam is None:
                    continue
                xy = np.array([self._coords(img_idx, feat_idx)])
                e = reprojection_errors(cam.K, cam.R, cam.t,
                                        pt.xyz.reshape(1, 3), xy)[0]
                if np.isfinite(e):
                    errs.append(e)
            if errs:
                mean_e = float(np.mean(errs))
                pt.error = mean_e
                if mean_e > thresh:
                    to_remove.append(pid)
            else:
                to_remove.append(pid)
        for pid in to_remove:
            self.recon.remove_point(pid)
        if to_remove:
            self.log(f"[sfm] filtered {len(to_remove)} high-error points")

    def _bundle(self, label: str):
        run_bundle_adjustment(
            self.recon, self._coords,
            refine_focal=self.cfg.refine_intrinsics,
            max_iters=self.cfg.ba_max_iters,
            ftol=self.cfg.ba_ftol, verbose=False,
            log=self.log, label=label)

    # ------------------------------------------------------------------ #
    def run(self) -> Reconstruction:
        self._blacklist: set[int] = set()
        if self._choose_initial_pair() is None:
            self.log("[sfm] ERROR: no suitable initial pair found")
            return self.recon

        self._bundle("init")
        self._filter_outliers()

        n_since_ba = 0
        while True:
            new_img = self._register_next()
            if new_img is None:
                break
            self._triangulate_new(new_img)
            n_since_ba += 1
            if n_since_ba >= self.cfg.ba_after_n_images:
                self._bundle(f"global@{len(self.recon.cameras)}")
                self._filter_outliers()
                n_since_ba = 0

        # final global refinement
        self._bundle("final")
        self._filter_outliers()
        self.log(f"[sfm] complete: {self.recon.stats()}")
        return self.recon
