"""Feature-track construction via union-find.

A *track* is a set of feature observations across multiple images that all
correspond to the same physical 3D point.  We build them by unioning every
verified pairwise match: node = ``(image_idx, feature_idx)``.  Tracks that are
inconsistent (two features from the *same* image in one track) are split/dropped
because a real 3D point projects to at most one feature per image.
"""

from __future__ import annotations

import numpy as np

from .matching import MatchGraph


class _UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def find(self, x: int) -> int:
        p = self.parent
        # iterative with path compression
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def add(self, x: int):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


class TrackTable:
    """Maps ``(image_idx, feature_idx) -> track_id`` and back."""

    def __init__(self):
        # track_id -> dict(image_idx -> feature_idx)
        self.tracks: dict[int, dict[int, int]] = {}
        # (image_idx, feature_idx) -> track_id
        self.obs_to_track: dict[tuple[int, int], int] = {}

    def track_length_hist(self) -> dict[int, int]:
        hist: dict[int, int] = {}
        for t in self.tracks.values():
            L = len(t)
            hist[L] = hist.get(L, 0) + 1
        return hist


def _encode(img: int, feat: int) -> int:
    # pack (image, feature) into one int; 24 bits for feature index is plenty
    return (img << 24) | (feat & 0xFFFFFF)


def _decode(code: int) -> tuple[int, int]:
    return code >> 24, code & 0xFFFFFF


def build_tracks(graph: MatchGraph, min_len: int = 2, log=print) -> TrackTable:
    uf = _UnionFind()

    for (i, j), matches in graph.pairs.items():
        for fi, fj in matches:
            a = _encode(i, int(fi))
            b = _encode(j, int(fj))
            uf.add(a)
            uf.add(b)
            uf.union(a, b)

    # gather members per root
    groups: dict[int, list[int]] = {}
    for node in uf.parent:
        root = uf.find(node)
        groups.setdefault(root, []).append(node)

    table = TrackTable()
    tid = 0
    dropped_conflict = 0
    dropped_short = 0

    for members in groups.values():
        # resolve conflicts: if an image contributes >1 feature, the track is
        # ambiguous. Keep the largest consistent subset by dropping the whole
        # image from the track (safer than guessing which feature is right).
        by_image: dict[int, list[int]] = {}
        for code in members:
            img, feat = _decode(code)
            by_image.setdefault(img, []).append(feat)

        obs = {}
        conflicted = False
        for img, feats in by_image.items():
            if len(feats) == 1:
                obs[img] = feats[0]
            else:
                conflicted = True  # skip this image's obs entirely
        if conflicted and len(obs) < 2:
            dropped_conflict += 1
            continue
        if len(obs) < min_len:
            dropped_short += 1
            continue

        table.tracks[tid] = obs
        for img, feat in obs.items():
            table.obs_to_track[(img, feat)] = tid
        tid += 1

    hist = table.track_length_hist()
    hist_str = ", ".join(f"len{k}:{v}" for k, v in sorted(hist.items())[:6])
    log(f"[tracks] {len(table.tracks):,} tracks "
        f"(dropped {dropped_short:,} short, {dropped_conflict:,} conflicted) "
        f"[{hist_str}]")
    return table
