# recon3d

CPU-only 3D reconstruction from photos, with a fast OpenGL point-cloud viewer
and full theme customization. No GPU required — the whole pipeline (feature
extraction, matching, structure-from-motion, dense MVS) runs on the CPU, and the
viewer stays smooth into the millions of points.

## Features

- **Photogrammetry pipeline** — turn a folder of overlapping photos into a
  coloured 3D point cloud: SIFT features → geometric-verified matching →
  incremental structure-from-motion with bundle adjustment → SGBM dense stereo.
- **Fast PLY viewer** — open existing `.ply` clouds and orbit them fluidly.
  Points are drawn as lit sphere imposters with distance attenuation; huge
  clouds stay interactive via adaptive level-of-detail during camera moves.
- **Themes** — eight polished presets (Midnight, Graphite, Nord, Solarized
  Dark, Amber Lab, Neon, Snow, Paper) plus a live theme editor that recolours
  both the UI and the 3D canvas. Custom themes are saved to disk.

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.11. Dependencies: numpy, opencv-contrib-python, pillow,
PySide6, scipy, tqdm.

## Usage

### Desktop app

```bash
recon3d-gui             # or: python main.py
recon3d-gui cloud.ply   # open a PLY directly
```

- **Orbit** left-drag · **Pan** right-drag · **Zoom** wheel · **Reset** `F`
- Drag-and-drop a `.ply` onto the window to open it.
- The left dock runs a reconstruction; the right dock controls rendering and
  the active theme.

### Command line

```bash
recon3d --images ./photos --out ./output --quality balanced
```

Quality presets: `fast`, `balanced`, `high`, `ultra` (trade speed for image
resolution, feature count, and dense stereo range). Add `--no-dense` for a
sparse-only cloud, `--workers N` to set the process pool size.

Outputs land in the work directory: `sparse.ply`, `dense.ply`, `cameras.ply`,
and `scene.json` (summary + camera count + timing).

## How it works

| Stage | Module | What it does |
|-------|--------|--------------|
| Features | `features.py` | SIFT keypoints + descriptors, cached to `.npz` |
| Matching | `matching.py` | FLANN ratio test + mutual check + fundamental-matrix RANSAC |
| Tracks | `tracks.py` | Union-find linking of matches into multi-view tracks |
| SfM | `sfm.py` | Seed pair → PnP registration → triangulation → bundle adjustment |
| Dense | `dense.py` | Per-pair rectified SGBM stereo, fused and outlier-filtered |
| Viewer | `gui/` | PySide6 + OpenGL point-cloud renderer with theming |

Reconstruction is seeded (`Config.seed`) for near-reproducible runs (FLANN's
approximate matching leaves a fraction of a percent of variation).
