from __future__ import annotations

from dataclasses import dataclass, field
import os
@dataclass
class Config:
    #IO
    image_dir: str = ""
    work_dir: str = "output"
    max_image_size: int = 1600
    
    sift_n_features: int = 8000
    sift_contrast_threshold: float = 0.004
    sift_edge_threshold: float = 10.0
    
    ratio_test: float = 2.0
    max_reproj_error_px: float = 4.0
    pnp_ransac_thresh_px: float = 4.0
    init_min_inliers: int = 80
    
    ba_after_n_images: int = 1
    ba_max_iters: int = 30
    ba_local_max_iters: int = 10
    ba_ftol: float = 1e-4
    refine_intrinsics: bool = True
    
    focal_factor: float = 1.2
    
    radial_distortion: float = 0.0
    undistort_images: bool = True
    
    do_dense: bool = True
    dense_max_size: int = 1200
    dense_num_neighbors: int = 4
    sgbm_min_disp: int = 0
    sgbm_num_disp: int = 256
    sgbm_block_size: int = 5
    sgbm_uniqueness: int = 5
    sgbm_speckle_window: int = 100
    sgbm_speckle_range: int = 2
    dense_min_views: int = 2
    dense_geo_consistency_px: float = 1.0
    dense_max_depth_diff_ratio: float = 0.01
    
    sor_neighbors: int = 16
    sor_std_ratio: float = 2.0
    voxel_downsample: float = 0.0
    # Upper bound on the final dense point count. The dense stage can fuse tens
    # of millions of points; the viewer has level-of-detail so a few million is
    # plenty and keeps the memory-heavy cleanup (KD-tree neighbour search) fast.
    dense_max_points: int = 5000000
    
    num_workers: int = max(1, (os.cpu_count() or 4) - 1)
    seed: int = 42
    verbose: bool = True
    
    def ensure_dirs(self) -> None:
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, 'cache'), exist_ok=True)
        
    @property
    def cache_dir(self) -> str:
        return os.path.join(self.work_dir, 'cache')