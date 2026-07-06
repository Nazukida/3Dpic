from __future__ import annotations
import cv2
import numpy as np

def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute the projection matrix P = K * [R | t]"""
    Rt = np.hstack((R, t.reshape(-1, 1)))  # Combine R and t into a single matrix
    P = K @ Rt  # Matrix multiplication
    return P

def triangulate_point(P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    pts1 = np.ascontiguousarray(pts1.T.astype(np.float64))
    pts2 = np.ascontiguousarray(pts2.T.astype(np.float64))
    X4 = cv2.triangulatePoints(P1, P2, pts1, pts2)
    X4 /= (X4[3:4, :] + 1e-12)
    return X4[:3].T.copy()

def reprojection_errors(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray, xy: np.ndarray) -> np.ndarray:
    Xc = (R @ X.T + t.reshape(3, 1)).T
    z = Xc[:, 2:3]
    valid = z[:, 0] > 1e-8
    proj = np.full((len(X), 2), np.inf)
    Xc_safe = Xc.copy()
    Xc_safe[~valid, 2] = 1.0
    uv = (K @ Xc_safe.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    proj[valid] = uv[valid]
    err = np.linalg.norm(proj - xy, axis = 1)
    err[~valid] = np.inf
    return err

def triangulation_angles(C1: np.ndarray, C2: np.ndarray, X: np.ndarray) -> np.ndarray:
    r1 = C1.reshape(1, 3) - X
    r2 = C2.reshape(1, 3) - X
    n1 = np.linalg.norm(r1, axis = 1) + 1e-12
    n2 = np.linalg.norm(r2, axis = 1) + 1e-12
    cosang = np.sum(r1 * r2, axis = 1) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (-R.T @ t.reshape(3, 1)).ravel()

def recover_pose_from_E(K1: np.ndarray, K2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, thresh_px: float = 1.5, conf: float = 0.9999):
    n1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2).astype(np.float64), K1, None).reshape(-1, 2)
    n2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2).astype(np.float64), K2, None).reshape(-1, 2)
    favg = 0.25 * (K1[0, 0] + K1[1, 1] + K2[0, 0] + K2[1, 1])
    thresh_n = thresh_px / max(favg, 1e-6)
    eye = np.eye(3)
    E, mask = cv2.findEssentialMat(n1, n2, cameraMatrix=eye, method=cv2.RANSAC, prob=conf, threshold=thresh_n)
    if E is None or mask is None:
        return None, None, None
    if E.shape[0] > 3:
        E = E[:3, :3]
    mask = mask.ravel().astype(bool)
    if mask.sum() < 5:
        return None, None, None
    # recoverPose's mask is an in/out param over ALL points: on input it
    # restricts to the essential-matrix inliers, on output it also encodes the
    # cheirality (in-front-of-both-cameras) test. Use it directly.
    pose_mask = mask.astype(np.uint8).reshape(-1, 1).copy()
    n_in, R, t, pose_mask = cv2.recoverPose(E, n1, n2, cameraMatrix=eye, mask=pose_mask)
    if R is None or n_in < 5:
        return None, None, None
    full_mask = pose_mask.ravel().astype(bool)
    return R, t.reshape(3), full_mask
    
def rodrigues_to_R(rvec:np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R