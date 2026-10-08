"""Camera geometry: pose construction, convention conversion, SE(3), error metrics.

Conventions used throughout the project
---------------------------------------
* The dataset's `RT` is **OpenGL world-to-camera** (camera looks -z, y up).
* gsplat / OpenCV / cv2.solvePnP look +z.  The two differ by S = diag(1,-1,-1),
  which is its own inverse.
* Cameras sit on a sphere about the world ORIGIN and look at the object's mesh
  bounding-box centre (NOT the origin — using the origin leaves a ~0.5° residual):

      C = r * (cos(el)cos(az), sin(el), cos(el)sin(az))        # world is y-up
      z = normalise(C - target); x = normalise(up x z); y = z x x
      R_w2c = [x y z]^T ;  t = -R_w2c C

  `verify_pose_convention()` reproduces every shipped reference `RT` from its
  (radius, azimuth, elevation) to ~1e-7 with this formula.
"""

import numpy as np

S3 = np.diag([1.0, -1.0, -1.0])
S4 = np.diag([1.0, -1.0, -1.0, 1.0])          # OpenGL <-> OpenCV (involution)
UP = np.array([0.0, 1.0, 0.0])


# --------------------------------------------------------------------------- #
# Pose construction
# --------------------------------------------------------------------------- #
def camera_centre_from_azel(radius, azim_deg, elev_deg):
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    return radius * np.array([np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)])


def pose_from_azel(radius, azim_deg, elev_deg, target):
    """(radius, azimuth, elevation) -> 4x4 OpenGL world-to-camera matrix."""
    C = camera_centre_from_azel(radius, azim_deg, elev_deg)
    z = C - np.asarray(target, dtype=np.float64)
    z = z / np.linalg.norm(z)
    x = np.cross(UP, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1).T
    RT = np.eye(4)
    RT[:3, :3] = R
    RT[:3, 3] = -R @ C
    return RT


# --------------------------------------------------------------------------- #
# Convention conversion
# --------------------------------------------------------------------------- #
def gl_w2c_to_cv_viewmat(RT_gl):
    """OpenGL world-to-camera -> OpenCV/gsplat viewmat."""
    return S4 @ np.asarray(RT_gl, dtype=np.float64)


def cv_viewmat_to_gl_w2c(V_cv):
    """OpenCV/gsplat viewmat -> OpenGL world-to-camera (S4 is an involution)."""
    return S4 @ np.asarray(V_cv, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Error metrics
# --------------------------------------------------------------------------- #
def camera_centre(R, t):
    """Camera centre in world coords for a world-to-camera (R, t)."""
    return -np.asarray(R).T @ np.asarray(t)


def rotation_error_deg(R_a, R_b):
    """Geodesic angle between two rotations, in degrees."""
    c = (np.trace(np.asarray(R_a) @ np.asarray(R_b).T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def pose_error(RT_est, RT_gt):
    """Return (rotation_deg, camera_centre_distance) between two OpenGL w2c poses."""
    RT_est, RT_gt = np.asarray(RT_est), np.asarray(RT_gt)
    rot = rotation_error_deg(RT_est[:3, :3], RT_gt[:3, :3])
    d = np.linalg.norm(camera_centre(RT_est[:3, :3], RT_est[:3, 3])
                       - camera_centre(RT_gt[:3, :3], RT_gt[:3, 3]))
    return rot, float(d)


def azimuth_diff(a, b):
    """Smallest absolute azimuth difference in degrees (handles 0/360 wrap)."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def azimuth_diff_symmetric(a, b, period=180.0):
    """Azimuth difference modulo an object symmetry period.

    Object-dependent: the bag `shape_8430` is ~180°-symmetric in appearance
    (theta and theta+180 render near-identically, grayscale L1 1.9/255 vs 21.8
    for an 18° neighbour), so for it theta vs theta+180 is not resolvable from a
    single image. Most chairs are NOT symmetric — pass period=0 to disable.
    """
    d = azimuth_diff(a, b)
    return min(d, abs(period - d)) if period > 0 else d


def world_rotation_about_up(angle_deg, target):
    """4x4 world transform rotating by `angle` about the up axis through `target`."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])   # about +y (world up)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(target, dtype=np.float64) - R @ np.asarray(target, dtype=np.float64)
    return T


def symmetry_group(period_deg, target):
    """World transforms of a discrete azimuth symmetry (identity first).

    period 0 or 360 -> [identity]; 180 -> [I, rot180]; 90 -> [I, 90, 180, 270].
    """
    if period_deg <= 0 or period_deg >= 360:
        return [np.eye(4)]
    n = int(round(360.0 / period_deg))
    return [world_rotation_about_up(k * period_deg, target) for k in range(n)]


def pose_error_symmetric(RT_est, RT_gt, target, period_deg=0.0):
    """Pose error minimised over the object's azimuth symmetry group.

    If the object is symmetric, the camera pose is only determined up to that
    symmetry, so scoring against a single ground-truth pose over-penalises an
    equally-valid solution. A world symmetry S maps the ground truth to the
    equivalent pose `RT_gt @ S`.

    Returns (rotation_deg, centre_distance, k) where k indexes the group element.
    """
    best = None
    for k, S in enumerate(symmetry_group(period_deg, target)):
        rot, d = pose_error(RT_est, np.asarray(RT_gt) @ S)
        if best is None or rot + d < best[0] + best[1]:
            best = (rot, d, k)
    return best


# --------------------------------------------------------------------------- #
# SE(3) / quaternion helpers (torch, used by the pose refiner)
# --------------------------------------------------------------------------- #
def se3_exp(xi):
    """xi: (6,) torch tensor [omega(3), nu(3)] -> 4x4 SE(3) matrix."""
    import torch
    omega, nu = xi[:3], xi[3:]
    theta = torch.linalg.norm(omega) + 1e-12
    K = torch.zeros(3, 3, device=xi.device, dtype=xi.dtype)
    K[0, 1], K[0, 2], K[1, 0] = -omega[2], omega[1], omega[2]
    K[1, 2], K[2, 0], K[2, 1] = -omega[0], -omega[1], omega[0]
    K = K / theta
    I = torch.eye(3, device=xi.device, dtype=xi.dtype)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    V = (I + (1 - torch.cos(theta)) / theta * K
         + (theta - torch.sin(theta)) / theta * (K @ K))
    T = torch.eye(4, device=xi.device, dtype=xi.dtype)
    T[:3, :3] = R
    T[:3, 3] = V @ nu
    return T


def quat_mul(a, b):
    import torch
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def mat_to_quat(R):
    import torch
    w = torch.sqrt(torch.clamp(1 + R[0, 0] + R[1, 1] + R[2, 2], min=1e-8)) / 2
    return torch.stack([w, (R[2, 1] - R[1, 2]) / (4 * w),
                        (R[0, 2] - R[2, 0]) / (4 * w),
                        (R[1, 0] - R[0, 1]) / (4 * w)])


# --------------------------------------------------------------------------- #
def verify_pose_convention(references, target, tol=1e-5):
    """Rebuild each reference RT from its (radius, azim, elev) and compare.

    references: iterable of dicts with keys 'radius', 'azim', 'elev', 'RT'.
    Returns (max_dR, max_dt, ok).
    """
    dR, dt = [], []
    for ref in references:
        RT = pose_from_azel(ref["radius"], ref["azim"], ref["elev"], target)
        gt = np.asarray(ref["RT"], dtype=np.float64)
        dR.append(np.abs(RT[:3, :3] - gt[:3, :3]).max())
        dt.append(np.abs(RT[:3, 3] - gt[:3, 3]).max())
    mR, mt = float(max(dR)), float(max(dt))
    return mR, mt, (mR < tol and mt < tol)
