import numpy as np
_MIN_NORM: float = 1e-8


# Format conversion (xyzw ↔ wxyz)
def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """(..., 4) xyzw -> (..., 4) wxyz."""
    return np.ascontiguousarray(q[..., [3, 0, 1, 2]])


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 4) xyzw."""
    return np.ascontiguousarray(q[..., [1, 2, 3, 0]])

# Vector normalisation
def normalize_vec(v: np.ndarray, eps: float = _MIN_NORM) -> np.ndarray:
    """Normalise (…, N) vectors along the last axis.

    Zero-norm vectors are returned as-is.
    """
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    bad = (norm < eps) | np.isnan(norm)
    return np.where(bad, v, v / norm)

# Basis alignment
def angle_from_identity(q_local_xyzw: np.ndarray) -> np.ndarray:
    """Angular deviation (degrees) of each local quaternion from identity.

    ``angle = 2 * arccos(|w|)``  where *w* is the scalar part (index 3 in xyzw).
    """
    w = q_local_xyzw[..., 3]
    w = np.clip(np.abs(w), -1.0, 1.0)
    return np.rad2deg(2 * np.arccos(w))


def signed_angle_around_hinge(q_local_xyzw: np.ndarray) -> np.ndarray:
    """Signed angle (degrees) around the implied hinge axis.

    The hinge axis is assumed to be the local *x*-axis (y & z near zero for
    pure revolute joints like elbow / knee).  This returns a signed angle
    suitable for min/max revolute checks.

    For a quaternion ``(x, y, z, w)`` in xyzw order, the angle about the
    rotation axis is ``2 * atan2(norm(x,y,z), w)``.  Sign is determined by
    the dominant axis component.
    """
    x = q_local_xyzw[..., 0]
    w = q_local_xyzw[..., 3]
    # signed angle about the axis aligned with the largest vector component
    angle_rad = 2 * np.arctan2(np.abs(x), w)
    return np.rad2deg(angle_rad) * np.sign(x)