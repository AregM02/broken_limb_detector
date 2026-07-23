import numpy as np

# Extrinsic rotations around fixed parent frame XYZ axes.
# Root and Euler-wrapped axes intentionally retain their full rotation range.
ABSOLUTE_LIMITS: dict[str, np.ndarray] = {
    "Hip": np.array([[-180, 180], [-180, 180], [-180, 180]], dtype=float),
    "Ab": np.array([[-10, 20], [-10, 50], [-10, 10]], dtype=float),
    "Chest": np.array([[-15, 15], [-10, 50], [-15, 20]], dtype=float),
    "Neck": np.array([[-20, 95], [-40, 50], [-35, 30]], dtype=float),
    "LShoulder": np.array([[-5, 8], [-9, 24], [-15, 30]], dtype=float),
    "RShoulder": np.array([[-7, 5], [-30, 20], [-35, 30]], dtype=float),
    "LUArm": np.array([[-66, 104], [-56, 6], [-88, 5]], dtype=float),
    "RUArm": np.array([[-73, 87], [-27, 52], [-11, 84]], dtype=float),
    "LFArm": np.array([[-180, 180], [-90, 10], [-180, 180]], dtype=float),
    "RFArm": np.array([[-180, 180], [-2, 90], [-180, 180]], dtype=float),
    "LHand": np.array([[-50, 43], [-36, 33], [-63, 88]], dtype=float),
    "RHand": np.array([[-45, 50], [-33, 36], [-88, 63]], dtype=float),
    "LThigh": np.array([[-100, 14], [-8, 26], [-37, 26]], dtype=float),
    "RThigh": np.array([[-97, 10], [-19, 12], [-33, 44]], dtype=float),
    "LShin": np.array([[-15, 135], [-5, 5], [-5, 5]], dtype=float),
    "RShin": np.array([[-15, 135], [-5, 5], [-5, 5]], dtype=float),
    "LFoot": np.array([[-15, 52], [-43, 73], [-23, 43]], dtype=float),
    "RFoot": np.array([[-22, 44], [-73, 48.5], [-85, 66]], dtype=float),
    "Head": np.array([[-5, 40], [-40, 44], [-22, 21]], dtype=float),
}
END_EFFECTOR_BONES: tuple[str, ...] = tuple(ABSOLUTE_LIMITS)

# Maximum calibrated IMU angular acceleration magnitude in validation.parquet [rad/s^2].
# Ab and Neck use the nearest available IMUs: lower_back and head.
ANGULAR_ACCELERATION_LIMITS: dict[str, float] = {
    "Hip": 548.520,
    "Ab": 548.520,
    "Chest": 564.905,
    "Neck": 700.322,
    "LShoulder": 667.633,
    "RShoulder": 647.653,
    "LUArm": 1995.744,
    "RUArm": 1365.191,
    "LFArm": 4579.496,
    "RFArm": 2959.852,
    "LHand": 5427.315,
    "RHand": 3640.684,
    "LThigh": 748.306,
    "RThigh": 816.365,
    "LShin": 830.999,
    "RShin": 1159.091,
    "LFoot": 829.617,
    "RFoot": 1202.875,
    "Head": 700.322,
}

# Parent-relative angular speed limits [rad/s].
ANGULAR_VELOCITY_LIMITS: dict[str, float] = {
    "Hip": 10.0,
    "Ab": 10.0,
    "Chest": 10.0,
    "Neck": 10.0,
    "LShoulder": 10.0,
    "RShoulder": 10.0,
    "LUArm": 10.0,
    "RUArm": 10.0,
    "LFArm": 10.0,
    "RFArm": 10.0,
    "LHand": 10.0,
    "RHand": 10.0,
    "LThigh": 10.0,
    "RThigh": 10.0,
    "LShin": 10.0,
    "RShin": 10.0,
    "LFoot": 10.0,
    "RFoot": 10.0,
    "Head": 10.0,
}

# Maximum calibrated IMU acceleration magnitude in validation.parquet [m/s^2].
# Ab and Neck use the nearest available IMUs: lower_back and head.
LINEAR_ACCELERATION_LIMITS: dict[str, float] = {
    "Hip": 29.583,
    "Ab": 29.583,
    "Chest": 17.942,
    "Neck": 21.024,
    "LShoulder": 27.288,
    "RShoulder": 21.575,
    "LUArm": 53.977,
    "RUArm": 45.548,
    "LFArm": 87.811,
    "RFArm": 110.759,
    "LHand": 106.266,
    "RHand": 112.624,
    "LThigh": 28.769,
    "RThigh": 50.415,
    "LShin": 41.114,
    "RShin": 63.353,
    "LFoot": 15,
    "RFoot": 15,
    "Head": 21.024,
}