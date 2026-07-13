import numpy as np

# SensorSuit -> NatNet world conversion kept for diagnostics/tools that compare
# both world frames. The checker now uses NATNET_WORLD_GRAVITY directly.
WNN_R_WSS = np.array([[1.0, 0.0,  0.0],
                      [0.0, 0.0,  1.0],
                      [0.0, -1.0, 0.0]])

# NatNet world gravity reference used by the gravity checker.
NATNET_WORLD_GRAVITY = np.array([0.0, 9.81, 0.0])
