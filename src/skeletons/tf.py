import numpy as np

# World conversion used by the gravity pipeline.  SensorSuit world gravity is
# +Z; NatNet world up is +Y.  Gravity does not determine yaw about the vertical.
WNN_R_WSS = np.array([[1.0, 0.0,  0.0],
                      [0.0, 0.0,  1.0],
                      [0.0, -1.0, 0.0]])
