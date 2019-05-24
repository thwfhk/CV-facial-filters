import cv2
import numpy as np
from pykalman import KalmanFilter


class MyKalman:
    # dt 应该用两帧间差值
    def __init__(self, dt=1.0):
        self.dt = dt
        self.kf = KalmanFilter(transition_matrices=np.array([[1, 0, 0, self.dt, 0, 0],
                                                             [0, 1, 0, 0, self.dt, 0],
                                                             [0, 0, 1, 0, 0, self.dt],
                                                             [0, 0, 0, 1, 0, 0],
                                                             [0, 0, 0, 0, 1, 0],
                                                             [0, 0, 0, 0, 0, 1],
                                                             ]
                                                            ),
                               observation_matrices=np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]),
                               transition_covariance=0.0 * np.eye(6))
        # transition_matrices：公式中的A
        # observation_matrices：公式中的H
        # transition_covariance：公式中的Q
        self.filtered_state_means0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.filtered_state_covariances0 = np.eye(6)

    def update(self, x, y, z):
        current_measurement = np.array([np.float32(x), np.float32(y), np.float32(z)])

        filtered_state_means, filtered_state_covariances = (self.kf.filter_update(self.filtered_state_means0,
                                                                                  self.filtered_state_covariances0,
                                                                                  current_measurement))
        cpx, cpy, cpz = filtered_state_means[0], filtered_state_means[1], filtered_state_means[2]
        self.filtered_state_means0, self.filtered_state_covariances0 = filtered_state_means, filtered_state_covariances
        return cpx, cpy, cpz

