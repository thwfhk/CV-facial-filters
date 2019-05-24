import cv2
import numpy as np
from pykalman import KalmanFilter


class MyKalman:
    def __init__(self, dt=1.0):
        self.dt = dt
        self.kf = KalmanFilter(transition_matrices=np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]),
                               observation_matrices=np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
                               transition_covariance=0.05 * np.eye(4))
        # transition_matrices：公式中的A
        # observation_matrices：公式中的H
        # transition_covariance：公式中的Q
        self.filtered_state_means0 = np.array([0.0, 0.0, 0.0, 0.0])
        self.filtered_state_covariances0 = np.eye(4)

    def update(self, x, y):
        current_measurement = np.array([np.float32(x), np.float32(y)])

        filtered_state_means, filtered_state_covariances = (self.kf.filter_update(self.filtered_state_means0,
                                                                                  self.filtered_state_covariances0,
                                                                                  current_measurement))
        cpx, cpy = filtered_state_means[0], filtered_state_means[1]
        self.filtered_state_means0, self.filtered_state_covariances0 = filtered_state_means, filtered_state_covariances
        return cpx, cpy