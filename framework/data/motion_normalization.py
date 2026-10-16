"""Single source of truth for LiveFace motion scaling.

The training dataset historically constructed MinMaxScaler objects from these
fixed ranges, while generation loaded unrelated pickle files. Keeping the
formula here prevents silent train/evaluation normalization drift.
"""

import numpy as np


EXP_MIN = -3.0
EXP_MAX = 3.0
JAW_MIN = -0.1
JAW_MAX = 0.5


def normalize_motion(exp: np.ndarray, jaw: np.ndarray):
    exp_norm = (np.asarray(exp) - EXP_MIN) / (EXP_MAX - EXP_MIN)
    jaw_norm = (np.asarray(jaw) - JAW_MIN) / (JAW_MAX - JAW_MIN)
    return exp_norm, jaw_norm


def denormalize_motion(exp: np.ndarray, jaw: np.ndarray):
    exp_value = np.asarray(exp) * (EXP_MAX - EXP_MIN) + EXP_MIN
    jaw_value = np.asarray(jaw) * (JAW_MAX - JAW_MIN) + JAW_MIN
    return exp_value, jaw_value
