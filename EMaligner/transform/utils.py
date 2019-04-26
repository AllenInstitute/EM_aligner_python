import numpy as np


class AlignerTransformException(Exception):
    """Exception raised when there is a \
            problem creating a mesh lens correction"""
    pass


def aff_matrix(theta, offs=None):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    if offs is None:
        return R
    M = np.eye(3)
    M[0:2, 0:2] = R
    M[0, 2] = offs[0]
    M[1, 2] = offs[1]
    return M


def rotate(xy, theta):
    R = aff_matrix(theta)
    f = R.dot(xy.transpose()).transpose()
    return f
