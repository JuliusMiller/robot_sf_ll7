import numba

from robot_sf.util.types import Vec2D


@numba.njit(fastmath=True)
def euclid_dist(v1: Vec2D, v2: Vec2D) -> float:
    """
    Compute the Euclidean distance between two 2D vectors.

    Uses the formula: sqrt((x1 - x2)^2 + (y1 - y2)^2).

    Parameters
    ----------
    v1 : Vec2D
        The first 2D vector (x, y).
    v2 : Vec2D
        The second 2D vector (x, y).

    Returns
    -------
    float
        The Euclidean distance between v1 and v2.
    """
    return ((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2) ** 0.5
