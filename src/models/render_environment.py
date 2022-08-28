import functools

import matplotlib
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

FLY_MARKER_COLOR = 'firebrick'

FLY_MARKER_SIZE = 5


def render_odor_plume_frame_no_fly(plume_frame: np.ndarray, plot_axis: plt.Axes) -> plt.Axes:
    return plot_axis.imshow(X=plume_frame.T, cmap='gray', vmin=0, vmax=1, origin='lower').axes


def render_fly(position: np.ndarray, orientation: float, plot_axis: plt.Axes):
    arrow_marker, scale = gen_arrow_head_marker(orientation)
    plot_axis.scatter(position[0], position[1], marker=arrow_marker, s=(FLY_MARKER_SIZE * scale) ** 2,
                      c=FLY_MARKER_COLOR)
    return plot_axis



def gen_arrow_head_marker(rot):
    """
    https://stackoverflow.com/questions/23345565/is-it-possible-to-control-matplotlib-marker-orientation
    """
    """generate a marker to plot with matplotlib scatter, plot, ...

    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers

    rot=0: positive x direction
    Parameters
    ----------
    rot : float
        rotation in degree
        0 is positive x direction

    Returns
    -------
    arrow_head_marker : Path
        use this path for marker argument of plt.scatter
    scale : float
        multiply a argument of plt.scatter with this factor got get markers
        with the same size independent of their rotation.
        Paths are autoscaled to a box of size -1 <= x, y <= 1 by plt.scatter
    """
    arr = np.array([[.1, .3], [.1, -.3], [1, 0], [.1, .3]])  # arrow shape
    rot_mat = np.array([
        [np.cos(rot), np.sin(rot)],
        [-np.sin(rot), np.cos(rot)]
    ])
    arr = np.matmul(arr, rot_mat)  # rotates the arrow

    # scale
    x0 = np.amin(arr[:, 0])
    x1 = np.amax(arr[:, 0])
    y0 = np.amin(arr[:, 1])
    y1 = np.amax(arr[:, 1])
    scale = np.amax(np.abs([x0, x1, y0, y1]))
    codes = [mpl.path.Path.MOVETO, mpl.path.Path.LINETO, mpl.path.Path.LINETO, mpl.path.Path.CLOSEPOLY]
    arrow_head_marker = mpl.path.Path(arr, codes)
    return arrow_head_marker, scale
