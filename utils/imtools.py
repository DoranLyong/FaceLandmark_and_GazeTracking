""" (ref) https://github.com/DoranLyong/Awesome-Human-Gaze-Points/blob/main/tutorial01/gazeHeatmap_plot.py
    (ref) https://github.com/DoranLyong/Awesome-Human-Gaze-Points/blob/main/tutorial01/utils.py
"""
import numpy as np 







def gaussian_kernel(x, sx, y=None, sy=None) -> np.ndarray:


    """Returns an array of np arrays (a matrix) containing values between
        1 and 0 in a 2D Gaussian distribution
    
    [ arguments ] 
    x		-- width in pixels
    sx		-- width standard deviation
    [ keyword argments ] 
    y		-- height in pixels (default = x)
    sy		-- height standard deviation (default = sx)
    """

    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx

    # centers
    xo = x / 2
    yo = y / 2

    # matrix of zeros
    M = np.zeros([y, x], dtype=float)  # (ref) https://numpy.org/doc/stable/reference/generated/numpy.zeros.html

    # Gaussian matrix
    # (ref) https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy/29731818#29731818
    # (ref) https://gist.github.com/thomasaarholt/267ec4fff40ca9dff1106490ea3b7567
    from scipy import signal
    
    gkern1d = signal.gaussian(x, std=sx)
    M = np.outer(gkern1d, gkern1d)  # (ref) https://numpy.org/doc/stable/reference/generated/numpy.outer.html

    return M


