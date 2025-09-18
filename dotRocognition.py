import cv2
import numpy as np
from utils import *

def threshold_brightness(gray, thresh=200):
    """
    Zero out all pixels below threshold brightness.

    Parameters
    ----------
    gray : 2D uint8 array
        Grayscale frame.
    thresh : int
        Brightness threshold [0..255].

    Returns
    -------
    mask : 2D uint8 array
        Binary mask (0 or 255).
    filtered : 2D uint8 array
        Grayscale frame with pixels < thresh set to 0.
    """
    if gray.ndim != 2:
        raise ValueError("Input must be a single-channel grayscale image")
    if gray.dtype != np.uint8:
        g = cv2.normalize(gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        g = gray