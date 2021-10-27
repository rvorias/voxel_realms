import numpy as np
import matplotlib.pyplot as plt
import svglib
from svglib.svglib import svg2rlg
import io
from reportlab.graphics import renderPDF, renderPM
from skimage import data, filters, color, morphology
from skimage.segmentation import flood, flood_fill

import PIL.Image
import matplotlib.pyplot as plt

from reportlab.graphics.shapes import *

import logging

def flood_image(image, center_list, padding):
    """Assumes an uncropped image and uncropped centers"""
    data = np.asarray(image)[padding:-padding,padding:-padding,0]

    for center in center_list:
        cropped_center = (int(center[0]-padding), int(center[1]-padding))
        data = flood_fill(data, cropped_center, 0)

    return data