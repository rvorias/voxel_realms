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

def flood_image(image, center_lists, padding):
	data = np.asarray(image)[padding:-padding,padding:-padding,0]

	for center_list in center_lists:
		for center in center_list:
			data = flood_fill(data, center, 0)

	return data