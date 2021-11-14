import numpy as np
import matplotlib.pyplot as plt
import svglib
from svglib.svglib import svg2rlg
import io
from reportlab.graphics import renderPDF, renderPM
from skimage import data, filters, color, morphology
from skimage.segmentation import flood, flood_fill

from svg_extraction import get_island_coordinates, get_island_centers

import PIL.Image
import PIL.ImageDraw
import matplotlib.pyplot as plt

from reportlab.graphics.shapes import *

import logging
logger = logging.getLogger("realms")

def flood_image(image, center_list, padding, coast_coordinates=None, debug=False):
    """Assumes an uncropped image and uncropped centers"""
    if isinstance(image, np.ndarray):
        data = image
    else:
        data = np.asarray(image)[padding:-padding,padding:-padding,0]
 
    if debug:
        plt.figure(figsize=(30,30))
        plt.title("starting flooding image debug")
        plt.imshow(data)
        plt.show()

    for center in center_list:
        if coast_coordinates is not None:
            # check if too close to a coast
            target = np.array([center[0], center[1]])
            dis = np.sqrt(np.sum((coast_coordinates-target)**2, axis=1)).min()
            if dis < 10:
                continue

        cropped_center = (int(center[0]-padding), int(center[1]-padding))
        # don't fill if you're alread on a black spot (!or a stroke!)
        if data[cropped_center[0], cropped_center[1]] > 0:
            data = flood_fill(data, cropped_center, 0)
            if debug:
                logger.debug(f"flooding at {cropped_center}")
                plt.figure(figsize=(30,30))
                plt.imshow(data)
                plt.plot(cropped_center[1], cropped_center[0], 'ro')
                plt.show()
    return data

def flood_islands(image, drawing, padding, debug=False):
    """Fills in one island based off of its coast coordinates."""
    base = PIL.Image.new("L", image.shape, 1)
    drawer = PIL.ImageDraw.Draw(base)

    for group in drawing.contents[0].contents:
        points = group.contents[0].points
        if points[0]==points[-2] and points[1]==points[-1]: # if closed loop
            points = [(p*0.4+200)*2-padding for p in points]
            drawer.polygon(points, fill=0)

    data = np.asarray(base)
    data *= image
    data = np.clip(data, 0, 255)

    return data

def greedy_find_clusters(coordinates, k=4, max_radius=15, debug=False):
    """Finds clusters in a greedy fashion."""
    clusters = []
    
    if debug:
        plt.figure(figsize=(10,10))
        plt.scatter(coordinates[:,0], coordinates[:,1], s=1)
    
    while coordinates.shape[0] > 10:
        sel = coordinates[0]
        dis = np.linalg.norm(coordinates-sel, axis=1)
        close_idx = np.argwhere(dis<max_radius)
        if len(close_idx) < k:
            coordinates = coordinates[1:]
        else:
            clusters.append(coordinates[close_idx][:,0])
            coordinates = np.delete(coordinates, close_idx, axis=0)
    
    # find centers of clusters
    centroids = []
    for cluster in clusters:
        centroid = cluster.mean(axis=0)
        centroids.append(centroid)
        if debug:
            plt.scatter(centroid[0], centroid[1])
    if debug:
        plt.show()
    return centroids