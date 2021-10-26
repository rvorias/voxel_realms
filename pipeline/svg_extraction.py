"""
Author: rvorias
"""

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

class SVGExtractor:
    def __init__(self, drawing_path, scale=2):
        self.drawing_path = drawing_path
        self.scale = scale
        self.load_drawing()

    def coast(self):
    	self.load_drawing()
    	return self.get_cls(Path, "strokeWidth", 4.0)

    def cities(self):
    	self.load_drawing()
    	return self.get_cls(Circle)

    def height(self):
        self.load_drawing()
        return self.get_cls(Line)

    def rivers(self):
    	self.load_drawing()
    	return self.get_cls(Path, "strokeWidth", 2.0)
    
    def load_drawing(self):
        self.drawing = svg2rlg(self.drawing_path)
        self.drawing.scale(self.scale, self.scale)
        self.drawing.width *= self.scale
        self.drawing.height *= self.scale

    def get_cls(self, svgclass, key=None, value=None):
        self.load_drawing()
        shape_group = self.drawing.contents[0]
        contents = shape_group.contents
        new_contents = []
        for shape in contents:
            if isinstance(shape, Group):
                obj = shape.contents[0]
                if isinstance(obj, svgclass):
                    if key:
                        if obj.__dict__[key] == value:
                            obj.__dict__[key] = 1.
                            new_contents.append(shape)
                    else:
                        new_contents.append(shape)
            elif isinstance(shape, svgclass):
                new_contents.append(shape)

        shape_group.contents = new_contents
        self.drawing.contents = [shape_group]
        return self.drawing
    
    def get_img(self):
        buffer = io.BytesIO()
        renderPM.drawToFile(self.drawing, buffer, fmt="PNG")
        img = PIL.Image.open(buffer)
        return img
    
    def show(self):
        plt.figure(figsize=(10,10))
        plt.imshow(self.get_img())

def get_city_coordinates(drawing, padding=10, scaling=1.57):
    centers = []
    for circle in drawing.contents[0].contents:
        centers.append((
            int(circle.cy/scaling+drawing.height//2)-padding,
            int(circle.cx/scaling+drawing.width//2)-padding 
        ))
    return centers

def get_island_coordinates(drawing, padding=10, scaling=1.57):
    centers = []
    for group in drawing.contents[0].contents:
        points = group.contents[0].points
        if points[0]==points[-2] and points[1]==points[-1]: # if closed loop
            mx, my = 0, 0
            n_points = len(points[:-2])
            for i in range(0,n_points,2):
                mx += points[i]
                my += points[i+1]
            mx /= n_points
            my /= n_points
            centers.append((
                int(my*scaling+drawing.height//2)-padding,
                int(mx*scaling+drawing.width//2)-padding
            ))
    return centers






























