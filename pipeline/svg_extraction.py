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
    
    def show(self, size=(10,10)):
        plt.figure(figsize=size)
        plt.imshow(self.get_img())
        plt.show()

def get_city_coordinates(drawing, scaling=2):
    """Calculated in uncropped coordinates"""
    centers = []
    for circle in drawing.contents[0].contents: 
        centers.append((
            int(circle.cy*.4+200)*scaling,
            int(circle.cx*.4+200)*scaling 
        ))
    return centers

def get_island_coordinates(drawing, scaling=2):
    """Calculated in uncropped coordinates"""
    centers = []
    for group in drawing.contents[0].contents:
        points = group.contents[0].points
        if points[0]==points[-2] and points[1]==points[-1]: # if closed loop
            mx, my = 0, 0
            n_points = len(points[:-2])
            for i in range(0,n_points,2):
                mx += points[i]
                my += points[i+1]
            mx /= n_points//2
            my /= n_points//2
            centers.append((
                int((my*.4+200)*scaling),
                int((mx*.4+200)*scaling)
            ))
    return centers

def get_orthogonal_samples(path, scaling=2):
    """This function takes a path and iterates on its points.
    It draws a line between two subsequent points (a,b) and finds the line that
    is orthogonal and goes through the center (o1,o2). It then returns two points
    on either side of the original line.
            o1
            |
    a ===== c ===== b
            |
            o2
    """
    DIS_FROM_CENTER = 5
    
    samples1 = []
    samples2 = []
    for i in range(0, len(path.points)-2, 2):
        a = np.array([path.points[i], path.points[i+1]])
        b = np.array([path.points[i+2], path.points[i+3]])
        
        if a[1] == b[1]:
            print("y aligned")
        elif a[0] == b[0]:
            print("x aligned")
        else:
            m_inv = -(b[0]-a[0])/(b[1]-a[1])
            center = (a+b)/2
            c = center[1] - m_inv*center[0]
            other_1 = np.array([center[0]+1, m_inv*(center[0]+1)+c])
            other_2 = np.array([center[0]-1, m_inv*(center[0]-1)+c])
            dis = np.linalg.norm(other_1-center)
            other_1 = np.array([center[0]+DIS_FROM_CENTER/dis, m_inv*(center[0]+DIS_FROM_CENTER/dis)+c])
            other_2 = np.array([center[0]-DIS_FROM_CENTER/dis, m_inv*(center[0]-DIS_FROM_CENTER/dis)+c])
            samples1.append((other_1*0.4+200)*scaling)
            samples2.append((other_2*0.4+200)*scaling)
    return samples1, samples2





























