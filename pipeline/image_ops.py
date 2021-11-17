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

def close_svg(drawing, rng=460, debug=False):
    """This function tries to find open ends of paths and
    connects the ends while going around the image borders.
    
    Paths of lenght < 4 will be ignored.
    
    Args:
        - drawing: the drawing containing paths from extractor.coast()
        - rng: distance of the edge wrt center
    
    Returns:
        (1) list of lines that can be used to close the paths
        (2) drawing with closed paths
    """
    LIMIT = 440

    def extend(x, y, bound, limit):
        """Extrapolates points lying at ‘limit‘."""
        assert y != 0
        assert x != 0
        if x < -limit:
            return [-bound, y/x*-bound]
        elif x > limit:
            return [bound, y/x*bound] 
        elif y < -limit:
            return [x/y*-bound, -bound]
        elif y > limit:
            return [x/y*bound, bound]
        else:
            raise ValueError("edge not within limits.")
            
    # Stage 1: find the first set of open paths and closed paths (islands)
    paths = drawing.contents[0].contents[0].contents

    arrays = []
    for shape_group in drawing.contents[0].contents:
        path = shape_group.contents[0]
        plen = len(path.points)
        split_array = [[path.points[x],path.points[x+1]] for x in range(0,plen,2)]
        
        # here we are injecting extra points
        if split_array[0]==split_array[-1]:
            # 'tis an island
            pass
        else:
            first, last = split_array[0], split_array[-1]
            # check coordinates on the far left:
            first = extend(first[0], first[1], rng, LIMIT)
            last = extend(last[0], last[1], rng, LIMIT)
            new_array = [first]
            new_array.extend(split_array)
            new_array.append(last)
            split_array = new_array
        
        np_array = np.array(split_array)
        if len(np_array>3):
            arrays.append(np_array)
        
    # Stage 2: start from the upper left and sort all end points anti-clockwise.
    begends = []
    begends_list = []
    for a in arrays:
        begends.append({
            "beg": a[0],
            "end": a[-1],
        })
        begends_list.append(a[0])
        begends_list.append(a[-1])
    
    lefts = [
        a for a in begends_list
        if -rng == a[0]
    ]
    lefts.sort(key=lambda x: -x[1])
    
    bottoms = [
        a for a in begends_list
        if -rng == a[1]
    ]
    bottoms.sort(key=lambda x: x[0])
    
    rights = [
        a for a in begends_list
        if rng == a[0]
    ]
    rights.sort(key=lambda x: x[1])
    
    tops = [
        a for a in begends_list
        if rng == a[1]
    ]
    tops.sort(key=lambda x: -x[0])
    
    # Stage 3: draw lines in order to connect the sorted endpoints.
    lines = []

    up_left = np.array([-rng, rng])
    bottom_left = np.array([-rng, -rng])
    bottom_right = np.array([rng, -rng])
    up_right = np.array([rng, rng])
    draw = False

    now = up_left
    # go down from the left side
    for co in lefts:
        if draw:
            lines.append([now, co])
            draw = False
        elif not draw:
            now = co
            draw = True

    if draw:
        lines.append([now, bottom_left])
        now = bottom_left
    # go right at the bottom
    for co in bottoms:
        if draw:
            lines.append([now, co])
            draw = False
        elif not draw:
            now = co
            draw = True

    if draw:
        lines.append([now, bottom_right])
        now = bottom_right
    # go up at the right
    for co in rights:
        if draw:
            lines.append([now, co])
            draw = False
        elif not draw:
            now = co
            draw = True

    if draw:
        lines.append([now, up_right])
        now = up_right
    # go left at the top
    for co in tops:
        if draw:
            lines.append([now, co])
            draw = False
        elif not draw:
            now = co
            draw = True

    if draw:
        lines.append([now, up_left])
    
    # Stage 4: merge open paths and split off closed paths.
    # In the end, everything should become a closed path.
    def island_check(arrays):
        islands = []
        nislands = []
        for a in arrays:
            first, last = a[0], a[-1]
            if (first==last).all():
                islands.append(a)
            else:
                nislands.append(a)
        return islands, nislands
    islands, arrays = island_check(arrays) 
    
    if debug:
        print(f"{len(islands)=}")

    # Add lines to arrays. Lines can be seen as very short paths.
    for line in lines:
        arrays.append(np.vstack(line))

    while len(arrays) > 0:
        donts = [] # bookkeep which arrays to skip at the end
        alen = len(arrays)
        for i in range(alen):
            a = arrays[i]
            f1, l1 = a[0], a[-1]
            for j in range(i+1, alen):
                b = arrays[j]
                f2, l2 = b[0], b[-1]
                if (l1==f2).all():
                    arrays[j] = np.vstack([a,b])
#                     print(f"connected ij {i,j}")
                    donts.append(i)
                    break
                elif (l2==f1).all():
                    arrays[j] = np.vstack([b,a])
#                     print(f"connected ji {i,j}")
                    donts.append(i)
                    break
                elif (f1==f2).all():
                    arrays[j] = np.vstack([np.flip(b,axis=0), a])
#                     print(f"connected _ji{j,i}")
                    donts.append(i)
                    break
                elif (l1==l2).all():
                    arrays[j] = np.vstack([a, np.flip(b, axis=0)])
#                     print(f"connected i_j{i,j}")
                    donts.append(i)
                    break

        arrays = [a for (i,a) in enumerate(arrays) if i not in donts]

        new_islands, arrays = island_check(arrays)
        islands.extend(new_islands)

    if debug:
        print(f"{len(islands)=}")
        print(f"{len(arrays)=}")

        al = np.vstack(islands)
        plt.figure(figsize=(20,20))
        plt.scatter(al[:,0], -al[:,1], s=4)
        plt.xlim(-rng-100,rng+100)
        plt.ylim(-rng-100,rng+100)
        plt.show()
        
    # Stage 5: scale and cast to PIL.Image
    for i in range(len(islands)):
        islands[i] = (islands[i]*0.4+200)*2
    
    base = PIL.Image.new("L", (800,800), 0)
    drawer = PIL.ImageDraw.Draw(base)

    for island in islands:
        drawer.polygon(list(island.flatten()), fill=1)

    data = np.asarray(base)    
    return data