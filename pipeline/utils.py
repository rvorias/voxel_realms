import sys
sys.path.append("terrain-erosion-3-ways/")
from river_network import *

import numpy as np
from random import random, Random
from math import cos, sin, floor, sqrt, pi, ceil
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger("realms")

class Step:
    """This class is used as a wrapper for pipeline steps."""
    def __init__(self, text, realm_number):
        self.text = text
        self.realm_number = realm_number
    def __enter__(self):
        logger.info(self.text)
    def __exit__(self ,type, value, traceback):
        if type != None:
            with open(f"output/errors/{self.realm_number}.txt", "w") as f:
                f.write(value)
        logger.info("    \---DONE")

def imshow(image, title=None):
    plt.figure(figsize=(10,10))
    plt.title(title)
    plt.imshow(image)
    plt.show()

def norm(x, n=None):
    """norm `x` by `n`"""
    if n is None:
        n = x
    return (x - n.min()) / (n.max() - n.min())

def euclidean_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy

def poisson_disc_samples(width, height, r, k=5, distance=euclidean_distance, random=random):
    """
    Optimized poisson disc sampling.
    Credits to https://github.com/emulbreh/bridson
    """
    
    tau = 2 * pi
    cellsize = r / sqrt(2)

    grid_width = int(ceil(width / cellsize))
    grid_height = int(ceil(height / cellsize))
    grid = [None] * (grid_width * grid_height)

    def grid_coords(p):
        return int(floor(p[0] / cellsize)), int(floor(p[1] / cellsize))

    def fits(p, gx, gy):
        yrange = list(range(max(gy - 2, 0), min(gy + 3, grid_height)))
        sq_r = r*r
        for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
            for y in yrange:
                g = grid[x + y * grid_width]
                if g is None:
                    continue
                if distance(p, g) <= sq_r:
                    return False
        return True

    p = width * random(), height * random()
    queue = [p]
    grid_x, grid_y = grid_coords(p)
    grid[grid_x + grid_y * grid_width] = p

    while queue:
        qi = int(random() * len(queue))
        qx, qy = queue[qi]
        queue[qi] = queue[-1]
        queue.pop()
        for _ in range(k):
            alpha = tau * random()
            d = r * sqrt(3 * random() + 1)
            px = qx + d * cos(alpha)
            py = qy + d * sin(alpha)
            if not (0 <= px < width and 0 <= py < height):
                continue
            p = (px, py)
            grid_x, grid_y = grid_coords(p)
            if not fits(p, grid_x, grid_y):
                continue
            queue.append(p)
            grid[grid_x + grid_y * grid_width] = p
    p = [p for p in grid if p is not None]
    return np.asarray(p)

def filter_within_bounds(coordinates, width, height, svgpad):
    """This operates on uncropped coordinates, that is why we argpass svgpad"""
    logging.debug(f"filtering within: [{svgpad}, {width-svgpad}[, [{svgpad}, {height-svgpad}[")
    filtered_centers = []
    for co in coordinates:
        if svgpad <= co[1] < width-svgpad and svgpad <= co[0] < height-svgpad:
            filtered_centers.append(co)
        else:
            logging.warning(f"out of bounds coordinate!: {co}")
    return filtered_centers

def generate_terrain(
    mask,
    disc_radius=4.,
    max_delta=0.05,
    river_downcutting_constant=5.,
    directional_inertia=0.9,
    default_water_level=1.0,
    evaporation_rate=0.1,
    coastal_dropoff=50., # high: very small slope towards sea, low: abrupt change to sea
):
    """
    Modified version of https://github.com/dandrino/terrain-erosion-3-ways
    Will Largely take in parameters from the config file.
    """
    dim = mask.shape[0]
    shape = (dim,) * 2
    print('  ...initial terrain shape')
    land_mask = mask > 0
    coastal_dropoff = np.tanh(util.dist_to_mask(land_mask) / coastal_dropoff) * land_mask
    mountain_shapes = util.fbm(shape, -2, lower=2.0, upper=np.inf)
    initial_height = ( 
      (util.gaussian_blur(np.maximum(mountain_shapes - 0.40, 0.0), sigma=5.0) 
        + 0.1) * coastal_dropoff)
    deltas = util.normalize(np.abs(util.gaussian_gradient(initial_height))) 

    print('  ...sampling points')
#     points = util.poisson_disc_sampling(shape, disc_radius)
    prng = Random()
    prng.seed(42)
    points = poisson_disc_samples(*shape, r=disc_radius, random=prng.random)
    coords = np.floor(points).astype(int)

    print('  ...delaunay triangulation')
    tri = sp.spatial.Delaunay(points)
    (indices, indptr) = tri.vertex_neighbor_vertices
    neighbors = [indptr[indices[k]:indices[k + 1]] for k in range(len(points))]
    points_land = land_mask[coords[:, 0], coords[:, 1]]
    points_deltas = deltas[coords[:, 0], coords[:, 1]]

    print('  ...initial height map')
    points_height = compute_height(points, neighbors, points_deltas)

    print('  ...river network')
    (upstream, downstream, volume) = compute_river_network(
      points, neighbors, points_height, points_land,
      directional_inertia, default_water_level, evaporation_rate)

    print('  ...final terrain height')
    new_height = compute_final_height(
      points, neighbors, points_deltas, volume, upstream, 
      max_delta, river_downcutting_constant)
    return render_triangulation(shape, tri, new_height)

def get_wind_direction(direction):
    """Expects radians"""
    if direction > np.pi*7/16 and direction < np.pi*7/16:
        return "W"
    if direction > np.pi*5/16:
        return "NW"
    if direction < -np.pi*5/16:
        return "SW"
    if direction > np.pi*3/16:
        return "N"
    if direction < -np.pi*3/16:
        return "S"
    if direction > np.pi*1/16:
        return "NE"
    if direction < -np.pi*1/16:
        return "SE"
    return "E" 
