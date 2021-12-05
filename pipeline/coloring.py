import numpy as np
from perlin_numpy import generate_perlin_noise_2d

import logging
logger = logging.getLogger("realms")

def colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs):
    if isinstance(mu, list):
        mu1, mu2 = mu
    else:
        mu1 = mu2 = mu
    if isinstance(sig, list):
        sig1, sig2 = sig
    else:
        sig1 = sig2 = sig
    gl = np.exp(-np.power(hmap - mu1, 2.) / (2 * np.power(sig1, 2.)))
    gl = np.where(hmap < mu1, gl, 0)
    gr = np.exp(-np.power(hmap - mu2, 2.) / (2 * np.power(sig2, 2.)))
    gr = np.where(hmap > mu2, gr, 0)
    ans = np.clip(np.where(((mu1 < hmap) & (hmap < mu2)), 1, gl+gr),0, 1)

    ps = np.random.random((hmap.shape))

    needs_coloring = ps < ans
    pnoise = generate_perlin_noise_2d(hmap.shape, [perlin_res,perlin_res])
    pnoise = np.digitize(pnoise, np.linspace(-0.6,0.6,len(color_diffs)-1))

    select_idx = needs_coloring * pnoise

    colors = [color+cdiff for cdiff in color_diffs]

    final = np.expand_dims(needs_coloring, -1)*np.take(colors, select_idx, axis=0)

#     if debug:
#         plt.figure(figsize=debug_img_size)
#         plt.imshow(needs_coloring)
#         plt.show()
#         plt.figure(figsize=debug_img_size)
#         plt.imshow(pnoise)
#         plt.show()
#         plt.figure(figsize=debug_img_size)
#         plt.imshow(final)
#         plt.show()
    
    return final

def overlap(base, overlay):
    return np.where(overlay.sum(axis=-1, keepdims=True)>0, overlay, base)

def run_coloring(color_functions, hmap):
    x = np.zeros((*hmap.shape, 3))
    for cfn in color_functions:
        y = cfn(hmap)
        x = overlap(x, y)
    return x

def inject_water_tile(colorqmap, landmask, color):
    for x in range(landmask.shape[0]):
        for y in range(landmask.shape[1]):
            if landmask[x, y] == 0:
                colorqmap[x,y,:] = color
                logger.info(f"Injected water pixel {color} at {x}, {y}.")
                return colorqmap

def dirt_1(hmap):
    mu, sig = [0.05, 0.3], [0.01, 0.2]
    perlin_res = 16
    color = np.array([100,  76,  76])
    color_diffs = [+20, +10, 0, -10, -20]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def dirt_2(hmap):
    mu, sig = [0.3, 0.45], 0.02
    perlin_res = 16
    color = np.array([100,  76,  76])
    color_diffs = [0, 5, 10, 15]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def stone_1(hmap):
    mu, sig = [-1.0, 0.9], 0.1
    perlin_res = 16
    color = np.array([128,  128,  128])
    color_diffs = [+20, +10, 0, -10, -20]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def stone_2(hmap):
    mu, sig = [0.6, 1.0], 0.05
    perlin_res = 16
    color = np.array([128,  128,  128])
    color_diffs = [-5, 0, 5, 10, 15]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def sand_1(hmap):
    mu, sig = [-0.1, 0.01], 0.03
    perlin_res = 16
    color = np.array([151, 149, 130])
    color_diffs = [-5, 0, 5, 10, 15]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def sand_2(hmap):
    mu, sig = [-0.1, -0.3], 0.05
    perlin_res = 16
    color = np.array([151, 149, 130])
    color_diffs = [-10, -20, -30]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def grass_1(hmap):
    mu, sig = [0.3, 0.4], 0.1
    perlin_res = 16
    color = np.array([80, 140, 80])
    color_diffs = [0, -5, -10, -15, -20, -30]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def grass_2(hmap):
    mu, sig = 0.4, 0.15
    perlin_res = 16
    color = np.array((60, 117, 55))
    color_diffs = [5, 0, -5, -10]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def dirt_3(hmap):
    mu, sig = [0.05, 0.3], [0.01, 0.2]
    perlin_res = 16
    color = np.array([61, 47, 20])
    color_diffs = [+20, +10, 0, -10, -20]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def snow_1(hmap):
    mu, sig = [0.3, 1.0], 0.05
    perlin_res = 16
    color = np.array([225, 225, 225])
    color_diffs = [-10, -5, 0, 5]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def moss_1(hmap):
    mu, sig = 0.4, 0.15
    perlin_res = 16
    color = np.array((124, 135, 70))
    color_diffs = [20, 0, -20]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def sand_3(hmap):
    # tropical sand
    mu, sig = [-0.5, 0.01], 0.03
    perlin_res = 16
    color = np.array([232, 197, 151])
    color_diffs = [-5, 0, 5, 10, 15]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def sand_4(hmap):
    # white sand
    mu, sig = [-0.05, 0.1], 0.03
    perlin_res = 16
    color = np.array([220, 220, 220])
    color_diffs = [-5, 0, 5, 10, 15]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def greenery(hmap):
    # lush greenery
    mu, sig = [0.1, 1.], 0.03
    perlin_res = 16
    color = np.array([150, 190, 160])
    color_diffs = [-10, -20, -25, -30, -40]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def grass_cliffs(hmap):
    # lush greenery
    mu, sig = 0.1, [0.02, 0.1]
    perlin_res = 16
    color = np.array([120, 160, 100])
    color_diffs = [-10, -20, -25, -30, -40]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def rocks(hmap):
    mu, sig = [0.001, 1.], [0.001, 0.1]
    perlin_res = 16
    color = np.array([140, 140, 100])
    color_diffs = [-5, 0, 5, 10, 15]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def water_vegetation(hmap):
    mu, sig = [0.01, 0.1], [0.01, 0.01]
    perlin_res = 16
    color = np.array([152, 158, 95])
    color_diffs = [-5, 0, 5, 10, 15]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def some_rocks(hmap):
    mu, sig = [0.7, 1.], [0.01, 0.01]
    perlin_res = 16
    color = np.array([140, 140, 140])
    color_diffs = [-10, -5, 0, 5, 10]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def dunes(hmap):
    mu, sig = [0.001, 1.], [0.001, 0.1]
    perlin_res = 16
    color = np.array([151, 149, 130])
    color_diffs = [0, 5, 10, 15, 15]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def snow_2(hmap):
    mu, sig = [0.01, 0.6], [0.001, 0.1]
    perlin_res = 16
    color = np.array([220, 232, 232])
    color_diffs = [0, 5, 10, 15, 15]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def tundra(hmap):
    mu, sig = [0.01, 0.2], [0.001, 0.1]
    perlin_res = 16
    color = np.array([4, 64, 46])
    color_diffs = [0, 5, 10, 15, 15]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

moderate = [
    stone_1,
    dirt_1,
    sand_1,
    stone_2,
    dirt_2,
    grass_1,
    grass_2,
]

cold = [
    stone_1,
    dirt_3,
    sand_1,
    stone_2,
    dirt_2,
    snow_1,
    moss_1,
]

snow = [
    stone_1,
    stone_2,
    snow_1,
    snow_2,
    tundra,
]

savanna = [
    stone_1,
    sand_1,
    sand_2,
    rocks,
    water_vegetation,
    some_rocks,
]

desert = [
    stone_1,
    sand_1,
    sand_2,
    dunes,
    some_rocks,
]
