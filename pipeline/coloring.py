"""
This file holds the logic for all things coloring.
Author: rvorias
"""

import numpy as np
from perlin_numpy import generate_perlin_noise_2d

import logging
logger = logging.getLogger("realms")

WATER_COLORS = [
    np.array([74., 134., 168.]),
    np.array([18., 59., 115.]),
    np.array([66., 109., 138.])
]

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

    colors = [np.clip(color+cdiff, 0, 255) for cdiff in color_diffs]

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

DEFAULT_DIFFS = [-10, -5, 0, 5, 10]

def deep_stone(hmap):
    mu, sig = [-1.1, 1.1], 0.1
    perlin_res = 16
    color = np.array([113,  113,  113])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def deep_sea(hmap):
    mu, sig = [-1.1, 0.], [0.1, 0.001]
    perlin_res = 16
    color = np.array([188,  176,  133])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def deep_sea_2(hmap):
    mu, sig = [-1.1, 0.], [0.1, 0.001]
    perlin_res = 16
    color = np.array([50,  176,  133])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def shallow_sea(hmap):
    mu, sig = [-0.1, 0.], [0.05, 0.001]
    perlin_res = 16
    color = np.array([205,  198,  135])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def soil_brown_light(hmap):
    mu, sig = [0.01, 0.4], [0.001, 0.05]
    perlin_res = 16
    color = np.array([95,  81,  71])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def soil_brown_dark(hmap):
    mu, sig = [0.01, 0.1], [0.001, 0.05]
    perlin_res = 16
    color = np.array([61,  55,  50])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def desert_sand_1(hmap):
    mu, sig = [0.01, 0.3], [0.001, 0.05]
    perlin_res = 16
    color = np.array([227,  148,  105])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def desert_sand_2(hmap):
    mu, sig = [0.1, 0.2], [0.01, 0.05]
    perlin_res = 16
    color = np.array([168,  100,  77])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def desert_sand_3(hmap):
    mu, sig = [0.1, 0.3], [0.01, 0.05]
    perlin_res = 16
    color = np.array([243,  176,  89])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def desert_green(hmap):
    mu, sig = [0.01, 0.05], [0.001, 0.01]
    perlin_res = 16
    color = np.array([77,  115,  59])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def savannah_green(hmap):
    mu, sig = [0.01, 0.1], [0.001, 0.01]
    perlin_res = 16
    color = np.array([77,  115,  59])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def grass_low(hmap):
    mu, sig = [0.01, 0.3], [0.001, 0.1]
    perlin_res = 16
    color = np.array([125,  181,  53])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def grass_high(hmap):
    mu, sig = [0.3, 0.4], [0.01, 0.1]
    perlin_res = 16
    color = np.array([98,  133,  42])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def forest_1(hmap):
    mu, sig = [0.01, 0.3], [0.001, 0.1]
    perlin_res = 16
    color = np.array([98,  133,  42])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def forest_2(hmap):
    mu, sig = [0.3, 0.4], [0.05, 0.1]
    perlin_res = 16
    color = np.array([76,  103,  31])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def jungle_low(hmap):
    mu, sig = [0.1, 0.3], [0.05, 0.1]
    perlin_res = 16
    color = np.array([59,  125,  53])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def jungle_med(hmap):
    mu, sig = [0.3, 0.6], [0.05, 0.1]
    perlin_res = 16
    color = np.array([76,  139,  70])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def jungle_high(hmap):
    mu, sig = [0.6, 0.8], [0.05, 0.2]
    perlin_res = 16
    color = np.array([111,  143,  108])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def stone_high(hmap):
    mu, sig = [0.8, 1.1], [0.05, 0.1]
    perlin_res = 16
    color = np.array([133,  133,  133])
    color_diffs = DEFAULT_DIFFS
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def taiga_stone_1(hmap):
    mu, sig = [-1.0, 0.9], 0.1
    perlin_res = 16
    color = np.array([128,  128,  128])
    color_diffs = [+20, +10, 0, -10, -20]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def taiga_dirt_3(hmap):
    mu, sig = [0.05, 0.3], [0.01, 0.2]
    perlin_res = 16
    color = np.array([61, 47, 20])
    color_diffs = [+20, +10, 0, -10, -20]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def taiga_sand_1(hmap):
    mu, sig = [-0.1, 0.01], 0.03
    perlin_res = 16
    color = np.array([151, 149, 130])
    color_diffs = [-5, 0, 5, 10, 15]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def taiga_stone_2(hmap):
    mu, sig = [0.6, 1.0], 0.05
    perlin_res = 16
    color = np.array([128,  128,  128])
    color_diffs = [-5, 0, 5, 10, 15]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def taiga_dirt_2(hmap):
    mu, sig = [0.3, 0.45], 0.02
    perlin_res = 16
    color = np.array([100,  76,  76])
    color_diffs = [0, 5, 10, 15]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def taiga_snow_1(hmap):
    mu, sig = [0.3, 1.1], 0.05
    perlin_res = 16
    color = np.array([225, 225, 225])
    color_diffs = [-10, -5, 0, 5]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

def taiga_moss_1(hmap):
    mu, sig = 0.4, 0.15
    perlin_res = 16
    color = np.array((124, 135, 70))
    color_diffs = [20, 0, -20]
    return colorize_perlin(hmap, mu, sig, perlin_res, color, color_diffs)

# biomes = {
#     # a green forest all year round
#     "forest": [
#         deep_stone,
#         deep_sea,
#         soil_brown_light,
#         forest_1,
#         forest_2,
#         stone_high
#     ],
#     # normal soil with lush grass at normal level and tough grass
#     # at higher levels
#     "grassland": [
#         deep_stone,
#         deep_sea_2,
#         soil_brown_light,
#         grass_low,
#         grass_high,
#         stone_high
#     ],
#     "jungle": [
#         deep_stone,
#         deep_sea,
#         soil_brown_dark, 
#         stone_high,
#         forest_1,
#         jungle_low,
#         jungle_high
#     ],
#     "savannah": [
#         deep_stone,
#         deep_sea,
#         soil_brown_light,
#         savannah_green
#     ],
#     "desert": [
#         deep_stone,
#         deep_sea,
#         desert_sand_1,
#         desert_sand_2,
#         desert_sand_3,
#         desert_green
#     ],
#     "taiga": [
#         taiga_stone_1,
#         taiga_dirt_3,
#         taiga_sand_1,
#         taiga_stone_2,
#         taiga_dirt_2,
#         taiga_snow_1,
#         taiga_moss_1,
#     ],
#     "tundra": [
#         deep_stone,
#         deep_sea,
#         soil_brown_light,
#     ],
#     "arctic": [
#         deep_stone,
#         deep_sea
#     ]
# }

# coloring from file
def color_from_json(hmap, biome):
    import json
    with open("resources/colors.json", "r") as file:
        data = json.load(file)
    cmap = np.zeros((*hmap.shape, 3))
    for i, color in enumerate(data[biome]["colors"]):
        layer =  colorize_perlin(
            hmap,
            data[biome]["mus"][i],
            data[biome]["sigmas"][i],
            16,
            np.array(color[:3]),
            [-10, -5, 0, 5, 10]
        )
        cmap = np.where(layer.sum(axis=-1, keepdims=True)>0, layer, cmap)
    return cmap

biomes = [
    "grassland",
    "forest",
    "savannah",
    "desert",
    "lava",
    "taiga",
    "tundra",
    "ice"
]
# order does not matter between pairs
biome_pairs = [
    ["grassland", "forest"],
    ["grassland", "savannah"],
    ["grassland", "desert"],
    ["grassland", "taiga"],
    ["jungle",    "forest"],
    ["jungle",    "savannah"],
    ["savannah",  "desert"],
    ["forest",    "taiga"],
    ["taiga",     "tundra"],
    ["taiga",     "ice"],
    ["tundra",    "ice"],
    ["ice",       "lava"],
    ["desert",    "lava"]
]