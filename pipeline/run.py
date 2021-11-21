"""
Author: rvorias
"""

import sys
import click
from omegaconf import OmegaConf

import numpy as np
import random as rand
from random import choice

import matplotlib.pyplot as plt
import PIL
import PIL.ImageOps
import skimage

import json

import logging
logger = logging.getLogger("realms")

sys.path.append("terrain-erosion-3-ways/")
from river_network import *

sys.path.append("pipeline")
from svg_extraction import SVGExtractor, get_heightline_centers
from image_ops import close_svg
from utils import *

from coloring import inject_water_tile, run_coloring
from coloring import cold, moderate, savanna, desert, snow


def run_pipeline(realm_path, config, debug=False):
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # set some often-used parameters
    svgp = config.svg.padding
    debug_img_size = (10,10)
    
    realm_number = int(realm_path.split("/")[-1][:-4])
    logging.info(f"Processing realm number: {realm_number}")
    np.random.seed(realm_number)
    rand.seed(realm_number)
    
    with step("Setting up extractor"):
        extractor = SVGExtractor(realm_path, scale=config.svg.scaling)
        if debug:
            extractor.show(debug_img_size)
        
    with step("Extracting coast"):
        coast_drawing = extractor.coast()
        
    with step("Extracting heightlines"):
        heightline_drawing = extractor.height()

    #############################################
    # MASKING
    #############################################
    
    with step("Starting ground-sea mask logic"):
        # uses a fixed padding of 32
        mask = close_svg(coast_drawing, debug=debug)

        # check if we need to flip
        centers = get_heightline_centers(heightline_drawing)
        sum = _sum = 0
        _mask = (mask-1)//255
        for center in centers:
            sum += mask[int(center[0]), int(center[1])]
            _sum += _mask[int(center[0]), int(center[1])]
        if _sum > sum:
            mask = _mask

        pad = 32
        h,w = mask.shape
        for i in range(pad):
            mask[:,i] = mask[:,pad]
            mask[:,-i] = mask[:,w-pad]
            mask[i,:] = mask[pad,:]
            mask[-i,:] = mask[h-pad,:]

        if debug:
            logger.debug(f"mask_shape: {mask.shape}")
            plt.figure(figsize=debug_img_size)
            plt.title("land-sea mask")
            plt.imshow(mask)
            plt.show()

    #------------------------------------------------------------------------------
    with step("----Extracting rivers"):
        extractor.rivers()
        rivers = np.asarray(extractor.get_img()) # rivers is now [0,255]

        rivers = skimage.filters.gaussian(rivers, sigma=1.2)[...,0] # rivers is now [0,1]
        rivers = (rivers < 0.99)*1
        rivers = rivers.astype(np.uint8)
        
        if debug:
            plt.figure(figsize=debug_img_size)
            plt.title("rivers")
            plt.imshow(rivers)
            plt.show()
    
    with step("----Combining coast and rivers"):
        final_mask = (mask-rivers)*255
        anti_final_mask = final_mask*-1+255
        if debug:
            plt.figure(figsize=debug_img_size)
            plt.title("final water")
            plt.imshow(final_mask)
            plt.show()
    
    #############################################
    # GENERATION
    #############################################
    
    #------------------------------------------------------------------------------
    # image_scaling
    if config.pipeline.extra_scaling != 1:
        import scipy.ndimage
        final_mask = scipy.ndimage.zoom(final_mask, config.pipeline.extra_scaling, order=0)
        anti_final_mask = scipy.ndimage.zoom(anti_final_mask, config.pipeline.extra_scaling, order=0)
        rivers = scipy.ndimage.zoom(rivers, config.pipeline.extra_scaling, order=0)

    with step("----Terrain generation"):
        wpad = config.terrain.water_padding
        if wpad > 0:
            wp = np.zeros(
                (final_mask.shape[0]+2*wpad,
                final_mask.shape[1]+2*wpad))
            wp[wpad:-wpad,wpad:-wpad] = final_mask
            final_mask=wp
        terrain_height = generate_terrain(final_mask, **config.terrain.land)
        if debug:
            plt.figure(figsize=debug_img_size)
            plt.title("terrain height")
            plt.imshow(terrain_height)
            plt.show()
    
    #------------------------------------------------------------------------------
    with step("----Underwater generation"):
        wpad = config.terrain.water_padding
        if wpad > 0:
            wp = np.ones(
                (anti_final_mask.shape[0]+2*wpad,
                anti_final_mask.shape[1]+2*wpad))*255
            wp[wpad:-wpad,wpad:-wpad] = anti_final_mask
            anti_final_mask=wp
        water_depth = generate_terrain(anti_final_mask, **config.terrain.water)
        if debug:
            plt.figure(figsize=debug_img_size)
            plt.title("water depth")
            plt.imshow(water_depth)
            plt.show()
    
    #------------------------------------------------------------------------------
    with step("----Combining terrain and water heights"):
        m1 = final_mask[wpad:-wpad,wpad:-wpad] if wpad > 0 else final_mask
        _m1 = m1 < 255
        m2 = terrain_height[wpad:-wpad,wpad:-wpad] if wpad > 0 else terrain_height
        m3 = water_depth[wpad:-wpad,wpad:-wpad] if wpad > 0 else water_depth
        combined = m2 - 0.4*_m1*m3

        combined = combined[pad:-pad,pad:-pad]

        if debug:
            plt.figure(figsize=debug_img_size)
            plt.title("combined map")
            plt.imshow(combined)
            plt.show()
    
    #############################################
    # EXPORT 1
    #############################################
    
    with step("Exporting height map"):
        qmap = (combined-combined.min())/(combined.max()-combined.min())
        qmap = (qmap*255).astype(np.uint8)
        qimg = PIL.Image.fromarray(qmap).convert('L')
        if config.export.size > 0:
            qimg = qimg.resize(
                (config.export.size, config.export.size),
                PIL.Image.NEAREST
            )
        qimg = PIL.ImageOps.mirror(qimg)
        qimg.save(f"output/height_{realm_number}.png")
    
    #############################################
    # COLORING
    #############################################

    # TODO: put this in coloring.py
    WATER_COLORS = [
        np.array([ 74., 134., 168.]),
        np.array([ 18.,  59., 115.]),
        np.array([ 66., 109., 138.])
    ]
    water_color = choice(WATER_COLORS)
    
    with step("Coloring"):
        biomes = [moderate, cold, snow, savanna, desert]
        biome = choice(biomes)
        colorqmap = run_coloring(biome, combined)
        colorqmap = inject_water_tile(colorqmap, m1, water_color) #m1 is the landmap
    
    #############################################
    # EXPORT 2
    #############################################
    
    with step("Exporting color map"):
        colorqmap_export = colorqmap.astype(np.uint8)
        colorqmap_export = PIL.Image.fromarray(colorqmap_export)
        if config.export.size > 0:
            colorqmap_export = colorqmap_export.resize(
                (config.export.size, config.export.size),
                PIL.Image.NEAREST
            )
        colorqmap_export = PIL.ImageOps.mirror(colorqmap_export)
        colorqmap_export.save(f"output/color_{realm_number}.png")

    #############################################
    # FileToVox
    #############################################

    with step("Finding index of water tile"):
        # need to do an exhaustive check here
        colors_used = np.unique(np.reshape(colorqmap, [-1, 3]), axis=0)
        for i, c in enumerate(list(colors_used)):
            if np.array_equal(c, water_color):
                break
        water_index = len(list(colors_used))-i
        logger.debug(f"water_index: {water_index}")

        with open("resources/flood.json") as json_file:
            data = json.load(json_file)
        # change flooding water index
        data["steps"][0]["TargetColorIndex"] = water_index-1
        data["steps"][0]["water_color"] = [
            int(water_color[0]),
            int(water_color[1]),
            int(water_color[2]),
        ]
        # TODO: also change the water height here!
        with open(f"output/flood_{realm_number}.json", "w") as json_file:
            json.dump(data, json_file)

    if debug:

        # also save intermediate files
        def export_np_array(arr, name):
            arr = arr.astype(np.uint8)
            img = PIL.Image.fromarray(arr)
            if config.export.size > 0:
                img = img.resize(
                    (config.export.size, config.export.size),
                    PIL.Image.NEAREST
                )
            img = PIL.ImageOps.mirror(img)
            img.save(f"debug/debug_{name}.png")
        export_np_array(rivers, "rivers")
        export_np_array(m1, "final_mask")
        export_np_array(m2, "terrain_height")
        export_np_array(m3, "sea_depth")

        return combined

    #############################################
    # VoxManipulation
    #############################################

@click.command()
@click.argument("realm_path")
@click.option("--config", default="pipeline/config.yaml")
@click.option("--debug", default=False)
def parse(realm_path, config, debug):
	config = OmegaConf.load(config)
	run_pipeline(realm_path, config, debug)

if __name__=="__main__":
	parse()
