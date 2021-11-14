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
import skimage
from skimage.segmentation import flood_fill
from skimage.morphology import dilation, square

import json

import logging
logger = logging.getLogger("realms")

sys.path.append("terrain-erosion-3-ways/")
from river_network import *

sys.path.append("pipeline")
from svg_extraction import SVGExtractor, get_heightline_centers
from svg_extraction import (
    get_city_coordinates, get_island_coordinates, get_island_centers,
    get_orthogonal_samples, get_coast_coordinates)
from image_ops import flood_image, flood_islands, greedy_find_clusters
from utils import *

from coloring import inject_water_tile, moderate, cold, run_coloring, tropical, savanna, desert

def run_pipeline(realm_path, config, debug=False):
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # set some often-used parameters
    svgp = config.svg.padding
    debug_img_size = (10,10)
    extra_scaling = None
    
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
        coast_img = extractor.get_img()

    #############################################
    # MASKING
    #############################################
    
    logger.info("Starting ground-sea mask logic")
    
    #------------------------------------------------------------------------------
    with step("----Calculating island centers"):
        island_centers = get_island_centers(
            coast_drawing,
            scaling=config.svg.scaling
        )
        logger.debug(f"island center [y,x]: {island_centers}")
    
    with step("----Extracting city centers"):
        city_drawing = extractor.cities()
        city_centers = get_city_coordinates(
            city_drawing,
            scaling=config.svg.scaling
        )
        logger.debug(f"city centers [y,x]: {city_centers}")

    with step("----Extracting heightline centers"):
        height_drawing = extractor.height()
        heightline_centers = get_heightline_centers(height_drawing)
        heightline_clusters = greedy_find_clusters(heightline_centers)
        logger.debug(f"first 3 heightline clusters [y,x]: {heightline_clusters[:3]}")

    #------------------------------------------------------------------------------
    # some small checks
    logger.info("----Filtering centers")
    with step("--------Filtering island centers"):
        island_centers = filter_within_bounds(
            island_centers,
            coast_drawing.width,
            coast_drawing.height,
            config.svg.padding,
        )

    with step("--------Filtering city centers"):
        city_centers = filter_within_bounds(
            city_centers,
            city_drawing.width,
            city_drawing.height,
            config.svg.padding,
        )

    with step("--------Filtering height clusters"):
        heightline_clusters = filter_within_bounds(
            heightline_clusters,
            city_drawing.width,
            city_drawing.height,
            config.svg.padding,
        )

    if debug:
        plt.figure(figsize=debug_img_size)
        plt.title("coast image with island centers (red) and city centers (green)")
        plt.imshow(coast_img)
        for center in island_centers:
            plt.plot(center[1], center[0], 'ro')
        for center in city_centers:
            plt.plot(center[1], center[0], 'go')
        for centroid in heightline_clusters:
            plt.plot(centroid[1], centroid[0], 'go')
        plt.show()

    #------------------------------------------------------------------------------
    with step("----Cropping and flooding cities"):
        coast_coordinates = get_coast_coordinates(coast_drawing)
        flooded_image = flood_image(
            coast_img,
            city_centers, 
            config.svg.padding,
            coast_coordinates=coast_coordinates,
            debug=debug
        )

    #------------------------------------------------------------------------------
    with step("----Flooding from heightline clusters"):
        flooded_image = flood_image(
            flooded_image,
            heightline_clusters, 
            config.svg.padding,
            coast_coordinates=coast_coordinates,
            debug=debug
        )

    #------------------------------------------------------------------------------
    with step("----Cropping and flooding islands"):
        flooded_image = flood_islands(
            flooded_image,
            coast_drawing,
            config.svg.padding,
            debug=debug
        )
    if debug:
        plt.figure(figsize=debug_img_size)
        plt.title("Cropped and flooded image")
        plt.imshow(flooded_image)
        plt.show()

    #------------------------------------------------------------------------------
    with step("----Refining flooded image"):
        if debug:
            plt.figure(figsize=debug_img_size)
            plt.title("sampling locations for refining mask + found centers")
            plt.imshow(flooded_image)
            # no plt.show() here!
            
        refined_flooded_image = flooded_image.copy()
            
        for path_group in coast_drawing.contents[0].contents:
            path = path_group.contents[0]
            samples1, samples2 = get_orthogonal_samples(path, scaling=config.svg.scaling)
            samples1, samples2 = np.vstack(samples1), np.vstack(samples2)
                
            aggr = []
            for (s1, s2) in zip(samples1, samples2):
                x1, y1 = int(s1[0]-svgp), int(s1[1]-svgp)
                x2, y2 = int(s2[0]-svgp), int(s2[1]-svgp)
                # only add colors if BOTH orthogonals are within bounds
                if 0 <= x1 < flooded_image.shape[0] and 0 <= y1 < flooded_image.shape[1]:
                    if 0 <= x2 < flooded_image.shape[0] and 0 <= y2 < flooded_image.shape[1]:
                        aggr.append((flooded_image[y1,x1], flooded_image[y2,x2]))

            if len(aggr) > 6:
                aggr = np.vstack(aggr)
                aggr = np.mean(np.abs(aggr[:,0]-aggr[:,1]))
                if aggr < 20:
                    center = np.mean(samples1-svgp, axis=0)
                    center = np.clip(center, 0, flooded_image.shape[0]-1)
                    refined_flooded_image = flood_fill(
                        refined_flooded_image,
                        (int(center[1]), int(center[0])),
                        0)
                    if debug:
                        plt.plot(center[0], center[1], 'go', markersize=5)
            if debug:
                plt.plot(samples1[:,0]-svgp, samples1[:,1]-svgp, 'ro', markersize=1)
                plt.plot(samples2[:,0]-svgp, samples2[:,1]-svgp, 'ro', markersize=1)
        flooded_image = refined_flooded_image.copy()
        if debug:
            plt.figure(figsize=debug_img_size)
            plt.title("refined flooded image")
            plt.imshow(refined_flooded_image)
            plt.show()

    #------------------------------------------------------------------------------
    with step("----Extracting rivers"):
        extractor.rivers()
        rivers = np.asarray(extractor.get_img())[svgp:-svgp,svgp:-svgp,0] # rivers is now [0,255]

        rivers = skimage.filters.gaussian(rivers, sigma=1.2) # rivers is now [0,1]
        rivers = (rivers < 0.99)*255
        
        if debug:
            plt.figure(figsize=debug_img_size)
            plt.title("rivers")
            plt.imshow(rivers)
            plt.show()
    
    with step("----Extracting rivers"):
        final_mask = (np.logical_or(flooded_image, rivers)-1)*-255
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
    if extra_scaling:
        import scipy.ndimage
        final_mask = scipy.ndimage.zoom(final_mask, 2, order=0)
        anti_final_mask = scipy.ndimage.zoom(anti_final_mask, 2, order=0)
        rivers = scipy.ndimage.zoom(rivers, 2, order=0)

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
        qimg = qimg.resize(
            (config.export.size, config.export.size),
            PIL.Image.NEAREST
        )
        qimg = PIL.ImageOps.mirror(qimg)
        qimg.save(f"output/height_{realm_number}.png")
    
    #############################################
    # COLORING
    #############################################
    
    with step("Coloring"):
        biomes = [moderate, cold, tropical, savanna, desert]
        biome = choice(biomes)
        colorqmap = run_coloring(biome, combined)
        colorqmap = inject_water_tile(colorqmap, m1) #m1 is the landmap
    
    #############################################
    # EXPORT 2
    #############################################
    
    with step("Exporting color map"):
        colorqmap_export = colorqmap.astype(np.uint8)
        colorqmap_export = PIL.Image.fromarray(colorqmap_export)
        colorqmap_export = colorqmap_export.resize(
            (config.export.size, config.export.size),
            PIL.Image.NEAREST
        )
        colorqmap_export = PIL.ImageOps.mirror(colorqmap_export)
        colorqmap_export.save(f"output/color_{realm_number}.png")

    #############################################
    # FileToVox
    #############################################

    WATER_COLORS = [
        np.array([ 74., 134., 168.])
    ]
    water_color = choice(WATER_COLORS)

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

    with step("Converting file to Vox"):
        pass

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
