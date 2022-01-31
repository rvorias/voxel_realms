"""
This file holds the main run logic for the pipeline.
Author: rvorias
"""
import os
import sys
import click
from omegaconf import OmegaConf
from pathlib import Path

import numpy as np
import random as rand
from random import choice

import matplotlib.pyplot as plt
import PIL
import PIL.ImageOps
import skimage
import skimage.filters

import json
from functools import partial

import logging

logger = logging.getLogger("realms")

sys.path.append("terrain-erosion-3-ways/")
from river_network import *

sys.path.append("pipeline")
from svg_extraction import SVGExtractor, get_heightline_centers, get_city_coordinates
from image_ops import close_svg, slice_cont, generate_city, put_cities, extract_land_sea_direction
from utils import *

from coloring import biomes, WATER_COLORS, color_from_json


def run_pipeline(realm_path, config="pipeline/config.yaml", debug=False):
    REL_SEA_SCALING = config.terrain.relative_sea_depth_scaling
    HSCALES = config.terrain.height_scales
    # hscale = choice(list(HSCALES))
    hscale = "hi"
    PAD = config.pipeline.general_padding
    MAIN_OUTPUT_DIR = Path(config.pipeline.main_output_dir)
    RESOURCES_DIR = Path(config.pipeline.resources_dir)

    DEBUG_IMG_SIZE = (10, 10)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    realm_number = int(realm_path.replace("svgs/", "").replace("svgs\\", "").replace(".svg", "").replace("../", ""))
    step = partial(Step, realm_number=realm_number)

    with step("Creating output folder if needed"):
        
        subdirs = [
            "directions",
            "colors",
            "errors",
            "heights",
            "heights_no_cities",
            "hslices",
            "flood_configs",
            "masks",
            "palettes",
            "rivers",
        ]
        if not os.path.isdir(MAIN_OUTPUT_DIR):
            os.mkdir(MAIN_OUTPUT_DIR)
        for subdir in subdirs:
            if not os.path.isdir(MAIN_OUTPUT_DIR /subdir):
                os.mkdir(MAIN_OUTPUT_DIR / subdir)
            

    with step("Set seeds and init"):
        logging.info(f"Processing realm number: {realm_number}")
        np.random.seed(realm_number)
        rand.seed(realm_number)

    with step("Randomizing config parameters"):
        config.terrain.land.river_downcutting_constant = rand.uniform(0.1, 0.3)
        config.terrain.land.default_water_level = rand.uniform(0.9, 1.1)
        config.terrain.evaporation_rate = rand.uniform(0.1, 0.3)
        config.terrain.coastal_dropoff = rand.uniform(70, 90)

    with step("Setting up extractor"):
        extractor = SVGExtractor(realm_path, scale=config.svg.scaling)
        if debug:
            extractor.show(DEBUG_IMG_SIZE)

    with step("Extracting coast"):
        coast_drawing = extractor.coast()

    with step("Extracting heightlines"):
        heightline_drawing = extractor.height()

    #############################################
    # MASKING
    #############################################

    with step("Starting ground-sea mask logic"):
        # uses a fixed padding of 32
        print(realm_number)
        mask = close_svg(coast_drawing, debug=debug)

        centers = get_heightline_centers(heightline_drawing)
        sum = _sum = 0
        _mask = (mask - 1) // 255
        for center in centers:
            sum += mask[int(center[0]), int(center[1])]
            _sum += _mask[int(center[0]), int(center[1])]
        if _sum > sum:
            mask = _mask

        # add islands
        mask += close_svg(coast_drawing, debug=debug, islands_only=True)
        mask = mask.clip(0, 1)

        # extend land towards edges
        h, w = mask.shape
        for i in range(PAD):
            mask[:, i] = mask[:, PAD]
            mask[:, -i - 1] = mask[:, w - PAD]
            mask[i, :] = mask[PAD, :]
            mask[-i - 1, :] = mask[h - PAD, :]

        if debug:
            logger.debug(f"mask_shape: {mask.shape}")
            plt.figure(figsize=DEBUG_IMG_SIZE)
            plt.title("land-sea mask")
            plt.imshow(mask)
            plt.show()

    # -----------------------------------------------------------------------------

    with step("---Calculating land-sea direction"):
        direction = extract_land_sea_direction(mask[PAD:-PAD,PAD:-PAD], debug=debug)
        with open(MAIN_OUTPUT_DIR / f"directions/{realm_number}_{direction:3f}.direction", "w") as file:
            file.write("")
        if debug:
            imshow(mask[PAD:-PAD,PAD:-PAD], "cropped mask")

@click.command()
@click.argument("realm_path")
@click.option("--config", default="pipeline/config.yaml")
@click.option("--debug", default=False)
def parse(realm_path, config, debug):
    config = OmegaConf.load(config)
    run_pipeline(realm_path, config, debug)


if __name__ == "__main__":
    parse()
