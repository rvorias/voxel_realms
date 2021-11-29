"""
Author: rvorias
"""

import logging
logger = logging.getLogger("realms")

import sys
import json
import glob
from omegaconf import OmegaConf

from metaflow import FlowSpec, step, Parameter

import numpy as np
import random as rand
from random import choice

import matplotlib.pyplot as plt
import PIL
import PIL.ImageOps
import skimage

sys.path.append("terrain-erosion-3-ways/")
from river_network import *

sys.path.append("pipeline")
from svg_extraction import SVGExtractor, get_heightline_centers, get_city_coordinates
from image_ops import close_svg, draw_cities
from utils import generate_terrain

from coloring import inject_water_tile, run_coloring
from coloring import cold, moderate, savanna, desert, snow

class ParameterFlow(FlowSpec):
    @step
    def start(self):
        self.realm_paths = glob.glob("svgs/*.svg")

        self.config = OmegaConf.load("pipeline/config.yaml")

        self.debug = False
        if self.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        self.REL_SEA_SCALING = 0.2
        self.hscales = {
            # k:   (scale, hmap, flood)
            "low": (0.33, 64),
            "med": (0.66, 64),
            "hi":  (  1., 64),
        }
        self.HSCALE = "hi"
        # set some often-used parameters
        self.debug_img_size = (10,10)
        self.next(self.set_seeds_and_init, foreach="realm_paths")

    @step
    def set_seeds_and_init(self):
        self.realm_path = self.input
        self.realm_number = int(self.realm_path.split("/")[-1][:-4])
        logging.info(f"Processing realm number: {self.realm_number}")
        np.random.seed(self.realm_number)
        rand.seed(self.realm_number)
        self.next(self.setup_extractor)

    @step
    def setup_extractor(self):
        self.extractor = SVGExtractor(self.realm_path, scale=self.config.svg.scaling)
        if self.debug:
            self.extractor.show(self.debug_img_size)
        self.next(self.extract_drawings)

    @step
    def extract_drawings(self):
        self.coast_drawing = self.extractor.coast()
        self.heightline_drawing = self.extractor.height()
        self.cities_drawing = self.extractor.cities()
        self.city_centers = get_city_coordinates(self.cities_drawing)
        self.next(self.extract_rivers, self.create_land_sea_mask)

    @step
    def extract_rivers(self):
        self.extractor.rivers()
        rivers = np.asarray(self.extractor.get_img()) # rivers is now [0,255]
        original_rivers = rivers.copy()

        # make bit thicker
        rivers = skimage.filters.gaussian(rivers, sigma=1.2)[...,0] # rivers is now [0,1]
        rivers = (rivers < 0.99)*1
        rivers = rivers.astype(np.uint8)
        original_rivers = skimage.filters.gaussian(original_rivers, sigma=0.2)[...,0] # rivers is now [0,1]
        original_rivers = (original_rivers < 0.85)*1
        original_rivers = original_rivers.astype(np.uint8)

        self.rivers = rivers
        self.original_rivers = original_rivers
        
        if self.debug:
            plt.figure(figsize=self.debug_img_size)
            plt.title("fat rivers")
            plt.imshow(rivers)
            plt.show()
            plt.figure(figsize=self.debug_img_size)
            plt.title("original rivers")
            plt.imshow(original_rivers)
            plt.show()

        self.next(self.combine_coast_and_rivers)

    @step
    def create_land_sea_mask(self):
        # uses a fixed padding of 32
        mask = close_svg(self.coast_drawing, debug=self.debug)

        # check if we need to flip
        centers = get_heightline_centers(self.heightline_drawing)
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

        self.mask = mask

        if self.debug:
            logger.debug(f"mask_shape: {mask.shape}")
            plt.figure(figsize=self.debug_img_size)
            plt.title("land-sea mask")
            plt.imshow(mask)
            plt.show()

        self.next(self.combine_coast_and_rivers)

    @step
    def combine_coast_and_rivers(self, inputs):
        self.merge_artifacts(inputs, exclude=['extractor'])
        self.final_mask = (
            self.mask - 
            self.rivers) * 255
        self.anti_final_mask = self.final_mask*-1+255

        if self.debug:
            plt.figure(figsize=self.debug_img_size)
            plt.title("final water")
            plt.imshow(self.final_mask)
            plt.show()

        self.next(self.scale_images)

    @step
    def scale_images(self):
        if self.config.pipeline.extra_scaling != 1:
            import scipy.ndimage
            self.final_mask = scipy.ndimage.zoom(
                self.final_mask, self.config.pipeline.extra_scaling, order=0)
            self.anti_final_mask = scipy.ndimage.zoom(
                self.anti_final_mask, self.config.pipeline.extra_scaling, order=0)
            self.rivers = scipy.ndimage.zoom(
                self.rivers, self.config.pipeline.extra_scaling, order=0)
            self.original_rivers = scipy.ndimage.zoom(
                self.original_rivers, self.config.pipeline.extra_scaling, order=0)
        self.next(self.generate_terrain, self.generate_sea)

    @step
    def generate_terrain(self):
        np.random.seed(self.realm_number)
        rand.seed(self.realm_number)
        wpad = self.config.terrain.water_padding
        if wpad > 0:
            wp = np.zeros(
                (self.final_mask.shape[0]+2*wpad,
                self.final_mask.shape[1]+2*wpad))
            wp[wpad:-wpad,wpad:-wpad] = self.final_mask
            final_mask_padded=wp
        self.terrain_height = generate_terrain(final_mask_padded, **self.config.terrain.land)
        if self.debug:
            plt.figure(figsize=self.debug_img_size)
            plt.title("terrain height")
            plt.imshow(self.terrain_height)
            plt.show()
        self.next(self.combine_terrain_and_sea_heights)

    @step
    def generate_sea(self):
        np.random.seed(self.realm_number)
        rand.seed(self.realm_number)
        wpad = self.config.terrain.water_padding
        if wpad > 0:
            wp = np.ones(
                (self.anti_final_mask.shape[0]+2*wpad,
                self.anti_final_mask.shape[1]+2*wpad))*255
            wp[wpad:-wpad,wpad:-wpad] = self.anti_final_mask
            anti_final_mask_padded=wp
        self.water_depth = generate_terrain(anti_final_mask_padded, **self.config.terrain.water)
        if self.debug:
            plt.figure(figsize=self.debug_img_size)
            plt.title("water depth")
            plt.imshow(self.water_depth)
            plt.show()
        self.next(self.combine_terrain_and_sea_heights)

    @step
    def combine_terrain_and_sea_heights(self, inputs):
        self.merge_artifacts(inputs)
        wpad = self.config.terrain.water_padding
        m1 = self.final_mask
        _m1 = m1 < 255
        m2 = self.terrain_height[wpad:-wpad,wpad:-wpad] if wpad > 0 else self.terrain_height
        m3 = self.water_depth[wpad:-wpad,wpad:-wpad] if wpad > 0 else self.water_depth
        combined = m2 - self.REL_SEA_SCALING*_m1*m3

        # fix rivers height
        # combined = np.where(((combined > 0) & (original_rivers > 0)), 55, combined)
        combined = np.where(self.original_rivers > 0, -.1, combined)

        # TODO: put to config
        pad = 32
        if self.config.pipeline.extra_scaling != 1:
            pad = int(pad*self.config.pipeline.extra_scaling)
        combined = combined[pad:-pad, pad:-pad]

        # this snippet takes care of holes in the sea floor
        # sometimes the bottom layer gets remove so we just set one voxel to be the lowest.
        co = np.unravel_index(np.argmin(combined, axis=None), combined.shape)
        lowest_val = combined[co]
        combined = combined.clip(combined.min()+0.05, combined.max())
        combined[co] = lowest_val

        self.combined = combined

        if self.debug:
            plt.figure(figsize=self.debug_img_size)
            plt.title("combined map")
            plt.imshow(self.combined)
            plt.show()

        self.next(self.scale_heights)

    @step
    def scale_heights(self):
        hmap_normalized = (self.combined - self.combined.min()) / (self.combined.max() - self.combined.min())

        # first transform the real sea scaling the same as we are going
        # to transform the map itself
        coast_height_rescaled = (self.REL_SEA_SCALING - self.combined.min()) / (self.combined.max() - self.combined.min())
        # rescale the height of the map
        self.hmap = np.where(
            hmap_normalized > coast_height_rescaled,
            (hmap_normalized - coast_height_rescaled) * self.hscales[self.HSCALE][0] + coast_height_rescaled,
            hmap_normalized
        )
        self.next(self.export_heightmap)

    @step
    def export_heightmap(self):
        hmap = (self.hmap * 255).astype(np.uint8)
        himg = PIL.Image.fromarray(hmap).convert('L')

        himg, _ = draw_cities(
            self.city_centers,
            himg=himg,
            extra_scaling=self.config.pipeline.extra_scaling)

        if self.config.export.size > 0:
            himg = himg.resize(
                (self.config.export.size, self.config.export.size),
                PIL.Image.NEAREST
            )
        himg = PIL.ImageOps.mirror(himg)
        himg.save(f"output/height_{self.realm_number}.png")
        self.next(self.color)

    @step
    def color(self):
        np.random.seed(self.realm_number)
        rand.seed(self.realm_number)
        # TODO: put this in coloring.py
        WATER_COLORS = [
            np.array([74., 134., 168.]),
            np.array([18., 59., 115.]),
            np.array([66., 109., 138.])
        ]
        self.water_color = choice(WATER_COLORS)

        biomes = [moderate, cold, snow, savanna, desert]
        biome = choice(biomes)
        self.colormap = run_coloring(biome, self.combined)

        self.next(self.export_color)

    @step
    def export_color(self):
        colormap = self.colormap.astype(np.uint8)
        colorimg = PIL.Image.fromarray(colormap)

        _, colorimg = draw_cities(
            self.city_centers,
            cimg=colorimg,
            extra_scaling=self.config.pipeline.extra_scaling
        )

        # oops, quick reconvert
        colormap = np.array(colorimg)
        colormap = inject_water_tile(colormap, self.final_mask, self.water_color)  # m1 is the landmap
        colormap = colormap.astype(np.uint8)
        colorimg = PIL.Image.fromarray(colormap)

        if self.config.export.size > 0:
            colorimg = colorimg.resize(
                (self.config.export.size, self.config.export.size),
                PIL.Image.NEAREST
            )
        colorimg = PIL.ImageOps.mirror(colorimg)
        colorimg.save(f"output/color_{self.realm_number}.png")

        self.next(self.export_vox_settings)

    @step
    def export_vox_settings(self):
        # need to do an exhaustive check here
        colors_used = np.unique(np.reshape(self.colormap, [-1, 3]), axis=0)
        for i, c in enumerate(list(colors_used)):
            if np.array_equal(c, self.water_color):
                break
        water_index = len(list(colors_used))-i
        logger.debug(f"water_index: {water_index}")

        with open("resources/flood.json") as json_file:
            data = json.load(json_file)
        # change flooding water index
        data["steps"][0]["TargetColorIndex"] = water_index-1
        data["steps"][0]["water_color"] = [
            int(self.water_color[0]),
            int(self.water_color[1]),
            int(self.water_color[2]),
        ]
        data["steps"][0]["hm_param"] = self.hscales[self.HSCALE][1]
        with open(f"output/flood_{self.realm_number}.json", "w") as json_file:
            json.dump(data, json_file)
        self.next(self.join_for)

    @step
    def join_for(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    ParameterFlow()































