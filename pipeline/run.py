"""
Author: rvorias
"""

from svg_extraction import SVGExtractor
from svg_extraction import get_city_coordinates

import click

from omegaconf import OmegaConf

# 1 get the SVG file and process it
def main():
	config = OmegaConf.load("pipeline/config.yaml")
	print(config)
	extractor = SVGExtractor("../resources/pen.svg")
	# coast

# 2 Do some checks and make sure all input exists and is valid

# 3 Run the terrain generation

# 4 Figure out texturizing

# 5 Pipe to MagicaVoxel

if __name__=="__main__":
	main()