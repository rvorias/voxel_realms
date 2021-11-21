# Voxel_realms

## Features
- data extraction for all svg elements: cities, coast, heightlines, names
- land-sea mask finding algorithm robust over many variations
- terrain generation for both land and sea, with rivers taken into account
- biomes can be composed through color functions that act on height data
- heightmap + colormap export to .vox format
- filling of the sea with water
- conversion of water from diffuse to glass material for see-through effect
- desired camera parameters are injected into the generated .vox file
- current maps are 700x700 pixels, with the current bottlenecks 1000x1000 is also possible

## Future work
- Rivers: shapes of rivers, making sure rivers have actual water in them.
- More water types, water color tied to biome
- More styles of terrain
- Better biome coloring, more biomes, biomes tied to realm resources
- Adding clouds to the .vox. (helps with setting the scale vibe)
- Automatic rendering (now have to click "render' button)
- Remove conversion bottlenecks so that higher scales are possible

## Requirements
This pipeline has been tested on linux, but will likely also
work on Windows. For linux you need Wine > 6.

## Quickstart
- Download Conversion tools and MV: `$ bash setup.sh` 
- Run `$ git submodule update --init --recursive`
- Install a venv e.g.: `pipenv install -r requirements.txt`
- Check out `notebooks/pipeline.ipynb`

## Acknowledgements
- https://github.com/ephtracy/ephtracy.github.io
- https://github.com/Zarbuz/FileToVox
- https://github.com/alexhunsley/numpy-vox-io
