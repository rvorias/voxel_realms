# voxel_realms

## Quickstart

- Run `$ git submodule update --init --recursive`
- I've added a requirements.txt file but it might not work 100% yet.
- Check out `notebooks/pipeline.ipynb`

## FileToVox

You need to have Wine (>4.0.0) installed when running on Linux

- Run `$ bash setup.sh` to download and unzip the FileToVox binaries.
- Then run `$ wine FileToVox-v1.13-win/FileToVox.exe --i output/test_total.png -o output/map --hm=75`

## Todo

Pipeline
- [x] Use config files for parameters
- [x] Hook up FileToVox
- [ ] Hook up more data like resources etc
- [ ] Hook up biomes

Terrain Generation
- [x] Identify islands
- [x] Automatically create land-sea mask
- [x] Add support for rivers
- [ ] Add support for cities

MagicaVoxel
- [ ] Write (plugin) entry for pipeline to interact with
- [ ] Figure out Texturing
- [x] Fix bug of png images not loading into MagicaVoxel

Code
- [x] Migrate to .py files
- [ ] CLI
