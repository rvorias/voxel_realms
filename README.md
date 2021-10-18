# voxel_realms

## Quickstart

- Run `$ git submodule update --init --recursive`
- I've added a requirements.txt file but it might not work 100% yet.
- Check out `notebooks/visualizer.ipynb`
- In order to use the `xs height` in MV: copy `resources/height.txt` into MV's `shaders`

## Todo

Pipeline
- [ ] Use config files for parameters
- [ ] Hook up FileToVox
- [ ] Hook up more data like resources etc
- [ ] Hook up biomes

Terrain Generation
- [ ] Add support for rivers
- [ ] Add support for cities

MagicaVoxel
- [ ] Write (plugin) entry for pipeline to interact with
- [ ] Figure out Texturing
- [ ] Fix bug of png images not loading into MagicaVoxel

Code
- [ ] Migrate to .py files
- [ ] CLI
