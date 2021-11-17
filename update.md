# Update

It's possible to go from an .svg to a renderable .vox model in magicavoxel with Biomes `[moderate, cold, tropical, savanna, desert]` and transparent water of different colors.

## Pipeline

Right now the pipeline is at v4.
- Lots of hours spent into getting the land-sea masking working right.
- I might have over-engineered a functionn or 3, but now going back to an early idea: loop closing. So I'll close all land loops and then just polyfill the shapes with scipy. All things work great for 10 realms, but when you scale to 50, you already get a couple of odd ones out.
- After the generation of height map + color map, we have to go from `.pngs` to `.vox`: this is done with FileToVox. It needed quite some updates on the `.vox` struct which has changed without backward comp.
- FileToVox is used to extract height from a `.png` and apply another colormap onto the generated map. After that, a flooding shader is used to add water to the map.
- The first FileToVox call is made with `wine` due to some error. This unfortunately limits the size of the generated map due to OutOfMemory issues. It can be mitigated by calling this step in the pipeline on a windows machine.
- The second FileToVox call is made in linux, but there is another trick. The generated height map is lifted from the ground. So we first need to do a fill command up to `132`, then do a subtract fill to `127` to remove a portion again. This might also cause OoM issues. It could be mitigated by moving the height map to the ground plane.
- The final step in the process is done by manipulating the `.vox` struct directly because we want to set the water shader to glass and also the camera settings. Because of the struct, I just created a donor file with all the right settings and then copy over the chunks (sub structs) that I needed.

## Codebase

- Trying to add documentation on the go.
- Feedback on readability is appreciated, must be really low at this point.

## Performance

- Python script is fully parallalizable. This means we can pump out color map, height maps and some metadata.
- Made optimizations towards coloring (`<1s`) and added a faster poisson disc sampler.
- Performance bottleneck is the MagicaVoxel renderer as it does not have a machine-friendly interface
  - Looked into the voxviewer from the same dev but is just a lightweight port of MV
  - Right now working with a `donor` that has all the rendering elements such as camera/environment set. These binary structs are copied to an `acceptor` (the new file to be rendered).
  - For rendering, we just need to have the computer press `8` to go to the preset camera view and then press `render`

## Ideas and improvements

- Water
  - Optimize river shape
  - Flood sometimes does not reach river beds in the mountains
  - Couple water color to biome selection, now it is decoupled
- Biomes
  - Optimize colors of biomes
  - Add biomes
  - Tie biomes to other svg metadata such as order
- Cities
  - Render placeholder for cities
- Procedurality
  - Find different terrain types for the terrain generation.
- Size
  - Right now the vox map is 700x700 but I've made changes to the pipeline such that it also supports higher res maps.
- Clouds can really add a sense of scale to the realm, but are not yet put into code.