import dearpygui.dearpygui as dpg
import json
import numpy as np
import array

from pipeline.coloring import run_coloring, biomes, colorize_perlin
import PIL
import matplotlib.pyplot as plt

dpg.create_context()
dpg.create_viewport(title='Custom Title', width=2000, height=1200)

###############################################
# FUNCTIONS
###############################################

def save_callback(sender, app_data):
    for i, color in enumerate(data[dpg.get_value("biome")]["colors"]):
        data[dpg.get_value("biome")]["colors"][i] = dpg.get_value(f"c{i}")
        data[dpg.get_value("biome")]["mus"][i] = [
            dpg.get_value(f"c{i}m1"), dpg.get_value(f"c{i}m2")
        ]
        data[dpg.get_value("biome")]["sigmas"][i] = [
            dpg.get_value(f"c{i}s1"), dpg.get_value(f"c{i}s2")
        ]
    with open("resources/colors.json", "w") as file:
        json.dump(data, file)

def load_biome_callback(sender, app_data):
    for i, color in enumerate(data[dpg.get_value("biome")]["colors"]):
        dpg.set_value(f"c{i}", data[dpg.get_value("biome")]["colors"][i])
        dpg.set_value(f"c{i}m1", data[dpg.get_value("biome")]["mus"][i][0])
        dpg.set_value(f"c{i}m2", data[dpg.get_value("biome")]["mus"][i][1])
        dpg.set_value(f"c{i}s1", data[dpg.get_value("biome")]["sigmas"][i][0])
        dpg.set_value(f"c{i}s2", data[dpg.get_value("biome")]["sigmas"][i][1])
    update_dynamic_texture()

def update_dynamic_texture(*_):

    # build biome
    cmap = np.zeros((*HMAP.shape, 3))
    for i, color in enumerate(data[dpg.get_value("biome")]["colors"]):
        layer =  colorize_perlin(
            HMAP,
            [dpg.get_value(f"c{i}m1"), dpg.get_value(f"c{i}m2")],
            [dpg.get_value(f"c{i}s1"), dpg.get_value(f"c{i}s2")],
            16,
            np.array(dpg.get_value(f"c{i}")[:3]),
            [-10, -5, 0, 5, 10]
        )
        cmap = np.where(layer.sum(axis=-1, keepdims=True)>0, layer, cmap)

    new_data = np.reshape(cmap, (-1))/255
    for i in range(len(raw_data)):
        raw_data[i] = new_data[i]

###############################################
# DATA and VARS
###############################################

with open("resources/colors.json", "r") as file:
    data = json.load(file)

H, W = 32*6, 32*3
hmap = np.linspace(1.1,-0.4,H)
HMAP = np.tile(hmap,(W,1)).T

texture_data = []
for i in range(0, H * W):
    texture_data.append(255 / 255)
    texture_data.append(0)
    texture_data.append(255 / 255)

raw_data = array.array('f', texture_data)

###############################################
# WINDOWS
###############################################

with dpg.texture_registry():
    dpg.add_raw_texture(W, H, raw_data, format=dpg.mvFormat_Float_rgb, tag="texture_tag")

with dpg.window(label="Colors", pos=[500,0]):
    dpg.add_combo(
        items=[k for k in data],
        default_value="forest",
        tag="biome",
        callback=load_biome_callback
    )
    dpg.add_button(label="save", callback=save_callback)
    for i, color in enumerate(data[dpg.get_value("biome")]["colors"]):
        dpg.add_color_picker(
            default_value=(color),
            width=200,
            tag=f"c{i}",
            callback=update_dynamic_texture,
            no_side_preview=True,
            alpha_bar=False,
            pos=[i*200, 80]
        )
        dpg.add_slider_float(min_value=-0.4, max_value=1.1,
            width=200, callback=update_dynamic_texture,
            pos=[i*200, 340], tag=f"c{i}m1")
        dpg.add_slider_float(min_value=-0.4, max_value=1.1,
            width=200, callback=update_dynamic_texture,
            pos=[i*200, 360], tag=f"c{i}m2")
        dpg.add_slider_float(min_value=0.01, max_value=0.5,
            width=200, callback=update_dynamic_texture,
            pos=[i*200, 380], tag=f"c{i}s1")
        dpg.add_slider_float(min_value=0.01, max_value=0.5,
            width=200, callback=update_dynamic_texture,
            pos=[i*200, 400], tag=f"c{i}s2")

with dpg.window(label="Tutorial"):
    dpg.add_image("texture_tag", width=W*4, height=H*4)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()