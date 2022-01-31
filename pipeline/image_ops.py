"""
This file holds the logic for applying various operations to bitmasks and svgs.
Author: rvorias
"""

import logging
from math import inf
from random import choice, randint, uniform, random

import numpy as np
import PIL.ImageColor
from PIL import ImageDraw

import matplotlib.pyplot as plt

from reportlab.graphics.shapes import *

logger = logging.getLogger("realms")


def close_svg(drawing, rng=460, debug=False, islands_only=False):
    """This function tries to find open ends of paths and
    connects the ends while going around the image borders.
    
    Paths of lenght < 4 will be ignored.
    
    Args:
        - drawing: the drawing containing paths from extractor.coast()
        - rng: distance of the edge wrt center
        - islands_only: only return islands
    
    Returns:
        bitmap of closed islands
    """
    LIMIT = 30
    OUTPUT_SIZE = 800
    SCALING = 2

    def extend(x, y, bound, limit):
        """Extrapolates points lying at ‘limit‘."""
        assert y != 0
        assert x != 0
        if x < -limit:
            return [-bound, y / x * -bound]
        elif x > limit:
            return [bound, y / x * bound]
        elif y < -limit:
            return [x / y * -bound, -bound]
        elif y > limit:
            return [x / y * bound, bound]
        else:
            print(x,y)
            raise ValueError(f"edge not within limits.")

    # Stage 1: find the first set of open paths and closed paths (islands)
    # paths = drawing.contents[0].contents[0].contents

    arrays = []
    pure_islands = []
    for shape_group in drawing.contents[0].contents:
        path = shape_group.contents[0]
        plen = len(path.points)
        split_array = [[path.points[x], path.points[x + 1]] for x in range(0, plen, 2)]

        # here we are injecting extra points
        if split_array[0] == split_array[-1]:
            # 'tis an island
            pure_islands.append(np.array(split_array))
        else:
            first, last = split_array[0], split_array[-1]
            # check coordinates on the far left:
            first = extend(first[0], first[1], rng, LIMIT)
            last = extend(last[0], last[1], rng, LIMIT)
            new_array = [first]
            new_array.extend(split_array)
            new_array.append(last)
            split_array = new_array

        np_array = np.array(split_array)
        if len(np_array > 3):
            arrays.append(np_array)

    # Stage 2: start from the upper left and sort all end points anti-clockwise.
    begends = []
    begends_list = []
    for a in arrays:
        begends.append({
            "beg": a[0],
            "end": a[-1],
        })
        begends_list.append(a[0])
        begends_list.append(a[-1])

    lefts = [
        a for a in begends_list
        if -rng == a[0]
    ]
    lefts.sort(key=lambda x: -x[1])

    bottoms = [
        a for a in begends_list
        if -rng == a[1]
    ]
    bottoms.sort(key=lambda x: x[0])

    rights = [
        a for a in begends_list
        if rng == a[0]
    ]
    rights.sort(key=lambda x: x[1])

    tops = [
        a for a in begends_list
        if rng == a[1]
    ]
    tops.sort(key=lambda x: -x[0])

    # Stage 3: draw lines in order to connect the sorted endpoints.
    lines = []

    up_left = np.array([-rng, rng])
    bottom_left = np.array([-rng, -rng])
    bottom_right = np.array([rng, -rng])
    up_right = np.array([rng, rng])
    draw = False

    now = up_left
    # go down from the left side
    for co in lefts:
        if draw:
            lines.append([now, co])
            draw = False
        elif not draw:
            now = co
            draw = True

    if draw:
        lines.append([now, bottom_left])
        now = bottom_left
    # go right at the bottom
    for co in bottoms:
        if draw:
            lines.append([now, co])
            draw = False
        elif not draw:
            now = co
            draw = True

    if draw:
        lines.append([now, bottom_right])
        now = bottom_right
    # go up at the right
    for co in rights:
        if draw:
            lines.append([now, co])
            draw = False
        elif not draw:
            now = co
            draw = True

    if draw:
        lines.append([now, up_right])
        now = up_right
    # go left at the top
    for co in tops:
        if draw:
            lines.append([now, co])
            draw = False
        elif not draw:
            now = co
            draw = True

    if draw:
        lines.append([now, up_left])

    # Stage 4: merge open paths and split off closed paths.
    # In the end, everything should become a closed path.
    def island_check(arrays):
        islands = []
        nislands = []
        for a in arrays:
            first, last = a[0], a[-1]
            if (first == last).all():
                islands.append(a)
            else:
                nislands.append(a)
        return islands, nislands

    islands, arrays = island_check(arrays)

    if debug:
        print(f"{len(islands)}")

    # Add lines to arrays. Lines can be seen as very short paths.
    for line in lines:
        arrays.append(np.vstack(line))

    # chain the paths
    while len(arrays) > 0:
        donts = []  # bookkeep which arrays to skip at the end
        alen = len(arrays)
        for i in range(alen):
            a = arrays[i]
            f1, l1 = a[0], a[-1]
            for j in range(i + 1, alen):
                b = arrays[j]
                f2, l2 = b[0], b[-1]
                if (l1 == f2).all():
                    arrays[j] = np.vstack([a, b])
                    donts.append(i)
                    break
                elif (l2 == f1).all():
                    arrays[j] = np.vstack([b, a])
                    donts.append(i)
                    break
                elif (f1 == f2).all():
                    arrays[j] = np.vstack([np.flip(b, axis=0), a])
                    donts.append(i)
                    break
                elif (l1 == l2).all():
                    arrays[j] = np.vstack([a, np.flip(b, axis=0)])
                    donts.append(i)
                    break

        arrays = [a for (i, a) in enumerate(arrays) if i not in donts]

        new_islands, arrays = island_check(arrays)
        islands.extend(new_islands)

    if debug:
        print(f"{len(islands)}")
        print(f"{len(arrays)}")

        al = np.vstack(islands)
        plt.figure(figsize=(10, 10))
        plt.scatter(al[:, 0], -al[:, 1], s=4)
        plt.xlim(-rng - 100, rng + 100)
        plt.ylim(-rng - 100, rng + 100)
        plt.show()

    if islands_only:
        islands = pure_islands

    # Stage 5: scale and cast to PIL.Image
    for i in range(len(islands)):
        islands[i] = (islands[i] * 0.4 + 200) * SCALING

    base = PIL.Image.new("L", (OUTPUT_SIZE, OUTPUT_SIZE), 0)
    drawer = PIL.ImageDraw.Draw(base)

    for island in islands:
        drawer.polygon(list(island.flatten()), fill=1)

    data = np.asarray(base)
    return data


def put_cities(cities, hmap=None, cmap=None, extra_scaling=1., sealevel=0.3):
    """
    Args:
        cities: (x, y, r, hdata, cdata).
        himg:   PIL height nparray.
        cimg:   PIL color nparray.
    """
    for city in cities:
        y, x, r, hdata, cdata = city
        y = int((y - 32) * extra_scaling)
        x = int((x - 32) * extra_scaling)
        r = int(r // 2 * extra_scaling)

        if hmap is not None:
            mean_height = max(hmap[y - r:y + r, x - r:x + r].mean(), sealevel + 0.05)
            hmap[y - r:y + r, x - r:x + r] = np.where(hdata > 0, mean_height + hdata / 6 / 255.,
                                                      hmap[y - r:y + r, x - r:x + r])
        if cmap is not None:
            cmap[y - r:y + r, x - r:x + r] = np.where(cdata > 0, cdata, cmap[y - r:y + r, x - r:x + r])

    return hmap, cmap


def generate_city(r=40):
    from PIL import Image # keep this import here
    dirt = PIL.ImageColor.getrgb("#50352E")

    combinations = {
        # type    # bld_wall, bld_roof, walls
        "wood_0": ["#4d3933", "#543B34", "#483029"],
        "wood_1": ["#867336", "#6C5327", "#70624A"],
        "sand_0": ["#A0A081", "#7E7E4E", "#A8A897"],
        "sand_1": ["#99795B", "#806A55", "#A98C83"],
        "stone_0": ["#9D9DA6", "#3F3F75", "#777788"],
        "stone_1": ["#8F8F8F", "#773838", "#767676"],
        "stone_2": ["#BABABA", "#B79036", "#281C00"],
    }

    selection = choice(list(combinations.keys()))
    pad = 0
    building_height = 20
    highrise_height = 80
    wall_height = 40
    wall_sides = randint(4, 10)
    wall_rot = randint(0, 360)
    base_height = 10
    bw_color, br_color, w_color = [PIL.ImageColor.getrgb(color) for color in combinations[selection]]

    himg = Image.new('L', (r + 2 * pad, r + 2 * pad), color=0)
    cimg = Image.new('RGB', (r + 2 * pad, r + 2 * pad), color=(0, 0, 0))
    hdrawer = ImageDraw.Draw(himg)
    cdrawer = ImageDraw.Draw(cimg)

    # draw walls
    hdrawer.regular_polygon((r // 2 + pad, r // 2 + pad, r // 2), wall_sides, rotation=wall_rot, outline=wall_height,
                            fill=base_height)
    cdrawer.regular_polygon((r // 2 + pad, r // 2 + pad, r // 2), wall_sides, rotation=wall_rot, outline=w_color,
                            fill=dirt)

    # get drawing data for logic
    hdata = np.asarray(himg)
    cdata = np.asarray(cimg)

    # place houses
    n_wall_pixels = (hdata > base_height).sum()
    n_city_pixels = (cdata > 0).sum() // 3
    n_available = n_city_pixels - n_wall_pixels
    building_density = uniform(0.5, 0.8)
    n_building_pixels = int(n_available * building_density)
    chance_highrise = uniform(0.0, 0.3)

    built = 0
    while built < n_building_pixels:
        x = randint(pad, r + pad - 1)
        y = randint(pad, r + pad - 1)

        if cdata[x, y, 0] > 0 and hdata[x, y] == base_height:
            cdata[x, y] = choice([bw_color, br_color])
            hdata[x, y] = highrise_height if random() < chance_highrise else building_height
            built += 1

    return hdata, cdata


def slice_cont(
        orig,
        cmap,
        realm_number,
        water_mask,
        water_color,
        hmap_cities,
        output_dir,
        fill=10,
        zscale=5,
):
    """Slice a heightmap in z values and colorize.
    There are multiple tricks used here.
    
    Args:
        orig:           Original heightmap, expected in [0,255] uints.
        cmap:           Colormap.
        realm_number:   Realm number used to serialize height pngs.
        water_mask:          Water mask.
        water_color:    Water color.
        hmap_cities:    Cities hmap.
        output_dir:     Output_dir
        fill:           Fill water until this level, depends on zscale.
        zscale:         Divisor of 255 (max height).
    
    Outputs:
        Slices of pngs in output/hslice_{realm_number}.
    """
    orig = (orig.astype(np.float) / zscale).astype(np.uint8)
    min_val = orig.min()
    max_val = orig.max()
    bookkeeping = None
    for i in range(min_val, max_val + 1):
        new = np.zeros(orig.shape)
        c = np.where(orig == i, i, 0)

        # find spots where there is a gap in z
        new[:, :-1] += (orig[:, :-1] - c[:, 1:]) * c[:, 1:].clip(0, 1) > 1
        new[:, 1:] += (orig[:, 1:] - c[:, :-1]) * c[:, :-1].clip(0, 1) > 1
        new[:-1, :] += (orig[:-1, :] - c[1:, :]) * c[1:, :].clip(0, 1) > 1
        new[1:, :] += (orig[1:, :] - c[:-1, :]) * c[:-1, :].clip(0, 1) > 1

        # this checks if we reached the highest block
        if bookkeeping is not None:
            # also add from previous blocks
            new += bookkeeping
            # and update bookkeeping itself
            bookkeeping = orig * new.clip(0, 1)
            bookkeeping = np.where(bookkeeping <= i, 0, bookkeeping)

        # create the final mask
        new = new.clip(0, 1)
        # keep track of how much we actually have to fill
        if bookkeeping is None:
            bookkeeping = orig * new

        # add sides
        sides = np.zeros_like(new)
        sides[:, :1] = np.where(orig[:, :1] > i, 1, 0)
        sides[:, -1:] = np.where(orig[:, -1:] > i, 1, 0)
        sides[:1] = np.where(orig[:1] > i, 1, 0)
        sides[-1:] = np.where(orig[-1:] > i, 1, 0)

        final = (new + np.where(orig == i, 1, 0) + sides).clip(0, 1)

        # colorize
        output = np.tile(final, (3, 1, 1))
        output = np.transpose(output, (1, 2, 0)).astype(np.uint8)
        output = np.where(output == 1, cmap, 0)

        # add ground coloring
        sides_ground_1 = np.where(((sides > 0) & (i < orig - 1)), 1, 0)
        sides_ground_2 = np.where(((sides > 0) & (i < orig - 4)), 1, 0)
        sides_ground_1 = np.expand_dims(sides_ground_1, -1)
        sides_ground_2 = np.expand_dims(sides_ground_2, -1)
        output = np.where(sides_ground_1 == 1, [127, 127, 127], output)
        output = np.where(sides_ground_2 == 1, [100, 100, 100], output)

        # add water
        x = np.where(((i > orig) & (orig < fill) & (i < fill)), 1, 0)
        x = np.expand_dims(x, -1)
        water = np.where(
            ((output == 0) & (x == 1)),
            water_color,
            [0, 0, 0],
        )
        output = output + water

        # add rivers if also given
        x = np.where(((i == orig + 1) & (water_mask == 1) & (hmap_cities == 0) & (x[..., 0] == 0)), 1, 0)
        x = np.expand_dims(x, -1)
        water = np.where(
            x == 1,
            water_color,
            [0, 0, 0],
        )
        output = output + water

        img = PIL.Image.fromarray(output.astype(np.uint8))
        img = img.convert("RGBA")
        if not os.path.exists(f"{output_dir}/hslices_{realm_number}"):
            os.mkdir(f"{output_dir}/hslices_{realm_number}")
        img.save(f"{output_dir}/hslices_{realm_number}/{i:04d}.png")


def extract_land_sea_direction(
    cropped_land_mask, debug=False
):
    """
    This functions extract the direction land-sea so that
    the realms can be chained together.

    the main principle is to aggregate points along the edges and
    calculate the weights of the sides.

    Args:
        cropped_land_mask: bitmask of land (1) and sea (0) in cropped form
        debug: debug mode

    Returns:
        theta: land-sea orientation of the realm
    """
    
    top = np.sum(cropped_land_mask[0], dtype=np.float)
    left = np.sum(cropped_land_mask[:,0], dtype=np.float)
    bottom = np.sum(cropped_land_mask[-1], dtype=np.float)
    right = np.sum(cropped_land_mask[:,-1], dtype=np.float)

    # invert so that it points towards the sea
    vector = np.array([left-right, bottom-top])
    # normalize
    vector = vector / np.sqrt(np.sum(vector**2)+1)
    
    if vector[0] != 0.:
        direction = np.arctan2(vector[1],vector[0])
    else:
        direction = np.arctan2(vector[1],(vector[0]+0.001))

    if debug:
        print(f"direction vector: {vector}")
        print(f"direction degrees: {direction * 180 / np.pi}")
        print(f"direction pixels: {top, left, bottom, right}")

    return direction
