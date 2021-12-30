import sys

from pyvox.parser import VoxParser, Chunk
from pyvox.writer import VoxWriter
from pyvox.models import Material

from struct import unpack_from as unpack, pack

import click
import json

@click.command()
@click.argument("realm_number")
def parse(realm_number):
    operate(realm_number)

def operate(realm_number):
    with open(f"output/flood_{realm_number}.json") as json_file:
        data = json.load(json_file)
        water_color = data["steps"][0]["water_color"]

    donor = VoxParser("voxmaps/donor.vox")
    acceptor = VoxParser(f"voxmaps/wmap_{realm_number}.vox")

    m_donor = donor.parse()
    m_acceptor = acceptor.parse()

    print(water_color)
    for i, c in enumerate(m_donor._palette):
        if c.r == 74 and c.g == 134 and c.b == 168:
            water_idx_donor = i
            break

    print(f"donor water idx {water_idx_donor}")

    best_dis = 9999
    water_idx_acceptor = 0
    # water color can sometimes we quantized differently
    for i, c in enumerate(m_acceptor._palette):
        dis = abs(c.r-water_color[0]) + abs(c.g-water_color[1]) + abs(c.b-water_color[2])
        if dis < best_dis:
            best_dis = dis
            water_idx_acceptor = i

    print(f"acceptor water idx {water_idx_acceptor}")

    d_mat = m_donor.materials[-water_idx_donor-1]
    a_mat = m_acceptor.materials[-water_idx_acceptor-1]
    m_acceptor.materials[-water_idx_acceptor-1] = Material(a_mat.id, d_mat.type, a_mat.bid, d_mat.btype, d_mat.content)

    # set correct height
    for i,r in enumerate(m_acceptor.remnants):
        try:
            if c.id == b"nTRN":
                translation_bytes = c.content.split(b"_t")[-1]
                tsize = unpack(f"i", translation_bytes)[0]
                xyz_bytes = unpack(f"i{tsize}s", translation_bytes)[1]
                # throw away old z (assert it is 256)
                x, y, _ = xyz_bytes.split(b" ")
                # put in new z
                new_xyz = b" ".join([x, y, b"16"])
                new_ntrs_bytes = pack(
                    f'i{len(new_xyz)}s',
                    len(new_xyz),
                    new_xyz
                )
                m_acceptor.remnants[i].content = b"_t".join([c.content.split(b"_t")[0], new_ntrs_bytes])
        except AttributeError:
            pass

    writer = VoxWriter("voxmaps/temp.vox", m_acceptor)
    writer.write()
    # we somehow need to do it again ..
    m_temp = VoxParser("voxmaps/temp.vox")
    writer = VoxWriter(f"voxmaps/fmap_{realm_number:04d}.vox", m_temp.parse())
    writer.write()

if __name__=="__main__":
    parse()
