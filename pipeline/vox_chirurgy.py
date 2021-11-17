import sys

from pyvox.parser import VoxParser
from pyvox.writer import VoxWriter
from pyvox.models import Material

import click
import json

@click.command()
@click.argument("realm_number", type=int)
def run(realm_number):

    with open(f"output/flood_{realm_number}.json") as json_file:
        data = json.load(json_file)
        water_color = data["steps"][0]["water_color"]

    donor = VoxParser("MagicaVoxel-0.99.6.4-win64/vox/donor.vox")
    acceptor = VoxParser(f"MagicaVoxel-0.99.6.4-win64/vox/wmap_{realm_number}.vox")

    m_donor = donor.parse()
    m_acceptor = acceptor.parse()

    print(water_color)
    for i, c in enumerate(m_donor._palette):
        if c.r == 74 and c.g == 134 and c.b == 168:
            water_idx_donor = i
            break

    print(f"donor water idx {water_idx_donor}")

    for i, c in enumerate(m_acceptor._palette):
        if c.r == water_color[0] and c.g == water_color[1] and c.b == water_color[2]:
            water_idx_acceptor = i
            break
    
    print(f"acceptor water idx {water_idx_acceptor}")

    d_mat = m_donor.materials[-water_idx_donor-1]
    a_mat = m_acceptor.materials[-water_idx_acceptor-1]
    m_acceptor.materials[-water_idx_acceptor-1] = Material(a_mat.id, d_mat.type, a_mat.bid, d_mat.btype, d_mat.content)

    writer = VoxWriter("MagicaVoxel-0.99.6.4-win64/vox/out.vox", m_acceptor)
    writer.write()
    # we somehow need to do it again ..
    m_temp = VoxParser("MagicaVoxel-0.99.6.4-win64/vox/out.vox")
    writer = VoxWriter(f"MagicaVoxel-0.99.6.4-win64/vox/fmap_{realm_number:04d}.vox", m_temp.parse())
    writer.write()

if __name__=="__main__":
    print("test")
    run()