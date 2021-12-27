from multiprocessing import Pool
from functools import partial
import glob
from omegaconf import OmegaConf
import sys
sys.path.append('./pipeline')
from pipeline.vox_chirurgy import operate

from subprocess import Popen

paths = glob.glob("svgs/*.svg")[:5]
realm_numbers = [int(p.replace("svgs/","").replace("svgs\\","").replace(".svg","")) for p in paths]

POOL_SIZE = 2

if __name__ == '__main__':
    print(f"Found realm numbers: {realm_numbers}")
    with Pool(POOL_SIZE) as p:
        p.map(operate, realm_numbers)
