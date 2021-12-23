from multiprocessing import Pool
from functools import partial
import glob
from omegaconf import OmegaConf
from pipeline.run import run_pipeline

from subprocess import Popen

config = OmegaConf.load("pipeline/config.yaml")
f = partial(run_pipeline, config=config)

paths = glob.glob("output/height_*")[:5]


import time

POOL_SIZE = 2

if __name__ == '__main__':
    commands = []
    for p in paths:
        realm_number = p.split('_')[-1][:-4]
        commands.append(
            f"FileToVox-v1.13-win/FileToVox.exe" +
            f"--i output/hslices_{realm_number}" +
            f"-o MagicaVoxel-0.99.6.4-win64/vox/wmap_{realm_number}")
    procs = [Popen(i, shell=True) for i in commands]
    for p in procs:
        p.wait()
