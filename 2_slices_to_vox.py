from multiprocessing import Pool
from functools import partial
import glob
from omegaconf import OmegaConf
from pipeline.run import run_pipeline

from subprocess import Popen

config = OmegaConf.load("pipeline/config.yaml")
f = partial(run_pipeline, config=config)

paths = glob.glob("output/height_*")

import os
if os.name == 'nt':
    FTV_EXEC = "FileToVox\\FileToVox.exe"
else:
    FTV_EXEC = "FileToVox-v1.13-win/FileToVox.exe"

POOL_SIZE = 4

if __name__ == '__main__':
    commands = []
    for p in paths:
        realm_number = p.split('_')[-1][:-4]
        commands.append(
            f"{FTV_EXEC} " +
            f"--i output/hslices_{realm_number} " +
            f"--o vox/wmap_{realm_number}")
    print(commands[:3])
    procs = [Popen(i, shell=True) for i in commands]
    for p in procs:
        p.wait()
