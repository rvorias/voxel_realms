from multiprocessing import Pool
from functools import partial
import glob
from omegaconf import OmegaConf
from pipeline.run import run_pipeline

from subprocess import Popen

config = OmegaConf.load("pipeline/config.yaml")
f = partial(run_pipeline, config=config)

paths = glob.glob("svgs/*.svg")[:200]

import time

POOL_SIZE = 10

if __name__ == '__main__':
    start = time.time()
    print(f"Found svgs: {paths}")
    with Pool(POOL_SIZE) as p:
        p.map(f, paths)

    # commands = [f'python3 pipeline/flow.py --no-pylint run --realm_path {p} --config_path pipeline/config.yaml' for p in paths]
    # procs = [Popen(i, shell=True) for i in commands]
    # for p in procs:
    #     p.wait()

    end = time.time()
    total_time = end-start
    time_per_realm = total_time / POOL_SIZE
    print(f"Generating all will take {time_per_realm*8000/3600} hours.")