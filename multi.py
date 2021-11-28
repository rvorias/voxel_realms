from multiprocessing import Pool
from functools import partial
import glob
from omegaconf import OmegaConf
from pipeline.run import run_pipeline

from subprocess import Popen

config = OmegaConf.load("pipeline/config.yaml")
f = partial(run_pipeline, config=config)

paths = glob.glob("svgs/*.svg")

if __name__ == '__main__':
    print(f"Found svgs: {paths}")
    # with Pool(2) as p:
    #     p.map(f, paths)

    commands = [f'python3 pipeline/flow.py --no-pylint run --realm_path {p} --config_path pipeline/config.yaml' for p in paths]
    procs = [Popen(i, shell=True) for i in commands]
    for p in procs:
        p.wait()