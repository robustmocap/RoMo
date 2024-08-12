from argparse import ArgumentParser
import argparse
import os
import json
from omegaconf import OmegaConf


def parse_args_from_yaml():
    parser = ArgumentParser()
    parser.add_argument('--cfg_file', type=str, nargs="+")
    params = parser.parse_args()

    base_config_path = os.path.join(os.path.dirname(params.cfg_file[0]), "base.yaml")
    if os.path.exists(base_config_path):
        base_cfg = OmegaConf.load(base_config_path)
    else:
        base_cfg = {}

    if len(params.cfg_file) == 1:
        network_cfg = OmegaConf.load(params.cfg_file[0])
        cfg = OmegaConf.merge(base_cfg, network_cfg)
    elif len(params.cfg_file) == 2:
        network_cfg = OmegaConf.load(params.cfg_file[0])
        guidance_cfg = OmegaConf.load(params.cfg_file[1])
        cfg = OmegaConf.merge(base_cfg, network_cfg)
        cfg = OmegaConf.merge(cfg, guidance_cfg)
    else:
        raise ValueError
    cfg.LOG.save_dir = os.path.join(cfg.LOG.save_dir, cfg.NAME)

    return cfg
