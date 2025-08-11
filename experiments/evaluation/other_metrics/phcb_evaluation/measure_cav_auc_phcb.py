import copy
import gc
import logging
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import wandb
import yaml
from crp.attribution import CondAttribution
from torch.utils.data import DataLoader
from sklearn import metrics
from datasets import get_dataset, get_dataset_kwargs
from model_correction import get_correction_method
from models import get_fn_model_loader
from utils.artificial_artifact import get_artifact_kwargs
from utils.distance_metrics import cosine_similarities_batch
torch.random.manual_seed(0)

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file',
                        default="config_files/clarc/funnybirds_ch/local/efficientnet_b0_PClarc_lamb1_signal_cavs_max_adam_lr0.001_identity_0.yaml")
    parser.add_argument('--plots', default=False)
    parser.add_argument('--cav_type', type=str, default=None)
    parser.add_argument('--direction_type', type=str, default=None)
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--plot_only', type=bool, default=False) # For local experiments -> dont log to wandb
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    measure_cav_auc_phcb(config)
    
    

def measure_cav_auc_phcb(config):
    pass