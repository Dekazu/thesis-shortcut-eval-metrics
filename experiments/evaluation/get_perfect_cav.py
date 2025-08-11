#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full pipeline: generate a backgrounds dataset from an existing Waterbirds dataset,
and get the CAVs for the backgrounds.
"""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import logging
import random
import copy
from bisect import bisect_right
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
import torchvision.transforms as T

from utils.helper import load_config
from models import get_fn_model_loader
from datasets import load_dataset  # existing loader for Waterbirds
from experiments.preprocessing.global_collect_relevances_and_activations import run_collect_relevances_and_activations
from utils.cav_utils import get_cav_from_model


class WrappedSubset(Subset):
    def get_target(self, idx):
        return self.dataset.get_target(self.indices[idx])
    
    def get_subset_by_idxs(self, idxs):
        """
        Returns a new WrappedSubset with the given indices.
        """
        # idxs must be local indices in this subset
        return WrappedSubset(self.dataset, [self.indices[i] for i in idxs])

class ConcatWithSubset(ConcatDataset):
    def get_subset_by_idxs(self, idxs):
        return WrappedSubset(self, idxs)

    def get_target(self, idx):
        # same as you already have
        ds_idx = bisect_right(self.cumulative_sizes, idx)
        if ds_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[ds_idx - 1]
        return self.datasets[ds_idx].get_target(local_idx)

# -----------------------------------------------------------------------------
# Create backgrounds dataset from Waterbirds  
# -----------------------------------------------------------------------------
def crop_and_resize(source_img: Image.Image, target_size: tuple) -> Image.Image:
    """
    Center-crop or expand+crop source_img so it exactly matches target_size.
    """
    sw, sh = source_img.size
    tw, th = target_size

    # scale up if too small
    if sw < tw or sh < th:
        scale = max(tw / sw, th / sh)
        tmp = source_img.resize(
            (int(sw * scale) + 1, int(sh * scale) + 1),
            Image.Resampling.LANCZOS
        )
        return crop_and_resize(tmp, target_size)

    # Crop to target aspect
    src_aspect = sw / sh
    tgt_aspect = tw / th
    if src_aspect > tgt_aspect:
        # too wide: crop left/right
        new_w = int(tgt_aspect * sh)
        left = (sw - new_w) // 2
        box = (left, 0, left + new_w, sh)
    else:
        # too tall: crop top/bottom
        new_h = int(sw / tgt_aspect)
        top = (sh - new_h) // 2
        box = (0, top, sw, top + new_h)

    cropped = source_img.crop(box)
    return cropped.resize((tw, th), Image.Resampling.LANCZOS)

def generate_backgrounds_from_places(
    waterbirds_root: str,
    backgrounds_root: str,
    places_dir: str,
    image_size: int = 224
) -> None:
    """
    Given an existing Waterbirds dataset folder (with metadata.csv),
    generate a parallel 'backgrounds' dataset where each image is just the
    background (no bird), cropped+resized to image_size.
    """
    meta_csv = os.path.join(waterbirds_root, "metadata.csv")
    if not os.path.isfile(meta_csv):
        raise FileNotFoundError(f"{meta_csv} not found; please generate Waterbirds first.")
    df = pd.read_csv(meta_csv)

    os.makedirs(backgrounds_root, exist_ok=True)
    df.to_csv(os.path.join(backgrounds_root, "metadata.csv"), index=False)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating backgrounds"):
        place_fname = row["place_filename"]
        # try absolute path first
        if os.path.isabs(place_fname) and os.path.isfile(place_fname):
            src_path = place_fname
        else:
            src_path = os.path.join(places_dir, place_fname.lstrip("/"))
            if not os.path.isfile(src_path):
                raise FileNotFoundError(f"Cannot find place image {src_path}")

        with Image.open(src_path).convert("RGB") as img:
            bg = crop_and_resize(img, (image_size, image_size))

        out_path = os.path.join(backgrounds_root, row["img_filename"])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        bg.save(out_path, optimize=True, quality=90)

class BackgroundsDataset(Dataset):
    """
    Torch Dataset mirroring the Waterbirds metadata.csv but returns only
    the cropped background image (no bird).
    """
    classes = ['landbird', 'waterbird']
    def __init__(self, root: str, image_size: int = 224, normalize: bool = True, place_dir: str = None):
        meta = pd.read_csv(os.path.join(root, "metadata.csv"))
        self.root = root
        self.meta = meta
        self.size = image_size
        self.place_dir = place_dir
        transforms = [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor()
        ]
        if normalize:
            transforms.append(T.Normalize([0.5]*3, [0.5]*3))
        self.transform = T.Compose(transforms)

        self.idxs_train = meta.index[meta["split"]==0].tolist()
        self.idxs_val   = meta.index[meta["split"]==1].tolist()
        self.idxs_test  = meta.index[meta["split"]==2].tolist()

        ids_w = meta.index[meta["place"]==1].tolist()
        ids_l = meta.index[meta["place"]==0].tolist()
        self.sample_ids_by_artifact = {
            "waterbackground": ids_w,
            "landbackground":  ids_l,
            "background":      ids_w + ids_l
        }

        self.targets = meta["y"].values

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img = Image.open(f"{self.root}/{row['img_filename']}")
        img = img.convert("RGB")
        img = self.transform(img)
        target = self.get_target(idx)
        target_tensor = torch.tensor(target, dtype=torch.long)
        return img, target_tensor
    
    def get_target(self, idx):
        return self.targets[idx]
    
    def get_subset_by_idxs(self, idxs):
        """
        Return a torch.utils.data.Subset of this dataset containing only
        the examples at the specified indices.
        """
        return Subset(self, idxs)



def create_backgrounds_dataset(
    waterbirds_root,
    places_dir: str,
    image_size: int = 224,
    normalize: bool = True
) -> BackgroundsDataset:
    """
    Takes the path to an existing waterbirds dataset folder,
    generates a 'backgrounds' sibling folder if needed, and returns the Dataset.
    """
    parent, wb_folder = os.path.split(str(waterbirds_root).rstrip("/"))
    suffix = wb_folder.split("waterbirds_")[-1]
    bg_folder = f"backgrounds_{suffix}"
    bg_root = os.path.join(parent, bg_folder)

    if not os.path.isdir(bg_root):
        logging.info(f"Generating backgrounds dataset at {bg_root}")
        generate_backgrounds_from_places(waterbirds_root, bg_root, places_dir, image_size)

    return BackgroundsDataset(bg_root, image_size=image_size, normalize=normalize, place_dir=places_dir)


# -----------------------------------------------------------------------------
# Get CAV function (Preprocessing only)
# -----------------------------------------------------------------------------

def get_background_cav(
    config: dict,
    before_correction: bool, 
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config["device"] = device

    wb_ds = load_dataset(config, normalize_data=True)
    wb_root = getattr(wb_ds, "path", None) or getattr(wb_ds, "root", None)

    bg_ds = create_backgrounds_dataset(
        wb_root,
        config.get("places_dir", config.get("cub_places_path")[0] + "/data_256"),
        image_size=config.get("image_size", 224),
        normalize=True
    )

    ckpt = config["ckpt_path"] if before_correction else os.path.join(
        config["checkpoint_dir_corrected"], config["config_name"], "last.ckpt"
    )
    model = get_fn_model_loader(config["model_name"])(
        n_class=len(wb_ds.classes), ckpt_path=ckpt, device=device
    ).to(device)

    bg_config = copy.deepcopy(config)
    bg_config['dataset_name'] = "backgrounds"

    run_preprocessing(bg_config, bg_ds)

    return bg_config, bg_ds

def get_perfect_cav(
    config: dict,
    before_correction: bool,
) -> dict:
    # 1) Device + base dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = copy.deepcopy(config)
    cfg["device"] = device

    base_ds = load_dataset(cfg, normalize_data=True)
    full_idxs = base_ds.idxs_train

    if "imagenet" in cfg['dataset_name'].lower():
        subsampling = False
        n_total = len(full_idxs) 
    else:
        subsampling = False
        n_total = len(full_idxs)

    # 2) Draw your shared subsample
    if subsampling:
        random.seed(0)
        per_cls = n_total // len(base_ds.classes)
        chosen = []
        for cls in range(len(base_ds.classes)):
            cls_idxs = [i for i in full_idxs if base_ds.get_target(i) == cls]
            chosen += random.sample(cls_idxs, min(per_cls, len(cls_idxs)))
    else:
        chosen = full_idxs


    # 3) Load attacked & clean *full* datasets
    cfg_att = copy.deepcopy(cfg)
    cfg_att.update(p_artifact=1.0, attacked_classes=base_ds.classes)
    ds_att_full = load_dataset(cfg_att, normalize_data=True)

    sample_ids_by_artifact = ds_att_full.sample_ids_by_artifact

    cfg_cln = copy.deepcopy(cfg)
    cfg_cln.update(p_artifact=0.0, p_artifact_na=0.0)
    ds_cln_full = load_dataset(cfg_cln, normalize_data=True)

    # 4) Wrap each in your WrappedSubset
    ds_att = WrappedSubset(ds_att_full, chosen)
    ds_clean = WrappedSubset(ds_cln_full, chosen)

    # 5) Concat with your ConcatWithSubset
    ds = ConcatWithSubset([ds_att, ds_clean])
    # patch metadata just like before
    ds.classes = base_ds.classes
    ds.idxs_train = list(range(len(ds)))
    ds.idxs_val = []
    ds.idxs_test = []

    ds.sample_ids_by_artifact = sample_ids_by_artifact

    # 6) Preprocess once (skips if files exist)
    cfg_pp = copy.deepcopy(cfg)
    cfg_pp["dataset_name"] = f"perfect_cav_{cfg['dataset_name']}"
    run_preprocessing(cfg_pp, ds)

    return cfg_pp, ds

# -----------------------------------------------------------------------------
# CLI & Main
# -----------------------------------------------------------------------------
def get_args():
    p = ArgumentParser()
    p.add_argument(
        '--config_file',
        default="config_files/evaluate_metrics_clean/isic_attacked_clean/local/isic_attacked_clean_vgg16_p_artifact0.3_p_artifact_na0.05_features.29_baseline.yaml",
        help="YAML config for TCAV eval"
    )
    p.add_argument('--before_correction', action='store_true', help="Use pre-correction checkpoint")
    return p.parse_args()


def main():
    args = get_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s | %(message)s", level=logging.INFO)
    cfg = load_config(args.config_file)

    if cfg["dataset_name"] in ["isic_attacked", "imagenet_attacked"]:
        cfg, ds = get_perfect_cav(cfg, args.before_correction)
    elif cfg["dataset_name"] in ["waterbirds"]:
        cfg, ds = get_background_cav(cfg, args.before_correction)
    else:
        raise ValueError(f"Unsupported dataset {cfg['dataset_name']} for CAV generation.")
    
    # Get CAVs
    cfg["cav_scope"] = None

    if "resnet" in cfg['model_name']:
        layer_names = ["identity_2", "last_conv"]
    elif "vgg" in cfg['model_name']:
        layer_names = ["features.27", "features.28", "features.29"]

    for layer in layer_names:
        cfg["layer_name"] = layer

        # Get CAVs for the perfect CAV dataset
        logging.info(f"Running calc for CAVs on layer {layer} with config: {cfg}")
        cav_a = get_cav_from_model(
            model=get_fn_model_loader(cfg['model_name'])(n_class=len(ds.classes), ckpt_path=cfg["ckpt_path"], device=cfg["device"]).to(cfg["device"]),
            dataset=ds,
            config=cfg,
            artifact=cfg['artifact_type'],
            mode="cavs_max",
            store_cav=True
        )

        cav_rel = get_cav_from_model(
            model=get_fn_model_loader(cfg['model_name'])(n_class=len(ds.classes), ckpt_path=cfg["ckpt_path"], device=cfg["device"]).to(cfg["device"]),
            dataset=ds,
            config=cfg,
            artifact=cfg['artifact_type'],
            mode="rel",
            store_cav=True
        )




# -----------------------------------------------------------------------------
# Preprocessing & Attribution
# -----------------------------------------------------------------------------
def run_preprocessing(config, dataset):
    # Für ImageNet nur Klasse 0, sonst alle Klassen
    ds_name = config['dataset_name'].lower()
    class_indices = list(range(len(dataset.classes)))
    for class_idx in class_indices:
        run_collect_relevances_and_activations({**config,
                                                'class_idx': class_idx,
                                                'split': 'all'},
                                               True, dataset)

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

if __name__ == '__main__':
    main()







