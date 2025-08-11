import copy
import logging
import os
from argparse import ArgumentParser

import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from datasets import load_dataset
from utils.plots import visualize_dataset
from utils.helper import load_config
from collections import Counter

torch.multiprocessing.set_sharing_strategy('file_system')

logging.basicConfig(
    level=logging.INFO,  # Enables INFO-level logging
    format='%(levelname)s: %(message)s'  # Optional: controls output format
)

def get_args():
    parser = ArgumentParser(description='Visualize Dataset', )
    parser.add_argument('--config_file', 
                        default="config_files/evaluate_metrics_clean/waterbirds/local/waterbirds_resnet50d_p_artifact_landbirds0.05_p_artifact_waterbirds0.95_last_conv_baseline.yaml"
                        )
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    config = load_config(args.config_file)

    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'], project=config['wandb_project_name'], resume=True)

    start_training(config)


def start_training(config):
    """ Starts training for given config file.

    Args:
        config (dict): Dictionary with config parameters for training.
        config_name (str): Name of given config
    """

    dataset_name = config['dataset_name']
    batch_size = config['batch_size']

    # Attack Details
    attacked_classes = config.get('attacked_classes', [])

    

    dataset= load_dataset(config)


    idxs_val = dataset.idxs_val
    dataset_val = dataset.get_subset_by_idxs(idxs_val)
    dataset_test = dataset.get_subset_by_idxs(dataset.idxs_test)
    dataset_train = dataset.get_subset_by_idxs(dataset.idxs_train)

    dataset_val.do_augmentation = False
    dataset_test.do_augmentation = False

    dl_val_dict = {'val': DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)}

    if (len(attacked_classes) > 0) or ('waterbirds' in dataset_name):

        config_clean = copy.deepcopy(config)
        config_clean["p_artifact"] = 0.0
        if config_clean.get("p_artifact_na", None) is not None:
            config_clean["p_artifact_na"] = 0.0 
        if config_clean.get("p_artifact_landbirds", None) is not None:
            config_clean["p_artifact_landbirds"] = 0.0
        if config_clean.get("p_artifact_waterbirds", None) is not None:
            config_clean["p_artifact_waterbirds"] = 0.0
        dataset_clean= load_dataset(config_clean)

        all_classes = dataset.classes 

        config_attacked = copy.deepcopy(config)
        config_attacked["attacked_classes"] = all_classes
        config_attacked["p_artifact"] = 1.0
        if config_attacked.get("p_artifact_na", None) is not None:
            config_attacked["p_artifact_na"] = 1.0 
        if config_attacked.get("p_artifact_landbirds", None) is not None:
            config_attacked["p_artifact_landbirds"] = 1.0
        if config_attacked.get("p_artifact_waterbirds", None) is not None:
            config_attacked["p_artifact_waterbirds"] = 1.0
        dataset_attacked = load_dataset(config_attacked)

        dataset_test_clean = dataset_clean.get_subset_by_idxs(dataset.idxs_test)
        dataset_test_attacked = dataset_attacked.get_subset_by_idxs(dataset.idxs_test)
        
        dl_val_dict['test_clean'] = DataLoader(dataset_test_clean, batch_size=batch_size, shuffle=False, num_workers=8)
        dl_val_dict['test_attacked'] = DataLoader(dataset_test_attacked, batch_size=batch_size, shuffle=False, num_workers=8)
            
    

    visualization_path = f"results/datasets_visualized/{dataset_name}/{config['config_name']}"
    if not os.path.exists(visualization_path):
        os.makedirs(visualization_path)

    for format in ['png', 'svg']:

        os.makedirs(visualization_path, exist_ok=True)
        start_idx = max(0, dataset_val.artifact_ids[0] - 10) if hasattr(dataset_val, "artifact_ids") else 0
        fname_normal = f"dataset_attacked{attacked_classes[0]}_normal.{format}" if len(attacked_classes) > 0 else f"dataset_normal.{format}"
        visualize_dataset(dataset_val, f"{visualization_path}/{fname_normal}", start_idx)
        fname_clean = f"dataset_attacked{attacked_classes[0]}_clean.{format}" if len(attacked_classes) > 0 else f"dataset_clean.{format}"
        visualize_dataset(dl_val_dict['test_clean'].dataset, f"{visualization_path}/{fname_clean}", start_idx)
        fname_attacked = f"dataset_attacked{attacked_classes[0]}_attacked.{format}" if len(attacked_classes) > 0 else f"dataset_attacked.{format}"
        visualize_dataset(dl_val_dict['test_attacked'].dataset, f"{visualization_path}/{fname_attacked}", start_idx)

        if config.get('wandb_api_key', None) and format == 'png':
            wandb.log({"dataset_visualized_ch": wandb.Image(f"{visualization_path}/{fname_normal}")})
            wandb.log({"dataset_visualized_clean": wandb.Image(f"{visualization_path}/{fname_clean}")})
            wandb.log({"dataset_visualized_attacked": wandb.Image(f"{visualization_path}/{fname_attacked}")})



    # ---- Dataset Statistics ----

    # Count Dataset Samples
    total = len(dataset_val) + len(dataset_test) + len(dataset_train)
    logging.info(f"Dataset {dataset_name} contains {len(dataset_val)} validation samples. This makes up {len(dataset_val) / total:.2%} of the dataset.")
    logging.info(f"Dataset {dataset_name} contains {len(dataset_test)} test samples. This makes up {len(dataset_test) / total:.2%} of the dataset.")
    logging.info(f"Dataset {dataset_name} contains {len(dataset_train)} training samples. This makes up {len(dataset_train) / total:.2%} of the dataset.")

    # Count Classes
    classes = dataset.classes
    logging.info(f"Dataset {dataset_name} contains {len(classes)} classes: {classes}")

    split_group_tensors = {
        "train": dataset_train.groups[dataset.idxs_train],
        "validation": dataset_val.groups[dataset.idxs_val],
        "test": dataset_test.groups[dataset.idxs_test],
    }

    # Count and log samples per group for each split
    for split_name, group_tensor in split_group_tensors.items():
        # Convert tensor to list of scalars
        group_list = group_tensor.tolist()  # or [g.item() for g in group_tensor]
        group_counts = Counter(group_list)

        logging.info(f"Group distribution in {split_name} split:")
        for group, count in sorted(group_counts.items()):
            logging.info(f"  - Group {group}: {count} samples")


if __name__ == "__main__":
    main()
