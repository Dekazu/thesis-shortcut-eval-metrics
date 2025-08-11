import os
from argparse import ArgumentParser
import copy

import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from datasets import get_dataset, get_dataset_kwargs, load_dataset
from experiments.evaluation.compute_metrics import compute_metrics, compute_model_scores
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_fn_model_loader
from utils.artificial_artifact import get_artifact_kwargs
from utils.helper import load_config

torch.random.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file', 
                        default="config_files/evaluate_metrics_clean/waterbirds/local/waterbirds_vgg16_p_artifact_landbirds0.05_p_artifact_waterbirds0.95_features.29_baseline.yaml"
    )
    parser.add_argument('--before_correction', action="store_true")
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    config = load_config(args.config_file)

    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'], project=config['wandb_project_name'], resume=True)

    config['config_file'] = args.config_file

    evaluate_by_subset_attacked(config, args.before_correction)


def evaluate_by_subset_attacked(config, before_correction):
    """ Run evaluations for each data split (train/val/test) on 3 variants of datasets:
            1. Same as training (one attacked class)
            2. Attacked (artifact in all classes)
            3. Clean (no artifacts)

    Args:
        config (dict): config for model correction run
    """
    config_name = config["config_name"]

    print(f"Evaluating {config_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = config['dataset_name']
    model_name = config['model_name']

    data_paths = config['data_paths']
    batch_size = config['batch_size']
    img_size = config.get("img_size", 224)
    artifact_type = config.get("artifact_type", None)
    binary_target = config.get("binary_target", None)
    artifact_kwargs = get_artifact_kwargs(config)
    dataset_specific_kwargs = get_dataset_kwargs(config)
    base_attacked_classes = config.get("attacked_classes", None)

    dataset = load_dataset(config, normalize_data=True)

    n_classes = len(dataset.classes)
    ckpt_path = config['ckpt_path'] if before_correction else f"{config['checkpoint_dir_corrected']}/{config_name}/last.ckpt"

    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path, device=device)
    model = prepare_model_for_evaluation(model, dataset, device, config)

    sets = {
        'train': dataset.idxs_train,
        'val': dataset.idxs_val,
        'test': dataset.idxs_test,
    }

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

    if dataset_name == "waterbirds":
        # Also with 0.5 split
        config_waterbirds = copy.deepcopy(config)
        config_waterbirds["p_artifact_landbirds"] = 0.5
        config_waterbirds["p_artifact_waterbirds"] = 0.5
        dataset_waterbirds = load_dataset(config_waterbirds)
    else:
        dataset_waterbirds = None

    if dataset_name == "isic_attacked":
        # Also with 0.5 split
        config_attacked_isic_balanced = copy.deepcopy(config)
        config_attacked_isic_balanced["p_artifact"] = 0.5
        config_attacked_isic_balanced["p_artifact_na"] = 0.5
        config_attacked_isic_balanced = load_dataset(config_attacked_isic_balanced)
    else:
        config_attacked_isic_balanced = None

    if dataset_name == "isic_attacked":
        # Also with 0.5 split
        config_attacked_isic_extreme = copy.deepcopy(config)
        config_attacked_isic_extreme["p_artifact"] = 0.8
        config_attacked_isic_extreme["p_artifact_na"] = 0.6
        config_attacked_isic_extreme = load_dataset(config_attacked_isic_extreme)
    else:
        config_attacked_isic_extreme = None


    for split in [
        'test', 
        'val'
        ]:
        split_set = sets[split]
        #split_set = split_set[:100]

        dataset_ch_split = dataset.get_subset_by_idxs(split_set)
        dataset_clean_split = dataset_clean.get_subset_by_idxs(split_set)
        dataset_attacked_split = dataset_attacked.get_subset_by_idxs(split_set)

        dataset_groups = dataset.groups[split_set]
        dataset_attacked_groups = dataset_attacked.groups[split_set]
        dataset_clean_groups = dataset_clean.groups[split_set]

        classes = dataset.classes #None

        dl = DataLoader(dataset_ch_split, batch_size=batch_size, shuffle=False)
        model_outs, y_true = compute_model_scores(model, dl, device)
        metrics = compute_metrics(model_outs, y_true, classes, prefix=f"{split}_", suffix=f"_ch", groups=dataset_groups, attacked_classes=base_attacked_classes)

        dl_attacked = DataLoader(dataset_attacked_split, batch_size=batch_size, shuffle=False)
        model_outs_attacked, y_true_attacked = compute_model_scores(model, dl_attacked, device)
        metrics_attacked = compute_metrics(model_outs_attacked, y_true_attacked, classes, prefix=f"{split}_",
                                        suffix=f"_attacked", groups=dataset_attacked_groups, attacked_classes=base_attacked_classes)

        dl_clean = DataLoader(dataset_clean_split, batch_size=batch_size, shuffle=False)
        model_outs_clean, y_true_clean = compute_model_scores(model, dl_clean, device)  
        metrics_clean = compute_metrics(model_outs_clean, y_true_clean, classes, prefix=f"{split}_",
                                        suffix=f"_clean", groups=dataset_clean_groups, attacked_classes=base_attacked_classes)

        if config.get('wandb_api_key', None):
            wandb.log({**metrics, **metrics_attacked, **metrics_clean})

        # Calculate Accuracy Drops
        accuracy_drops = {}
        acc_drop_att_clean = metrics_attacked[f'{split}_accuracy_attacked'] - metrics_clean[f'{split}_accuracy_clean']
        acc_drop_att_ch = metrics_attacked[f'{split}_accuracy_attacked'] - metrics[f'{split}_accuracy_ch']
        acc_drop_clean_ch = metrics_clean[f'{split}_accuracy_clean'] - metrics[f'{split}_accuracy_ch']

        accuracy_drops[f'{split}_accuracy_drop_att_clean'] = acc_drop_att_clean
        accuracy_drops[f'{split}_accuracy_drop_att_ch'] = acc_drop_att_ch
        accuracy_drops[f'{split}_accuracy_drop_clean_ch'] = acc_drop_clean_ch
        
        if config.get('wandb_api_key', None):
            wandb.log(accuracy_drops)

        if dataset_waterbirds is not None:
            dataset_waterbirds_split = dataset_waterbirds.get_subset_by_idxs(split_set)
            dataset_waterbirds_groups = dataset_waterbirds.groups[split_set]

            dl_waterbirds = DataLoader(dataset_waterbirds_split, batch_size=batch_size, shuffle=False)
            model_outs_waterbirds, y_true_waterbirds = compute_model_scores(model, dl_waterbirds, device)
            metrics_waterbirds = compute_metrics(model_outs_waterbirds, y_true_waterbirds, classes,
                                                prefix=f"{split}_", suffix=f"_waterbirds_balanced",
                                                groups=dataset_waterbirds_groups, attacked_classes=base_attacked_classes)

            if config.get('wandb_api_key', None):
                wandb.log(metrics_waterbirds)
        
        if config_attacked_isic_balanced is not None:
            dataset_attacked_isic_balanced_split = config_attacked_isic_balanced.get_subset_by_idxs(split_set)
            dataset_attacked_isic_balanced_groups = config_attacked_isic_balanced.groups[split_set]
            dl_attacked_isic_balanced = DataLoader(dataset_attacked_isic_balanced_split, batch_size=batch_size, shuffle=False)
            model_outs_attacked_isic_balanced, y_true_attacked_isic_balanced = compute_model_scores(
                model, dl_attacked_isic_balanced, device)
            metrics_attacked_isic_balanced = compute_metrics(
                model_outs_attacked_isic_balanced, y_true_attacked_isic_balanced, classes,
                prefix=f"{split}_", suffix=f"_attacked_isic_balanced",
                groups=dataset_attacked_isic_balanced_groups, attacked_classes=base_attacked_classes)
            if config.get('wandb_api_key', None):
                wandb.log(metrics_attacked_isic_balanced)

        if config_attacked_isic_extreme is not None:
            dataset_attacked_isic_extreme_split = config_attacked_isic_extreme.get_subset_by_idxs(split_set)
            dataset_attacked_isic_extreme_groups = config_attacked_isic_extreme.groups[split_set]
            dl_attacked_isic_extreme = DataLoader(dataset_attacked_isic_extreme_split, batch_size=batch_size, shuffle=False)
            model_outs_attacked_isic_extreme, y_true_attacked_isic_extreme = compute_model_scores(
                model, dl_attacked_isic_extreme, device)
            metrics_attacked_isic_extreme = compute_metrics(
                model_outs_attacked_isic_extreme, y_true_attacked_isic_extreme, classes,
                prefix=f"{split}_", suffix=f"_attacked_isic_extreme",
                groups=dataset_attacked_isic_extreme_groups, attacked_classes=base_attacked_classes)
            if config.get('wandb_api_key', None):
                wandb.log(metrics_attacked_isic_extreme)

if __name__ == "__main__":
    main()
