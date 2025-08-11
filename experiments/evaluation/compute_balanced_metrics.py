import os
import copy
import random
from argparse import ArgumentParser
from collections import defaultdict

import torch
import wandb
from torch.utils.data import DataLoader

from datasets import get_dataset, get_dataset_kwargs, load_dataset
from experiments.evaluation.compute_metrics import compute_metrics, compute_model_scores
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_fn_model_loader
from utils.artificial_artifact import get_artifact_kwargs
from utils.helper import load_config


torch.random.manual_seed(0)
random.seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--config_file',
        default="config_files/evaluate_metrics_clean/waterbirds/local/"
                "waterbirds_resnet50d_p_artifact_landbirds0.2_"
                "p_artifact_waterbirds0.8_last_conv_baseline.yaml"
    )
    parser.add_argument('--before_correction', action="store_true")
    return parser.parse_args()


def balance_split_indices_global(
    raw_idxs,
    labels=None,
    groups=None,
    by='class',
    seed=0,
    sort_output=True
):
    """
    Given a list of global indices `raw_idxs` and a corresponding label- or
    group-array, down‐sample so that each class/group has the same number of
    examples (the size of the smallest bin). Returns a list of global indices.
    """
    rng = random.Random(seed)

    # choose key array
    if 'class' in by:
        if labels is None:
            raise ValueError("Labels must be provided for class‐balancing.")
        key_arr = labels
    elif 'group' in by:
        if groups is None:
            raise ValueError("Groups must be provided for group‐balancing.")
        key_arr = groups
    else:
        raise ValueError(f"Unknown balance key: {by!r}")

    # bucket by the key on the global indices
    buckets = defaultdict(list)
    for idx in raw_idxs:
        k = key_arr[idx]
        # if k is a tensor or numpy scalar, convert to Python scalar
        if not isinstance(k, (int, str)):
            k = k.item()
        buckets[k].append(idx)

    # debug print: show each bucket’s size
    for k, v in buckets.items():
        print(f"  class/group = {k!r}, count = {len(v)}")
    min_size = min(len(v) for v in buckets.values())
    print(f"  → sampling down to min_size = {min_size}\n")

    # sample without replacement
    balanced = []
    for v in buckets.values():
        balanced.extend(rng.sample(v, min_size))

    # return sorted (deterministic) or shuffled
    if sort_output:
        return sorted(balanced)
    else:
        rng.shuffle(balanced)
        return balanced


def evaluate_by_subset_attacked(config, before_correction):
    """Run evaluations for each split on clean vs attacked vs original datasets,
       balanced by class or group."""
    config_name = config["config_name"]
    dataset_name = config['dataset_name']
    base_attacked_classes = config.get("attacked_classes", None)
    print(f"Evaluating {config_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = config['batch_size']

    # load original dataset
    dataset = load_dataset(config, normalize_data=True)
    n_classes = len(dataset.classes)

    # model loading
    ckpt_path = (
        config['ckpt_path']
        if before_correction
        else f"{config['checkpoint_dir_corrected']}/{config_name}/last.ckpt"
    )
    model = get_fn_model_loader(model_name=config['model_name'])(
        n_class=n_classes,
        ckpt_path=ckpt_path,
        device=device
    )
    model = prepare_model_for_evaluation(model, dataset, device, config)

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


    # index splits
    splits = {
        'train': dataset.idxs_train,
        'val':   dataset.idxs_val,
        'test':  dataset.idxs_test,
    }

    for split in ['test', 'val']:
        raw_idxs = splits[split]

        for balance in ['class_balanced', 'group_balanced']:
            # get balanced global indices directly
            balanced_global = balance_split_indices_global(
                raw_idxs,
                labels=dataset.targets,
                groups=dataset.groups,
                by=balance,
                seed=0,
                sort_output=True
            )

            # subset each variant
            d_orig  = dataset.get_subset_by_idxs(balanced_global)
            d_clean = dataset_clean.get_subset_by_idxs(balanced_global)
            d_att   = dataset_attacked.get_subset_by_idxs(balanced_global)

            # prepare groups tensors
            groups_orig = torch.tensor([dataset.groups[i] for i in balanced_global])
            groups_clean = torch.tensor([dataset_clean.groups[i] for i in balanced_global])
            groups_att   = torch.tensor([dataset_attacked.groups[i] for i in balanced_global])

            # Print some info
            print(f"Split: {split}, Balance: {balance}")
            print(f"Original dataset size: {len(d_orig)}")
            print(f"Clean dataset size:    {len(d_clean)}")
            print(f"Attacked dataset size: {len(d_att)}\n")

            # compute and log metrics for original, clean, attacked
            dl_o = DataLoader(d_orig, batch_size=batch_size, shuffle=False)
            outs_o, y_o = compute_model_scores(model, dl_o, device)
            metrics_ch = compute_metrics(
                outs_o, y_o, dataset.classes,
                prefix=f"{split}_", suffix=f"_{balance}_ch",
                groups=groups_orig, attacked_classes=base_attacked_classes
            )

            dl_c = DataLoader(d_clean, batch_size=batch_size, shuffle=False)
            outs_c, y_c = compute_model_scores(model, dl_c, device)
            metrics_clean = compute_metrics(
                outs_c, y_c, dataset.classes,
                prefix=f"{split}_", suffix=f"_{balance}_clean",
                groups=groups_clean, attacked_classes=base_attacked_classes
            )

            dl_a = DataLoader(d_att, batch_size=batch_size, shuffle=False)
            outs_a, y_a = compute_model_scores(model, dl_a, device)
            metrics_attacked = compute_metrics(
                outs_a, y_a, dataset.classes,
                prefix=f"{split}_", suffix=f"_{balance}_attacked",
                groups=groups_att, attacked_classes=base_attacked_classes
            )

            if config.get('wandb_api_key'):
                wandb.log({**metrics_ch, **metrics_clean, **metrics_attacked})

            # compute accuracy drops
            acc_drop_att_clean = (
                metrics_attacked[f"{split}_accuracy_{balance}_attacked"]
                - metrics_clean[f"{split}_accuracy_{balance}_clean"]
            )
            acc_drop_att_ch = (
                metrics_attacked[f"{split}_accuracy_{balance}_attacked"]
                - metrics_ch[f"{split}_accuracy_{balance}_ch"]
            )
            acc_drop_clean_ch = (
                metrics_clean[f"{split}_accuracy_{balance}_clean"]
                - metrics_ch[f"{split}_accuracy_{balance}_ch"]
            )
            accuracy_drops = {
                f"{split}_accuracy_drop_att_{balance}_clean": acc_drop_att_clean,
                f"{split}_accuracy_drop_att_{balance}_ch":    acc_drop_att_ch,
                f"{split}_accuracy_drop_clean_{balance}_ch": acc_drop_clean_ch,
            }
            if config.get('wandb_api_key'):
                wandb.log(accuracy_drops)

            if dataset_waterbirds is not None:
                # compute metrics for waterbirds dataset
                d_waterbirds = dataset_waterbirds.get_subset_by_idxs(balanced_global)
                groups_waterbirds = torch.tensor([dataset_waterbirds.groups[i] for i in balanced_global])

                dl_w = DataLoader(d_waterbirds, batch_size=batch_size, shuffle=False)
                outs_w, y_w = compute_model_scores(model, dl_w, device)
                metrics_waterbirds = compute_metrics(
                    outs_w, y_w, dataset.classes,
                    prefix=f"{split}_", suffix=f"_{balance}_waterbirds_balanced",
                    groups=groups_waterbirds, attacked_classes=base_attacked_classes
                )

                if config.get('wandb_api_key'):
                    wandb.log(metrics_waterbirds)


def main():
    args = get_args()
    config = load_config(args.config_file)

    if config.get('wandb_api_key'):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(
            id=config['wandb_id'],
            project=config['wandb_project_name'],
            resume=True
        )

    config['config_file'] = args.config_file
    evaluate_by_subset_attacked(config, args.before_correction)


if __name__ == "__main__":
    main()
