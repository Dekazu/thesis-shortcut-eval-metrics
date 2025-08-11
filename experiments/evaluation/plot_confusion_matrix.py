import os
from argparse import ArgumentParser

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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tempfile


torch.random.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file', 
                        default="config_files/evaluate_metrics_clean/isic_attacked/local/isic_attacked_vgg16_p_artifact0.3_p_artifact_na0.05_features.29_baseline.yaml"
                        )
    parser.add_argument('--before_correction', action="store_true")
    parser.add_argument("--results_dir", default="results/confusion_matrices", type=str)
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    config = load_config(args.config_file)

    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'], project=config['wandb_project_name'], resume=True)

    config['config_file'] = args.config_file

    evaluate_by_subset_attacked(config, args.before_correction, args.results_dir)


def evaluate_by_subset_attacked(config, before_correction, path):
    """ Run evaluations for each data split (train/val/test) on 3 variants of datasets:
            1. Same as training (one attacked class)
            2. Attacked (artifact in all classes)
            3. Clean (no artifacts)

    Args:
        config (dict): config for model correction run
    """
    config_name = config["config_name"]
    dataset_name = config['dataset_name']
    base_path = f"{path}/{dataset_name}/{config_name}"

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

    if config.get('p_artifact_na', None) is not None:
        dataset_specific_kwargs['p_artifact_na'] = 0.0
    if config.get('p_artifact_landbirds', None) is not None:
        dataset_specific_kwargs['p_artifact_landbirds'] = 0.0
    if config.get('p_artifact_waterbirds', None) is not None:
        dataset_specific_kwargs['p_artifact_waterbirds'] = 0.0
    dataset_clean = get_dataset(dataset_name)(data_paths=data_paths,
                                              normalize_data=True,
                                              attacked_classes=[],
                                              p_artifact = 0.5 if "waterbirds" in dataset_name else 0.0,
                                              binary_target=binary_target,
                                              image_size=img_size,
                                              **artifact_kwargs, **dataset_specific_kwargs)
    
    if dataset_name == "isic_attacked":
        # Also with 0.5 split
        import copy
        config_attacked_isic_balanced = copy.deepcopy(config)
        config_attacked_isic_balanced["p_artifact"] = 0.5
        config_attacked_isic_balanced["p_artifact_na"] = 0.5
        config_attacked_isic_balanced = load_dataset(config_attacked_isic_balanced)
    else:
        config_attacked_isic_balanced = None

    if dataset_name == "isic_attacked":
        # Also with 0.5 split
        import copy
        config_attacked_isic_extreme = copy.deepcopy(config)
        config_attacked_isic_extreme["p_artifact"] = 0.8
        config_attacked_isic_extreme["p_artifact_na"] = 0.6
        config_attacked_isic_extreme = load_dataset(config_attacked_isic_extreme)
    else:
        config_attacked_isic_extreme = None

    if "imagenet" in dataset_name:
        all_classes = list(dataset.label_map.keys())
        if config.get("subset_correction", False):
            sets['test'] = sets['test'][::10]
            sets['val'] = sets['val'][::10]
    else:
        all_classes = dataset.classes if "isic" in config["dataset_name"] else range(len(dataset.classes))


    if config.get('p_artifact_landbirds', None) is not None:
        dataset_specific_kwargs['p_artifact_landbirds'] = 1.0
    if config.get('p_artifact_waterbirds', None) is not None:
        dataset_specific_kwargs['p_artifact_waterbirds'] = 1.0
    dataset_attacked = get_dataset(dataset_name)(data_paths=data_paths,
                                                 normalize_data=True,
                                                 p_artifact=1.0,
                                                 image_size=img_size,
                                                 artifact_type=artifact_type,
                                                 binary_target=binary_target,
                                                 attacked_classes=all_classes,
                                                 **artifact_kwargs, **dataset_specific_kwargs)
    for split in [
        'val',
        'test', 
        ]:
        split_set = sets[split]
        #split_set = split_set[:100]

        dataset_ch_split = dataset.get_subset_by_idxs(split_set)
        dataset_clean_split = dataset_clean.get_subset_by_idxs(split_set)
        dataset_attacked_split = dataset_attacked.get_subset_by_idxs(split_set)

        dataset_groups = dataset.groups[split_set]
        dataset_attacked_groups = dataset_attacked.groups[split_set]
        dataset_clean_groups = dataset_clean.groups[split_set]

        dl_clean = DataLoader(dataset_clean_split, batch_size=batch_size, shuffle=False)
        model_outs_clean, y_true_clean = compute_model_scores(model, dl_clean, device)

        dl = DataLoader(dataset_ch_split, batch_size=batch_size, shuffle=False)
        model_outs, y_true = compute_model_scores(model, dl, device)

        dl_attacked = DataLoader(dataset_attacked_split, batch_size=batch_size, shuffle=False)
        model_outs_attacked, y_true_attacked = compute_model_scores(model, dl_attacked, device)

        classes = dataset.classes
        if dataset_name == "isic_attacked":
            # Remove UNK class from ISIC dataset
            classes = [cls for cls in classes if cls != "UNK"]
        log_wandb = True if config.get('wandb_api_key', None) else False
        path = f"{base_path}/{split}"

        for normalize in [True, False, None]:
            plot_confusion_matrix(y_true, model_outs.argmax(1), normalize=normalize, labels=classes, title_str=f"{split}_confusion_matrix_ch", log_to_wandb=log_wandb, path=f"{path}/ch")
            plot_confusion_matrix(y_true_attacked, model_outs_attacked.argmax(1), normalize=normalize, labels=classes, title_str=f"{split}_confusion_matrix_attacked", log_to_wandb=log_wandb, path=f"{path}/attacked")
            plot_confusion_matrix(y_true_clean, model_outs_clean.argmax(1), normalize=normalize, labels=classes, title_str=f"{split}_confusion_matrix_clean", log_to_wandb=log_wandb, path=f"{path}/clean")

            for group in np.unique(dataset_groups):
                plot_confusion_matrix(
                    y_true, 
                    model_outs.argmax(1), 
                    groups=dataset_groups, 
                    group_to_plot=group, 
                    normalize=normalize, 
                    labels=classes, 
                    title_str=f"{split}_confusion_matrix_ch",
                    log_to_wandb=log_wandb,
                    path=f"{path}/ch/groups"
                )
                plot_confusion_matrix(
                    y_true_attacked, 
                    model_outs_attacked.argmax(1), 
                    groups=dataset_attacked_groups, 
                    group_to_plot=group, 
                    normalize=normalize, 
                    labels=classes, 
                    title_str=f"{split}_confusion_matrix_attacked",
                    log_to_wandb=log_wandb,
                    path= f"{path}/attacked/groups"
                )
                plot_confusion_matrix(
                    y_true_clean, 
                    model_outs_clean.argmax(1), 
                    groups=dataset_clean_groups, 
                    group_to_plot=group, 
                    normalize=normalize, 
                    labels=classes, 
                    title_str=f"{split}_confusion_matrix_clean",
                    log_to_wandb=log_wandb,
                    path= f"{path}/clean/groups"
                )


        # Plot CM for Artefact vs no Artefact Samples
        dataset_groups_artifact = np.array(['Art' if x % 2 == 1 else 'No_Art' for x in dataset_groups])
        dataset_attacked_groups_artifact = np.array(['Art' if x % 2 == 1 else 'No_Art' for x in dataset_attacked_groups])
        dataset_clean_groups_artifact = np.array(['Art' if x % 2 == 1 else 'No_Art' for x in dataset_clean_groups])
        for normalize in [True, False, None]:
            for group in np.unique(dataset_groups_artifact):
                plot_confusion_matrix(
                    y_true, 
                    model_outs.argmax(1), 
                    groups=dataset_groups_artifact, 
                    group_to_plot=group, 
                    normalize=normalize, 
                    labels=classes, 
                    title_str=f"{split}_confusion_matrix_ch",
                    log_to_wandb=log_wandb,
                    path= f"{path}/ch/Artifact_vs_no_Artifact"
                )
                plot_confusion_matrix(
                    y_true_attacked, 
                    model_outs_attacked.argmax(1), 
                    groups=dataset_attacked_groups_artifact, 
                    group_to_plot=group, 
                    normalize=normalize, 
                    labels=classes, 
                    title_str=f"{split}_confusion_matrix_attacked",
                    log_to_wandb=log_wandb,
                    path=f"{path}/attacked/Artifact_vs_no_Artifact"
                )
                plot_confusion_matrix(
                    y_true_clean, 
                    model_outs_clean.argmax(1), 
                    groups=dataset_clean_groups_artifact, 
                    group_to_plot=group, 
                    normalize=normalize, 
                    labels=classes, 
                    title_str=f"{split}_confusion_matrix_clean",
                    log_to_wandb=log_wandb,
                    path=f"{path}/clean/Artifact_vs_no_Artifact"
                )
        
        # Plot CM for Bias Aligned vs Bias Conflicting 
        if dataset_name == "isic_attacked":
            attacked_classes = config.get('attacked_classes', [])
            bias_aligned_groups = [0, 3, 4, 6, 8, 10, 12, 14]
            bias_conflicting_groups = [1, 2, 5, 7, 9, 11, 13, 15]
        elif dataset_name == "waterbirds":
            bias_aligned_groups = [0, 3]
            bias_conflicting_groups = [1, 2]
        elif dataset_name == "imagenet_attacked":
            bias_aligned_groups = [1, 2, 4, 6, 8, 10, 12, 14]
            bias_conflicting_groups = [0, 3, 5, 7, 9, 11, 13, 15]

        dataset_groups_bias_aligned = np.array(['Bias_Aligned' if x in bias_aligned_groups else 'Bias_Conflicting' for x in dataset_groups])
        dataset_attacked_groups_bias_aligned = np.array(['Bias_Aligned' if x in bias_aligned_groups else 'Bias_Conflicting' for x in dataset_attacked_groups])
        dataset_clean_groups_bias_aligned = np.array(['Bias_Aligned' if x in bias_aligned_groups else 'Bias_Conflicting' for x in dataset_clean_groups])
        for normalize in [True, False, None]:
            for group in np.unique(dataset_groups_bias_aligned):
                plot_confusion_matrix(
                    y_true, 
                    model_outs.argmax(1), 
                    groups=dataset_groups_bias_aligned, 
                    group_to_plot=group, 
                    normalize=normalize, 
                    labels=classes, 
                    title_str=f"{split}_confusion_matrix_ch",
                    log_to_wandb=log_wandb,
                    path=f"{path}/ch/bias_aligned_vs_conflicting"
                )
                plot_confusion_matrix(
                    y_true_attacked, 
                    model_outs_attacked.argmax(1), 
                    groups=dataset_attacked_groups_bias_aligned, 
                    group_to_plot=group, 
                    normalize=normalize, 
                    labels=classes, 
                    title_str=f"{split}_confusion_matrix_attacked",
                    log_to_wandb=log_wandb,
                    path=f"{path}/attacked/bias_aligned_vs_conflicting"
                )
                plot_confusion_matrix(
                    y_true_clean, 
                    model_outs_clean.argmax(1), 
                    groups=dataset_clean_groups_bias_aligned, 
                    group_to_plot=group, 
                    normalize=normalize, 
                    labels=classes, 
                    title_str=f"{split}_confusion_matrix_clean",
                    log_to_wandb=log_wandb,
                    path=f"{path}/clean/bias_aligned_vs_conflicting"
                )


        if config_attacked_isic_balanced is not None:
            dataset_attacked_balanced = config_attacked_isic_balanced
            dl_attacked_balanced = DataLoader(dataset_attacked_balanced.get_subset_by_idxs(split_set), batch_size=batch_size, shuffle=False)
            model_outs_attacked_balanced, y_true_attacked_balanced = compute_model_scores(model, dl_attacked_balanced, device)

            for normalize in [True, False, None]:
                plot_confusion_matrix(y_true_attacked_balanced, model_outs_attacked_balanced.argmax(1), normalize=normalize, labels=classes, title_str=f"{split}_confusion_matrix_attacked_balanced", log_to_wandb=log_wandb, path=f"{path}/special/balanced")

            # Plot CM for Artefact vs no Artefact Samples
            dataset_attacked_groups_artifact_balanced = np.array(['Art' if x % 2 == 1 else 'No_Art' for x in dataset_attacked_balanced.groups[split_set]])
            for normalize in [True, False, None]:
                for group in np.unique(dataset_attacked_groups_artifact_balanced):
                    plot_confusion_matrix(
                        y_true_attacked_balanced, 
                        model_outs_attacked_balanced.argmax(1), 
                        groups=dataset_attacked_groups_artifact_balanced, 
                        group_to_plot=group, 
                        normalize=normalize, 
                        labels=classes, 
                        title_str=f"{split}_confusion_matrix_attacked_balanced",
                        log_to_wandb=log_wandb,
                        path=f"{path}/special/balanced/Artifact_vs_no_Artifact"
                    )
        
        if config_attacked_isic_extreme is not None:
            dataset_attacked_extreme = config_attacked_isic_extreme
            dl_attacked_extreme = DataLoader(dataset_attacked_extreme.get_subset_by_idxs(split_set), batch_size=batch_size, shuffle=False)
            model_outs_attacked_extreme, y_true_attacked_extreme = compute_model_scores(model, dl_attacked_extreme, device)

            for normalize in [True, False, None]:
                plot_confusion_matrix(y_true_attacked_extreme, model_outs_attacked_extreme.argmax(1), normalize=normalize, labels=classes, title_str=f"{split}_confusion_matrix_attacked_extreme", log_to_wandb=log_wandb, path=f"{path}/special/extreme")

            # Plot CM for Artefact vs no Artefact Samples
            dataset_attacked_groups_artifact_extreme = np.array(['Art' if x % 2 == 1 else 'No_Art' for x in dataset_attacked_extreme.groups[split_set]])
            for normalize in [True, False, None]:
                for group in np.unique(dataset_attacked_groups_artifact_extreme):
                    plot_confusion_matrix(
                        y_true_attacked_extreme, 
                        model_outs_attacked_extreme.argmax(1), 
                        groups=dataset_attacked_groups_artifact_extreme, 
                        group_to_plot=group, 
                        normalize=normalize, 
                        labels=classes, 
                        title_str=f"{split}_confusion_matrix_attacked_extreme",
                        log_to_wandb=log_wandb,
                        path=f"{path}/special/extreme/Artifact_vs_no_Artifact"
                    )

def plot_confusion_matrix(y_true, y_pred, groups=None, group_to_plot=None, normalize=None, labels=None, cmap="Blues",title_str="confusion_matrix", log_to_wandb=False, path="results/plots"):
    """
    Plots a confusion matrix with optional group-based filtering and normalization.
    Optionally logs the figure to wandb if log_to_wandb is True.
    
    Parameters
    ----------
    y_true : array-like
        True labels (integer or one-hot).
    y_pred : array-like
        Predicted labels (integer).
    groups : array-like, optional
        Group labels. If provided, you may choose to filter the data by a particular group.
    group_to_plot : int, optional
        If provided, the function filters y_true and y_pred to only include entries
        where groups == group_to_plot. If None, no filtering is applied.
    normalize : bool, optional
        If True, normalizes the confusion matrix by row.
        If False, shows raw counts.
        If None, shows raw counts and normalized values.
    labels : list, optional
        A fixed list of label names corresponding to class indices (e.g. index 0 is 'MEL', etc.).
        This list is used as the complete set of classes, so any class not present in the data
        will have its row and column filled with zeros.
    cmap : str, optional
        Colormap for the heatmap.
    title_str : str, optional
        Title for the plot.
    log_to_wandb : bool, optional
        If True, logs the generated plot to wandb.
    """
    # If y_true is one-hot, convert to integer class labels.
    if len(np.shape(y_true)) > 1 and np.shape(y_true)[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Convert tensors to numpy arrays if needed.
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if groups is not None and isinstance(groups, torch.Tensor):
        groups = groups.detach().cpu().numpy()

    # Filter by group if requested.
    if groups is not None and group_to_plot is not None:
        mask = (groups == group_to_plot)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    # If a fixed label mapping is provided, use it.
    if labels is not None:
        # Here we assume that the provided list 'labels' is ordered such that
        # the integer label corresponds to its index. For example, if labels[0]=="MEL",
        # then class 0 will be mapped to "MEL".
        label_map = {i: label for i, label in enumerate(labels)}
        
        def map_label(x):
            if x in label_map:
                return label_map[x]
            else:
                # You can choose to raise an error if an unexpected class is encountered.
                raise ValueError(f"Encountered label {x} not in provided label mapping.")
        
        y_true_mapped = np.array([map_label(x) for x in y_true])
        y_pred_mapped = np.array([map_label(x) for x in y_pred])
        # Use the full provided list as the universe of classes.
        cm_labels = labels
    else:
        # If no custom labels are provided, use the unique labels from the data.
        y_true_mapped = y_true
        y_pred_mapped = y_pred
        cm_labels = np.unique(np.concatenate([y_true, y_pred]))

    # Compute the confusion matrix.
    # By supplying the full cm_labels list, any missing class in the data will have a row/column of zeros.
    if normalize:
        cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=cm_labels, normalize='true')
        fmt = '.2f'  # Use floating-point format for normalized values.
    elif normalize is False:
        cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=cm_labels)
        fmt = 'd'   # Use integer format for raw counts.
    else:
        cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=cm_labels)
        cm_norm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=cm_labels, normalize='true')
        
    if normalize is not None:
        # Create the plot.
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap=cmap, 
            xticklabels=cm_labels, 
            yticklabels=cm_labels,
            ax=ax
        )
    else:
        # Create the plot.
        # Plot raw counts with normalized counts in parentheses.
        fig, ax = plt.subplots(figsize=(7, 6))
        # Create custom annotations: "raw (normalized)"
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.2f})"
        # Use the normalized matrix for colors.
        sns.heatmap(
            cm_norm, 
            annot=annot, 
            fmt="", 
            cmap=cmap, 
            xticklabels=cm_labels, 
            yticklabels=cm_labels,
            ax=ax
        )
        
    

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    full_title = title_str
    if groups is not None and group_to_plot is not None:
        full_title += f"_group{group_to_plot}"
    if normalize:
        full_title += "_normalized"
    elif normalize is None:
        full_title += "_raw_normalized"
    ax.set_title(full_title)
    plt.tight_layout()

    # Save as jpeg, png, pdf
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(f"{path}/{full_title}.png", bbox_inches="tight", dpi=400)
    plt.savefig(f"{path}/{full_title}.jpeg", bbox_inches="tight", dpi=10)
    plt.savefig(f"{path}/{full_title}.pdf", bbox_inches="tight", dpi=400)
    plt.savefig(f"{path}/{full_title}.svg", bbox_inches="tight", dpi=400)

    if log_to_wandb:
        wandb.log({full_title: wandb.Image(f"{path}/{full_title}.jpeg")})
    else:
        plt.show()
    plt.close(fig)



if __name__ == "__main__":
    main()
