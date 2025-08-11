from argparse import ArgumentParser
import torch
import wandb
from datasets import load_dataset
from utils.helper import load_config
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_fn_model_loader
import numpy as np
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
import textwrap
from sklearn.metrics import confusion_matrix as cm
import seaborn as sns

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file',
                        default="config_files/evaluate_metrics_clean/isic_attacked/local/isic_attacked_resnet50d_p_artifact0.1_p_artifact_na0.0_last_conv_baseline.yaml"
                        )
    parser.add_argument('--before_correction', action="store_true")
    parser.add_argument('--visualize', type=bool, default=True)
    parser.add_argument('--print', type=bool, default=False)
    parser.add_argument("--results_dir", default="results/core_spurious_acc", type=str)
    args = parser.parse_args()
    return args

def get_dilated_soft_mask(hard_mask, iterations, k):
    """
    Dilate the hard mask and create a box to create a soft mask.

    Args:
        hard_mask: a tensor of shape (1, H, W) with binary values.
        iterations: number of iterations for dilation.
        k: kernel size for the structuring element.
    """
    soft_mask = hard_mask.clone().cpu().squeeze(0).numpy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    soft_mask = cv2.dilate(soft_mask, kernel, iterations=iterations)

    # Threshold the mask for binary values.
    _, soft_mask = cv2.threshold(soft_mask.astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find indices of the nonzero pixels.
    ys, xs = np.where(soft_mask == 1)

    if ys.size > 0 and xs.size > 0:
        # Get the bounding box coordinates:
        top = np.min(ys)
        bottom = np.max(ys)
        left = np.min(xs)
        right = np.max(xs)

        # Fill every point in the bounding box with 1.
        soft_mask[top:bottom+1, left:right+1] = 1

    return torch.from_numpy(soft_mask).float().unsqueeze(0)

def compute_core_accuracy(model, images, labels, core_masks, sigma=0.25, device='cuda'):
    """
    Returns:
        - class-balanced core accuracy
        - logits (for further analysis)
    """
    if core_masks.ndim == 3:
        core_masks = core_masks.unsqueeze(1)

    noise = torch.randn_like(images).to(device)
    non_core = 1 - core_masks
    images_corrupted = images + sigma * (noise * non_core)

    logits = []
    n_batches = int(np.ceil(images_corrupted.size(0) / 32))
    with torch.no_grad():
        for i in range(n_batches):
            logits.append(model(images_corrupted[i*32:(i+1)*32]))
    logits = torch.cat(logits, dim=0)
    preds = logits.argmax(dim=1)

    # Class-balanced accuracy
    accs = []
    for c in range(logits.size(1)):
        mask = labels == c
        if mask.sum() > 0:
            acc = (preds[mask] == labels[mask]).float().mean().item()
            accs.append(acc)
    class_bal_acc = sum(accs) / len(accs) if accs else 0.0
    return class_bal_acc, logits

def compute_spurious_accuracy(model, images, labels, core_masks, sigma=0.25, device='cuda'):
    """
    Returns:
        - class-balanced spurious accuracy
        - logits
    """
    if core_masks.ndim == 3:
        core_masks = core_masks.unsqueeze(1)

    noise = torch.randn_like(images).to(device)
    images_corrupted = images + sigma * (noise * core_masks)

    logits = []
    n_batches = int(np.ceil(images_corrupted.size(0) / 32))
    with torch.no_grad():
        for i in range(n_batches):
            logits.append(model(images_corrupted[i*32:(i+1)*32]))
    logits = torch.cat(logits, dim=0)
    preds = logits.argmax(dim=1)

    # Class-balanced accuracy
    accs = []
    for c in range(logits.size(1)):
        mask = labels == c
        if mask.sum() > 0:
            acc = (preds[mask] == labels[mask]).float().mean().item()
            accs.append(acc)
    class_bal_acc = sum(accs) / len(accs) if accs else 0.0
    return class_bal_acc, logits

def compute_relative_core_sensitivity(core_acc, spurious_acc):
    """
    Compute the Relative Core Sensitivity (RCS) as:
    
    RCS = (acc(C) - acc(S)) / (2 * min(ā, 1 - ā))

    where ā = (acc(C) + acc(S)) / 2.
    
    Args:
      core_acc: core accuracy (acc(C))
      spurious_acc: spurious accuracy (acc(S))

    Returns:
      rcs: the computed Relative Core Sensitivity.
    """
    avg_acc = (core_acc + spurious_acc) / 2
    possible_gap = 2 * min(avg_acc, 1 - avg_acc)
    # Avoid division by zero.
    rcs = (core_acc - spurious_acc) / possible_gap if possible_gap > 0 else 0
    return rcs

def main():
    args = get_args()
    config = load_config(args.config_file)
    
    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'], project=config['wandb_project_name'], resume=True)

    compute_core_spurious_accuracy(config, args.before_correction, args.visualize, args.results_dir, args.print)

def compute_core_spurious_accuracy(config, before_correction, visualize, path, print_res):
    """
    Compute Core and Spurious Accuracy and Relative Core Sensitivity.
    Uses hard masks and soft masks (boxes) for the core regions.

    Args:
      config (dict): experiment config.
      before_correction (bool): whether to use the model before correction.
    """
    model_name = config['model_name']
    config_name = config['config_name']
    dataset_name = config['dataset_name']
    base_path = f"{path}/{dataset_name}/{config_name}"

    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = config.get('device', default_device)

    # Load dataset initially to get classes and prepare for evaluation
    dataset = load_dataset(config, normalize_data=True, hm=True)
    classes = dataset.classes

    ckpt_path = config['ckpt_path'] if before_correction else f"{config['checkpoint_dir_corrected']}/{config_name}/last.ckpt"
    model = get_fn_model_loader(model_name=model_name)(n_class=dataset.classes.__len__(), ckpt_path=ckpt_path, device=device)
    model = prepare_model_for_evaluation(model, dataset, device, config)

    artifact_labels = list(dataset.sample_ids_by_artifact.keys())
    artifact_sample_ids = list(dataset.sample_ids_by_artifact.values())

    if dataset_name == 'isic_attacked' or dataset_name == 'imagenet_attacked':
        artifact_labels += ['full', 'clean']
        # Concatenate numpy arrays for artifact_sample_ids if they exist
        full_ids = np.arange(len(dataset))
        clean_ids = np.array(dataset.clean_sample_ids)

        # Ensure artifact_sample_ids is a list of numpy arrays for consistent appending
        artifact_sample_ids = [np.array(ids) for ids in artifact_sample_ids]
        artifact_sample_ids += [full_ids, clean_ids]

    train_set = dataset.idxs_train
    test_set = dataset.idxs_test
    val_set = dataset.idxs_val

    splits = {
        "val": val_set,
        "test": test_set,
        #"train": train_set # Commented out as in original
    }

    results = {}

    for split, sample_ids_for_split in splits.items(): # Renamed sample_ids to sample_ids_for_split to avoid confusion
      # Filter `artifact_sample_ids` to include only those within the current split
      sample_sets_split_raw = [[y for y in x if y in sample_ids_for_split] for x in artifact_sample_ids]

      # This is the crucial part for filtering samples without masks
      sample_sets_split_filtered = []
      for artifact_type_samples in sample_sets_split_raw:
          temp_filtered_samples = []
          for sample_idx in tqdm(artifact_type_samples, desc=f"Filtering samples for {split} with {artifact_labels[sample_sets_split_raw.index(artifact_type_samples)]}"):
              # Reload dataset for each get_segmentation_mask call if it's too heavy to keep in memory,
              # or if the dataset is reset during each loop. Assuming it's the same dataset instance.
              mask = dataset.get_segmentation_mask(sample_idx)
              # Check if the mask is valid (not np.nan)
              if not isinstance(mask, float) or not np.isnan(mask):
                  temp_filtered_samples.append(sample_idx)
              # else:
                  # print(f"Skipping sample {sample_idx} in {split} due to missing bounding box annotation.")
          sample_sets_split_filtered.append(temp_filtered_samples)
      
      # Use the filtered samples for the rest of the computation
      sample_sets_split = sample_sets_split_filtered

      for k, samples in enumerate(sample_sets_split):
        # Re-initialize dataset if necessary, though it seems `dataset` is already initialized above.
        # If dataset state changes (e.g., transforms, random seeds), re-initializing here might be needed.
        # Otherwise, the `dataset` object from outside the loop should be fine.
        # For simplicity, assuming the `dataset` object remains valid.
        # dataset = load_dataset(config, normalize_data=True, hm=True) # Uncomment if dataset re-initialization is truly needed per artifact type

        n_samples = len(samples)
        if n_samples == 0:
            print(f"No samples with GT mask found for split {split} and artifact {artifact_labels[k]}. Skipping evaluation.")
            continue

        n_batches = int(np.ceil(n_samples / config['batch_size']))

        print(f"Computing core and spurious accuracy for split {split} and artifact {artifact_labels[k]} ({n_samples} samples)")

        image_list = []
        target_list = []
        soft_mask_list = []
        hard_mask_list = []
        sample_batch_list = []

        for i in tqdm(range(n_batches)):
            samples_batch = samples[i * config['batch_size']:(i + 1) * config['batch_size']]
            
            # Since we've already filtered out samples without masks, we can directly stack here.
            images = torch.stack([dataset[j][0] for j in samples_batch], dim=0).to(device)
            targets = torch.stack([dataset[j][1] for j in samples_batch], dim=0).to(device)

            hard_masks = torch.stack([dataset.get_segmentation_mask(j) for j in samples_batch], dim=0).to(device)

            # Compute soft core mask (box) from hard core mask.
            soft_masks = torch.stack([get_dilated_soft_mask(hard_mask, iterations=3, k=2) for hard_mask in hard_masks]).to(device)

            if soft_masks.ndim == 3:
                soft_masks = soft_masks.unsqueeze(1)
            if hard_masks.ndim == 3:
                hard_masks = hard_masks.unsqueeze(1)

            image_list.append(images)
            target_list.append(targets)
            soft_mask_list.append(soft_masks)
            hard_mask_list.append(hard_masks)
            sample_batch_list.append(samples_batch)

        images = torch.cat(image_list, dim=0)
        targets = torch.cat(target_list, dim=0)
        soft_masks = torch.cat(soft_mask_list, dim=0)
        hard_masks = torch.cat(hard_mask_list, dim=0)
        sample_idxs = [item for sublist in sample_batch_list for item in sublist]

        # Overall Accuracy
        logits = []
        n_batches_eval = int(np.ceil(images.size(0) / 32)) # Use a separate variable for batch size for evaluation if different
        with torch.no_grad():
            for i in range(n_batches_eval):
                logits.append(model(images[i*32:(i+1)*32]))
        logits = torch.cat(logits, dim=0)
        preds = logits.argmax(dim=1)
        ov_acc = (preds == targets).float().mean().item()

        for sigma in [0.25, 0.5, 0.75]:
            prefix = f"{split}"
            suffix = f"sigma_{sigma}_artifact_{artifact_labels[k]}"

            # Compute Core Accuracy
            core_acc_soft, core_logits_soft = compute_core_accuracy(model, images, targets, soft_masks, device=device, sigma=sigma)
            core_acc_hard, core_logits_hard = compute_core_accuracy(model, images, targets, hard_masks, device=device, sigma=sigma)

            # Compute Spurious Accuracy
            spurious_acc_soft, spurrious_logits_soft = compute_spurious_accuracy(model, images, targets, soft_masks, device=device, sigma=sigma)
            spurious_acc_hard, spurrious_logits_hard = compute_spurious_accuracy(model, images, targets, hard_masks, device=device, sigma=sigma)

            # Compute Relative Core Sensitivity
            rcs_soft = compute_relative_core_sensitivity(core_acc_soft, spurious_acc_soft)
            rcs_hard = compute_relative_core_sensitivity(core_acc_hard, spurious_acc_hard)

            # Results
            results[f"{prefix}_overall_accuracy_{suffix}"] = ov_acc
            results[f"{prefix}_core_accuracy_soft_mask_{suffix}"] = core_acc_soft
            results[f"{prefix}_spurious_accuracy_soft_mask_{suffix}"] = spurious_acc_soft
            results[f"{prefix}_rcs_soft_mask_{suffix}"] = rcs_soft
            results[f"{prefix}_core_accuracy_hard_mask_{suffix}"] = core_acc_hard
            results[f"{prefix}_spurious_accuracy_hard_mask_{suffix}"] = spurious_acc_hard
            results[f"{prefix}_rcs_hard_mask_{suffix}"] = rcs_hard

            # Logits
            core_logits_soft_avg = torch.mean(core_logits_soft, dim=0)
            spurrious_logits_soft_avg = torch.mean(spurrious_logits_soft, dim=0)
            core_logits_hard_avg = torch.mean(core_logits_hard, dim=0)
            spurrious_logits_hard_avg = torch.mean(spurrious_logits_hard, dim=0)
            overall_logits_avg = torch.mean(logits, dim=0)

            # Logits to dict, class: avg logit
            core_logits_soft_avg = {classes[i]: core_logits_soft_avg[i].item() for i in range(core_logits_soft_avg.size(0))}
            spurrious_logits_soft_avg = {classes[i]: spurrious_logits_soft_avg[i].item() for i in range(spurrious_logits_soft_avg.size(0))}
            core_logits_hard_avg = {classes[i]: core_logits_hard_avg[i].item() for i in range(core_logits_hard_avg.size(0))}
            spurrious_logits_hard_avg = {classes[i]: spurrious_logits_hard_avg[i].item() for i in range(spurrious_logits_hard_avg.size(0))}
            overall_logits_avg = {classes[i]: overall_logits_avg[i].item() for i in range(overall_logits_avg.size(0))}

            results[f"{prefix}_core_logits_soft_mask_{suffix}"] = core_logits_soft_avg
            results[f"{prefix}_spurious_logits_soft_mask_{suffix}"] = spurrious_logits_soft_avg
            results[f"{prefix}_core_logits_hard_mask_{suffix}"] = core_logits_hard_avg
            results[f"{prefix}_spurious_logits_hard_mask_{suffix}"] = spurrious_logits_hard_avg
            results[f"{prefix}_overall_logits_{suffix}"] = overall_logits_avg

            core_acc_drop_soft = ov_acc - core_acc_soft
            spurious_acc_drop_soft = ov_acc - spurious_acc_soft
            core_acc_drop_hard = ov_acc - core_acc_hard
            spurious_acc_drop_hard = ov_acc - spurious_acc_hard
            spurious_core_acc_drop_soft = spurious_acc_soft - core_acc_soft
            spurious_core_acc_drop_hard = spurious_acc_hard - core_acc_hard

            results[f"{prefix}_core_acc_drop_soft_{suffix}"] = core_acc_drop_soft
            results[f"{prefix}_spurious_acc_drop_soft_{suffix}"] = spurious_acc_drop_soft
            results[f"{prefix}_core_acc_drop_hard_{suffix}"] = core_acc_drop_hard
            results[f"{prefix}_spurious_acc_drop_hard_{suffix}"] = spurious_acc_drop_hard
            results[f"{prefix}_spurious_core_acc_drop_soft_{suffix}"] = spurious_core_acc_drop_soft
            results[f"{prefix}_spurious_core_acc_drop_hard_{suffix}"] = spurious_core_acc_drop_hard

            # Logits
            spurious_core_logits_drop_soft = {classes[i]: spurrious_logits_soft_avg[classes[i]] - core_logits_soft_avg[classes[i]] for i in range(len(spurrious_logits_soft_avg))}
            spurious_core_logits_drop_hard = {classes[i]: spurrious_logits_hard_avg[classes[i]] - core_logits_hard_avg[classes[i]] for i in range(len(spurrious_logits_soft_avg))}
            sc_average_drop_soft = np.mean(list(spurious_core_logits_drop_soft.values()))
            sc_average_drop_hard = np.mean(list(spurious_core_logits_drop_hard.values()))
            # Core vs Overall
            core_logits_drop_soft = {classes[i]: core_logits_soft_avg[classes[i]] - overall_logits_avg[classes[i]] for i in range(len(spurrious_logits_soft_avg))}
            core_logits_drop_hard = {classes[i]: core_logits_hard_avg[classes[i]] - overall_logits_avg[classes[i]] for i in range(len(spurrious_logits_soft_avg))}

            co_average_drop_soft = np.mean(list(core_logits_drop_soft.values()))
            co_average_drop_hard = np.mean(list(core_logits_drop_hard.values()))

            # Spurious vs Overall
            spurrious_logits_drop_soft = {classes[i]: spurrious_logits_soft_avg[classes[i]] - overall_logits_avg[classes[i]] for i in range(len(spurrious_logits_soft_avg))}
            spurrious_logits_drop_hard = {classes[i]: spurrious_logits_hard_avg[classes[i]] - overall_logits_avg[classes[i]] for i in range(len(spurrious_logits_soft_avg))}

            so_average_drop_soft = np.mean(list(spurrious_logits_drop_soft.values()))
            so_average_drop_hard = np.mean(list(spurrious_logits_drop_hard.values()))

            results[f"{prefix}_spurious_core_logits_drop_soft_{suffix}"] = spurious_core_logits_drop_soft
            results[f"{prefix}_spurious_core_logits_drop_hard_{suffix}"] = spurious_core_logits_drop_hard
            results[f"{prefix}_sc_average_drop_soft_{suffix}"] = sc_average_drop_soft
            results[f"{prefix}_sc_average_drop_hard_{suffix}"] = sc_average_drop_hard
            results[f"{prefix}_core_logits_drop_soft_{suffix}"] = core_logits_drop_soft
            results[f"{prefix}_core_logits_drop_hard_{suffix}"] = core_logits_drop_hard
            results[f"{prefix}_co_average_drop_soft_{suffix}"] = co_average_drop_soft
            results[f"{prefix}_co_average_drop_hard_{suffix}"] = co_average_drop_hard
            results[f"{prefix}_spurrious_logits_drop_soft_{suffix}"] = spurrious_logits_drop_soft
            results[f"{prefix}_spurrious_logits_drop_hard_{suffix}"] = spurrious_logits_drop_hard
            results[f"{prefix}_so_average_drop_soft_{suffix}"] = so_average_drop_soft
            results[f"{prefix}_so_average_drop_hard_{suffix}"] = so_average_drop_hard


            # Confusion Matrix
            # Overall
            cm_overall = cm(targets.cpu().numpy(), preds.cpu().numpy(), labels=range(len(classes)))
            cm_overall_norm = cm(targets.cpu().numpy(), preds.cpu().numpy(), normalize='true', labels=range(len(classes)))
            cm_overall_norm = np.round(cm_overall_norm, 2) # Round to 2 decimals
            cm_overall_norm = cm_overall_norm.tolist()
            cm_overall = cm_overall.tolist()

        
            results[f"{prefix}_cm_overall_{suffix}"] = cm_overall
            results[f"{prefix}_cm_overall_norm_{suffix}"] = cm_overall_norm

            # Core Soft
            preds_core_soft = core_logits_soft.argmax(dim=1).cpu().numpy()
            cm_core_soft = cm(targets.cpu().numpy(), preds_core_soft, labels=range(len(classes)))
            cm_core_soft_norm = cm(targets.cpu().numpy(), preds_core_soft, normalize='true', labels=range(len(classes)))
            cm_core_soft_norm = np.round(cm_core_soft_norm, 2) # Round to 2 decimals
            cm_core_soft_norm = cm_core_soft_norm.tolist()
            cm_core_soft = cm_core_soft.tolist()

            results[f"{prefix}_cm_core_soft_{suffix}"] = cm_core_soft
            results[f"{prefix}_cm_core_soft_norm_{suffix}"] = cm_core_soft_norm

            # Spurious Soft
            preds_spur_soft = spurrious_logits_soft.argmax(dim=1).cpu().numpy()
            cm_spur_soft = cm(targets.cpu().numpy(), preds_spur_soft, labels=range(len(classes)))
            cm_spur_soft_norm = cm(targets.cpu().numpy(), preds_spur_soft, normalize='true', labels=range(len(classes)))
            cm_spur_soft_norm = np.round(cm_spur_soft_norm, 2) # Round to 2 decimals
            cm_spur_soft_norm = cm_spur_soft_norm.tolist()
            cm_spur_soft = cm_spur_soft.tolist()

            results[f"{prefix}_cm_spur_soft_{suffix}"] = cm_spur_soft
            results[f"{prefix}_cm_spur_soft_norm_{suffix}"] = cm_spur_soft_norm

            # Core Hard
            preds_core_hard = core_logits_hard.argmax(dim=1).cpu().numpy()
            cm_core_hard = cm(targets.cpu().numpy(), preds_core_hard, labels=range(len(classes)))
            cm_core_hard_norm = cm(targets.cpu().numpy(), preds_core_hard, normalize='true', labels=range(len(classes)))
            cm_core_hard_norm = np.round(cm_core_hard_norm, 2) # Round to 2 decimals
            cm_core_hard_norm = cm_core_hard_norm.tolist()
            cm_core_hard = cm_core_hard.tolist()

            results[f"{prefix}_cm_core_hard_{suffix}"] = cm_core_hard
            results[f"{prefix}_cm_core_hard_norm_{suffix}"] = cm_core_hard_norm

            # Spurious Hard
            preds_spur_hard = spurrious_logits_hard.argmax(dim=1).cpu().numpy()
            cm_spur_hard = cm(targets.cpu().numpy(), preds_spur_hard, labels=range(len(classes)))
            cm_spur_hard_norm = cm(targets.cpu().numpy(), preds_spur_hard, normalize='true', labels=range(len(classes)))
            cm_spur_hard_norm = np.round(cm_spur_hard_norm, 2) # Round to 2 decimals
            cm_spur_hard_norm = cm_spur_hard_norm.tolist()
            cm_spur_hard = cm_spur_hard.tolist()
            
            results[f"{prefix}_cm_spur_hard_{suffix}"] = cm_spur_hard
            results[f"{prefix}_cm_spur_hard_norm_{suffix}"] = cm_spur_hard_norm

            # Print some results
            if print_res:
                print(f"Split: {split}, Artifact: {artifact_labels[k]}, Sigma: {sigma}")
                print()
                # Accuracy
                print(f"Overall Accuracy: {ov_acc}")
                print(f"Core Accuracy Soft Mask: {core_acc_soft}")
                print(f"Spurious Accuracy Soft Mask: {spurious_acc_soft}")
                print(f"Relative Core Sensitivity Soft Mask: {rcs_soft}")
                print(f"Core Accuracy Hard Mask: {core_acc_hard}")
                print(f"Spurious Accuracy Hard Mask: {spurious_acc_hard}")
                print(f"Relative Core Sensitivity Hard Mask: {rcs_hard}")
                print()
                # Drop in accuracy
                print(f"Core Accuracy Drop Soft Mask: {core_acc_drop_soft}")
                print(f"Spurious Accuracy Drop Soft Mask: {spurious_acc_drop_soft}")
                print(f"Core Accuracy Drop Hard Mask: {core_acc_drop_hard}")
                print(f"Spurious Accuracy Drop Hard Mask: {spurious_acc_drop_hard}")
                print(f"Spurious Core Accuracy Drop Soft Mask: {spurious_core_acc_drop_soft}")
                print(f"Spurious Core Accuracy Drop Hard Mask: {spurious_core_acc_drop_hard}")
                print()
                # Logits & Logits Drop
                print(f"Overall Logits: {overall_logits_avg}")  
                print(f"Core Logits Soft Mask: {core_logits_soft_avg}")
                print(f"Spurious Logits Soft Mask: {spurrious_logits_soft_avg}")
                print(f"Core Logits Hard Mask: {core_logits_hard_avg}")
                print(f"Spurious Logits Hard Mask: {spurrious_logits_hard_avg}")
                print()
                print(f"Spurious Core Logits Drop Soft Mask: {spurious_core_logits_drop_soft}")
                print(f"Spurious Core Logits Drop Hard Mask: {spurious_core_logits_drop_hard}")
                print(f"Average Spurious Core Logits Drop Soft Mask: {sc_average_drop_soft}")
                print(f"Average Spurious Core Logits Drop Hard Mask: {sc_average_drop_hard}")
                print(f"Core Logits Drop Soft Mask: {core_logits_drop_soft}")
                print(f"Core Logits Drop Hard Mask: {core_logits_drop_hard}")
                print(f"Average Core Logits Drop Soft Mask: {co_average_drop_soft}")
                print(f"Average Core Logits Drop Hard Mask: {co_average_drop_hard}")
                print(f"Spurious Logits Drop Soft Mask: {spurrious_logits_drop_soft}")
                print(f"Spurious Logits Drop Hard Mask: {spurrious_logits_drop_hard}")
                print(f"Average Spurious Logits Drop Soft Mask: {so_average_drop_soft}")
                print(f"Average Spurious Logits Drop Hard Mask: {so_average_drop_hard}")
                print()
                # Confusion Matrix
                print(f"Confusion Matrix Overall: {cm_overall}")
                print(f"Confusion Matrix Normalized Overall: {cm_overall_norm}")
                print()
                print(f"Confusion Matrix Core Soft: {cm_core_soft}")
                print(f"Confusion Matrix Normalized Core Soft: {cm_core_soft_norm}")
                print()
                print(f"Confusion Matrix Spurious Soft: {cm_spur_soft}")
                print(f"Confusion Matrix Normalized Spurious Soft: {cm_spur_soft_norm}")
                print()
                print(f"Confusion Matrix Core Hard: {cm_core_hard}")
                print(f"Confusion Matrix Normalized Core Hard: {cm_core_hard_norm}")
                print()
                print(f"Confusion Matrix Spurious Hard: {cm_spur_hard}")
                print(f"Confusion Matrix Normalized Spurious Hard: {cm_spur_hard_norm}")
                print()

            if visualize:
                # Plot Confusion Matrix for Soft Mask and Hard Mask
                path = f"{base_path}/{artifact_labels[k]}/{split}"
                if not os.path.exists(path):
                    os.makedirs(path)
                if not os.path.exists(f"{path}/cm_soft"):
                    os.makedirs(f"{path}/cm_soft")
                if not os.path.exists(f"{path}/cm_hard"):
                    os.makedirs(f"{path}/cm_hard")
                
                # Overall vs Soft Mask (Raw and Normalized) (Core and Spurious)
                fig, axes = plt.subplots(3, 2, figsize=(10, 10))
                plt.suptitle(f"Split: {split}, Artifact: {artifact_labels[k]}, Sigma: {sigma}")
                sns.heatmap(cm_overall, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues', xticklabels=classes, yticklabels=classes)
                axes[0, 0].set_title(f"Overall Confusion Matrix, Accuracy: {np.round(ov_acc, 5)}")
                sns.heatmap(cm_overall_norm, annot=True, fmt='g', ax=axes[0, 1], cmap='Blues', xticklabels=classes, yticklabels=classes)
                axes[0, 1].set_title("Overall Normalized Confusion Matrix")
                sns.heatmap(cm_core_soft, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues', xticklabels=classes, yticklabels=classes)
                axes[1, 0].set_title(f"Core Soft Confusion Matrix, Accuracy: {np.round(core_acc_soft, 5)}")
                sns.heatmap(cm_core_soft_norm, annot=True, fmt='g', ax=axes[1, 1], cmap='Blues', xticklabels=classes, yticklabels=classes)
                axes[1, 1].set_title("Core Soft Normalized Confusion Matrix")
                sns.heatmap(cm_spur_soft, annot=True, fmt='d', ax=axes[2, 0], cmap='Blues', xticklabels=classes, yticklabels=classes)
                axes[2, 0].set_title(f"Spurious Soft Confusion Matrix, Accuracy: {np.round(spurious_acc_soft, 5)}")
                sns.heatmap(cm_spur_soft_norm, annot=True, fmt='g', ax=axes[2, 1], cmap='Blues', xticklabels=classes, yticklabels=classes)
                axes[2, 1].set_title("Spurious Soft Normalized Confusion Matrix")

                # Set X and Y labels
                for i in range(3):
                    for j in range(2):
                        axes[i, j].set_xlabel('Predicted')
                        axes[i, j].set_ylabel('True')
                
                plt.tight_layout()
                plt.savefig(f"{path}/cm_soft/{split}_confusion_matrix_soft_mask_{suffix}.png", dpi=400)
                plt.savefig(f"{path}/cm_soft/{split}_confusion_matrix_soft_mask_{suffix}.svg", dpi=400)
                plt.close()

                # Overall vs Hard Mask (Raw and Normalized) (Core and Spurious)
                fig, axes = plt.subplots(3, 2, figsize=(10, 10))
                plt.suptitle(f"Split: {split}, Artifact: {artifact_labels[k]}, Sigma: {sigma}")
                sns.heatmap(cm_overall, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues', xticklabels=classes, yticklabels=classes)
                axes[0, 0].set_title(f"Overall Confusion Matrix, Accuracy: {np.round(ov_acc, 5)}")
                sns.heatmap(cm_overall_norm, annot=True, fmt='g', ax=axes[0, 1], cmap='Blues', xticklabels=classes, yticklabels=classes)
                axes[0, 1].set_title("Overall Normalized Confusion Matrix")
                sns.heatmap(cm_core_hard, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues', xticklabels=classes, yticklabels=classes)
                axes[1, 0].set_title(f"Core Hard Confusion Matrix, Accuracy: {np.round(core_acc_hard, 5)}")
                sns.heatmap(cm_core_hard_norm, annot=True, fmt='g', ax=axes[1, 1], cmap='Blues', xticklabels=classes, yticklabels=classes)
                axes[1, 1].set_title("Core Hard Normalized Confusion Matrix")
                sns.heatmap(cm_spur_hard, annot=True, fmt='d', ax=axes[2, 0], cmap='Blues', xticklabels=classes, yticklabels=classes)
                axes[2, 0].set_title(f"Spurious Hard Confusion Matrix, Accuracy: {np.round(spurious_acc_hard, 5)}")
                sns.heatmap(cm_spur_hard_norm, annot=True, fmt='g', ax=axes[2, 1], cmap='Blues', xticklabels=classes, yticklabels=classes)
                axes[2, 1].set_title("Spurious Hard Normalized Confusion Matrix")
                
                # Set X and Y labels
                for i in range(3):
                    for j in range(2):
                        axes[i, j].set_xlabel('Predicted')
                        axes[i, j].set_ylabel('True')
                
                plt.tight_layout()
                plt.savefig(f"{path}/cm_hard/{split}_confusion_matrix_hard_mask_{suffix}.png", dpi=400)
                plt.savefig(f"{path}/cm_hard/{split}_confusion_matrix_hard_mask_{suffix}.svg", dpi=400)
                

                # Save to wandb
                if config.get('wandb_api_key', None):
                    plt.savefig(f"{path}/cm_soft/{split}_confusion_matrix_soft_mask_{suffix}.jpeg", dpi=100)    
                    wandb.log({f"{split}_confusion_matrix_soft_mask_{suffix}": wandb.Image(f"{path}/cm_soft/{split}_confusion_matrix_soft_mask_{suffix}.jpeg")})
                    plt.savefig(f"{path}/cm_hard/{split}_confusion_matrix_hard_mask_{suffix}.jpeg", dpi=100)
                    wandb.log({f"{split}_confusion_matrix_hard_mask_{suffix}": wandb.Image(f"{path}/cm_hard/{split}_confusion_matrix_hard_mask_{suffix}.jpeg")})

                plt.close()

            if visualize:
                # Visualize the results.
                n_samples_to_plot = 5
                target_to_plot = targets[:n_samples_to_plot].cpu().numpy()
                images_to_plot = images[:n_samples_to_plot].cpu().numpy()
                soft_masks_to_plot = soft_masks[:n_samples_to_plot].cpu().numpy()
                hard_masks_to_plot = hard_masks[:n_samples_to_plot].cpu().numpy()
                idxs = sample_idxs[:n_samples_to_plot]

                # Get Predictions
                model.eval()
                with torch.no_grad():
                    logits = model(torch.Tensor(images_to_plot).to(device))
                    preds_to_plot = logits.argmax(dim=1).cpu().numpy()

                # Get Predictions for corrupted images.
                # We do it here so we dont have a distribution shift because of a missing normalization and cliped images
                with torch.no_grad():
                    images_to_plot = torch.Tensor(images_to_plot).cpu()
                    images_corrupted_core_soft = images_to_plot + sigma * (torch.randn_like(images_to_plot) * (1 - soft_masks_to_plot))
                    images_corrupted_core_hard = images_to_plot + sigma * (torch.randn_like(images_to_plot) * (1 - hard_masks_to_plot))
                    images_corrupted_spurious_soft = images_to_plot + sigma * (torch.randn_like(images_to_plot) * soft_masks_to_plot)
                    images_corrupted_spurious_hard = images_to_plot + sigma * (torch.randn_like(images_to_plot) * hard_masks_to_plot)

                    logits_image = model(images_to_plot[:n_samples_to_plot].to(device))
                    logits_corrupted_core_soft = model(images_corrupted_core_soft[:n_samples_to_plot].to(device))
                    logits_corrupted_core_hard = model(images_corrupted_core_hard[:n_samples_to_plot].to(device))
                    logits_corrupted_spurious_soft = model(images_corrupted_spurious_soft[:n_samples_to_plot].to(device))
                    logits_corrupted_spurious_hard = model(images_corrupted_spurious_hard[:n_samples_to_plot].to(device))

                    pred_core_soft = logits_corrupted_core_soft.argmax(dim=1).cpu().numpy()
                    pred_spur_soft = logits_corrupted_spurious_soft.argmax(dim=1).cpu().numpy()
                    pred_core_hard = logits_corrupted_core_hard.argmax(dim=1).cpu().numpy()
                    pred_spur_hard = logits_corrupted_spurious_hard.argmax(dim=1).cpu().numpy()

                # Make images unnormlized and clipped to [0, 1] for plotting.
                images_to_plot = []
                for idx in idxs:
                    image = dataset.reverse_normalization(dataset[idx][0]) / 255
                    images_to_plot.append(image)

                images_to_plot = torch.stack(images_to_plot, dim=0).float().cpu()

                images_corrupted_core_soft = images_to_plot + sigma * (torch.randn_like(images_to_plot) * (1 - soft_masks_to_plot))
                images_corrupted_core_hard = images_to_plot + sigma * (torch.randn_like(images_to_plot) * (1 - hard_masks_to_plot))
                images_corrupted_spurious_soft = images_to_plot + sigma * (torch.randn_like(images_to_plot) * soft_masks_to_plot)
                images_corrupted_spurious_hard = images_to_plot + sigma * (torch.randn_like(images_to_plot) * hard_masks_to_plot)

                # Clip images to [0, 1]
                images_corrupted_core_soft = np.clip(images_corrupted_core_soft, 0, 1)
                images_corrupted_core_hard = np.clip(images_corrupted_core_hard, 0, 1)
                images_corrupted_spurious_soft = np.clip(images_corrupted_spurious_soft, 0, 1)
                images_corrupted_spurious_hard = np.clip(images_corrupted_spurious_hard, 0, 1)
                soft_masks_to_plot = np.clip(soft_masks_to_plot, 0, 1)
                hard_masks_to_plot = np.clip(hard_masks_to_plot, 0, 1)
                images_to_plot = np.clip(images_to_plot, 0, 1)


                # Plot image, soft_mask, hard_mask, images_corrupted_core_soft, images_corrupted_spurious_soft, images_corrupted_core_hard, images_corrupted_spurious_hard
                fig, axes = plt.subplots(n_samples_to_plot, 7, figsize=(20, 4 * n_samples_to_plot))
                plt.axis('off')
                plt.suptitle(f"Split: {split}, Artifact: {artifact_labels[k]}, Sigma: {sigma}")
                for i in range(n_samples_to_plot):
                    axes[i, 0].imshow(images_to_plot[i].permute(1, 2, 0))
                    axes[i, 0].set_title(f"Image {idxs[i]}, Target: {classes[target_to_plot[i]]}, Pred: {classes[preds_to_plot[i]]}", fontsize=10)
                    axes[i, 0].axis('off')

                    axes[i, 1].imshow(soft_masks_to_plot[i].squeeze(0), cmap='gray')
                    axes[i, 1].set_title(f"Soft Mask {i}", fontsize=10)
                    axes[i, 1].axis('off')

                    axes[i, 2].imshow(hard_masks_to_plot[i].squeeze(0), cmap='gray')
                    axes[i, 2].set_title(f"Hard Mask {i}", fontsize=10)
                    axes[i, 2].axis('off')

                    axes[i, 3].imshow(images_corrupted_core_soft[i].permute(1, 2, 0))
                    axes[i, 3].set_title(f"Core Soft {i}, Pred: {classes[pred_core_soft[i]]}", fontsize=10)
                    axes[i, 3].axis('off')

                    axes[i, 4].imshow(images_corrupted_spurious_soft[i].permute(1, 2, 0))
                    axes[i, 4].set_title(f"Spurious Soft {i}, Pred: {classes[pred_spur_soft[i]]}", fontsize=10)
                    axes[i, 4].axis('off')

                    axes[i, 5].imshow(images_corrupted_core_hard[i].permute(1, 2, 0))
                    axes[i, 5].set_title(f"Core Hard {i}, Pred: {classes[pred_core_hard[i]]}", fontsize=10)
                    axes[i, 5].axis('off')

                    axes[i, 6].imshow(images_corrupted_spurious_hard[i].permute(1, 2, 0))
                    axes[i, 6].set_title(f"Spurious Hard {i}, Pred: {classes[pred_spur_hard[i]]}", fontsize=10)
                    axes[i, 6].axis('off')
                plt.tight_layout()

                if not os.path.exists(path):
                    os.makedirs(path)
                if not os.path.exists(f"{path}/plot"):
                    os.makedirs(f"{path}/plot")

                plt.savefig(f"{path}/plot/{split}_plot_core_spurious_accuracy_sigma_{sigma}_artifact_{artifact_labels[k]}.png", dpi=400)
                plt.savefig(f"{path}/plot/{split}_plot_core_spurious_accuracy_sigma_{sigma}_artifact_{artifact_labels[k]}.jpeg", dpi=10)
                plt.savefig(f"{path}/plot/{split}_plot_core_spurious_accuracy_sigma_{sigma}_artifact_{artifact_labels[k]}.pdf" , dpi=400)
                plt.savefig(f"{path}/plot/{split}_plot_core_spurious_accuracy_sigma_{sigma}_artifact_{artifact_labels[k]}.svg" , dpi=400)

                if config.get('wandb_api_key', None):
                    path_to_save = f"{path}/plot/{split}_plot_core_spurious_accuracy_sigma_{sigma}_artifact_{artifact_labels[k]}.jpeg"
                    wandb.log({f"{split}_plot_core_spurious_accuracy_sigma_{sigma}_artifact_{artifact_labels[k]}": wandb.Image(path_to_save)})
                plt.close()

                # Plot only hard mask 
                fig, axes = plt.subplots(n_samples_to_plot, 4, figsize=(20, 4 * n_samples_to_plot))
                plt.axis('off')
                plt.suptitle(f"Split: {split}, Artifact: {artifact_labels[k]}, Sigma: {sigma}")
                for i in range(n_samples_to_plot):
                    axes[i, 0].imshow(images_to_plot[i].permute(1, 2, 0))
                    axes[i, 0].set_title(f"Image {idxs[i]}, Target: {classes[target_to_plot[i]]}, Pred: {classes[preds_to_plot[i]]}", fontsize=10)
                    axes[i, 0].axis('off')

                    axes[i, 1].imshow(hard_masks_to_plot[i].squeeze(0), cmap='gray')
                    axes[i, 1].set_title(f"Hard Mask {i}", fontsize=10)
                    axes[i, 1].axis('off')

                    axes[i, 2].imshow(images_corrupted_core_hard[i].permute(1, 2, 0))
                    axes[i, 2].set_title(f"Core Hard {i}, Pred: {classes[pred_core_hard[i]]}", fontsize=10)
                    axes[i, 2].axis('off')

                    axes[i, 3].imshow(images_corrupted_spurious_hard[i].permute(1, 2, 0))
                    axes[i, 3].set_title(f"Spurious Hard {i}, Pred: {classes[pred_spur_hard[i]]}", fontsize=10)
                    axes[i, 3].axis('off')
                plt.tight_layout()

                if not os.path.exists(path):
                    os.makedirs(path)
                if not os.path.exists(f"{path}/plot/hard_mask/"):
                    os.makedirs(f"{path}/plot/hard_mask")

                plt.savefig(f"{path}/plot/hard_mask/{split}_plot_hard_mask_core_spurious_accuracy_sigma_{sigma}_artifact_{artifact_labels[k]}.png", dpi=400)
                plt.savefig(f"{path}/plot/hard_mask/{split}_plot_hard_mask_core_spurious_accuracy_sigma_{sigma}_artifact_{artifact_labels[k]}.jpeg", dpi=10)
                plt.savefig(f"{path}/plot/hard_mask/{split}_plot_hard_mask_core_spurious_accuracy_sigma_{sigma}_artifact_{artifact_labels[k]}.pdf" , dpi=400)
                plt.savefig(f"{path}/plot/hard_mask/{split}_plot_hard_mask_core_spurious_accuracy_sigma_{sigma}_artifact_{artifact_labels[k]}.svg" , dpi=400)

                if config.get('wandb_api_key', None):
                    path_to_save = f"{path}/plot/hard_mask/{split}_plot_hard_mask_core_spurious_accuracy_sigma_{sigma}_artifact_{artifact_labels[k]}.jpeg"
                    wandb.log({f"{split}_plot_core_spurious_accuracy_sigma_{sigma}_artifact_{artifact_labels[k]}": wandb.Image(path_to_save)})
                plt.close()

                # Plot only soft mask 
                fig, axes = plt.subplots(n_samples_to_plot, 4, figsize=(20, 4 * n_samples_to_plot))
                plt.axis('off')
                plt.suptitle(f"Split: {split}, Artifact: {artifact_labels[k]}, Sigma: {sigma}")
                for i in range(n_samples_to_plot):
                    axes[i, 0].imshow(images_to_plot[i].permute(1, 2, 0))
                    axes[i, 0].set_title(f"Image {idxs[i]}, Target: {classes[target_to_plot[i]]}, Pred: {classes[preds_to_plot[i]]}", fontsize=10)
                    axes[i, 0].axis('off')

                    axes[i, 1].imshow(soft_masks_to_plot[i].squeeze(0), cmap='gray')
                    axes[i, 1].set_title(f"Soft Mask {i}", fontsize=10)
                    axes[i, 1].axis('off')

                    axes[i, 2].imshow(images_corrupted_core_soft[i].permute(1, 2, 0))
                    axes[i, 2].set_title(f"Core Soft {i}, Pred: {classes[pred_core_soft[i]]}", fontsize=10)
                    axes[i, 2].axis('off')

                    axes[i, 3].imshow(images_corrupted_spurious_soft[i].permute(1, 2, 0))
                    axes[i, 3].set_title(f"Spurious Soft {i}, Pred: {classes[pred_spur_soft[i]]}", fontsize=10)
                    axes[i, 3].axis('off')

                plt.tight_layout()

                if not os.path.exists(path):
                    os.makedirs(path)
                if not os.path.exists(f"{path}/plot/soft_mask"):
                    os.makedirs(f"{path}/plot/soft_mask")

                plt.savefig(f"{path}/plot/soft_mask/{split}_plot_soft_mask_core_spurious_accuracy_sigma_{sigma}_artifact_{artifact_labels[k]}.png", dpi=400)
                plt.savefig(f"{path}/plot/soft_mask/{split}_plot_soft_mask_core_spurious_accuracy_sigma_{sigma}_artifact_{artifact_labels[k]}.jpeg", dpi=10)
                plt.savefig(f"{path}/plot/soft_mask/{split}_plot_soft_mask_core_spurious_accuracy_sigma_{sigma}_artifact_{artifact_labels[k]}.pdf" , dpi=400)
                plt.savefig(f"{path}/plot/soft_mask/{split}_plot_soft_mask_core_spurious_accuracy_sigma_{sigma}_artifact_{artifact_labels[k]}.svg" , dpi=400)

                if config.get('wandb_api_key', None):
                    path_to_save = f"{path}/plot/soft_mask/{split}_plot_soft_mask_core_spurious_accuracy_sigma_{sigma}_artifact_{artifact_labels[k]}.jpeg"
                    wandb.log({f"{split}_plot_core_spurious_accuracy_sigma_{sigma}_artifact_{artifact_labels[k]}": wandb.Image(path_to_save)})
                plt.close()

                # Plot but with logit values and no masks
                # Plot:
                # Image (ID, Target, Pred) underneath Logits, Corrupted Images (Core/Spurious + Soft/Hard) underneath Logits
                fig, axes = plt.subplots(n_samples_to_plot, 5, figsize=(20, 4 * n_samples_to_plot))
                plt.axis('off')
                plt.suptitle(f"Split: {split}, Artifact: {artifact_labels[k]}, Sigma: {sigma} -- Logits\nClasses: {classes}", fontsize=10)
                for i in range(n_samples_to_plot):
                    axes[i, 0].imshow(images_to_plot[i].permute(1, 2, 0))
                    axes[i, 0].set_title(f"Image {idxs[i]}\nTarget: {classes[target_to_plot[i]]}\nLogits: {format_logits(logits_image[i])}", fontsize=8)
                    axes[i, 0].axis('off')

                    axes[i, 1].imshow(images_corrupted_core_soft[i].permute(1, 2, 0))
                    axes[i, 1].set_title(f"Core Soft {i}, Pred: {classes[pred_core_soft[i]]}\nLogits: {format_logits(logits_corrupted_core_soft[i])}", fontsize=8)
                    axes[i, 1].axis('off')

                    axes[i, 2].imshow(images_corrupted_spurious_soft[i].permute(1, 2, 0))
                    axes[i, 2].set_title(f"Spurious Soft {i}, Pred: {classes[pred_spur_soft[i]]}\nLogits: {format_logits(logits_corrupted_spurious_soft[i])}", fontsize=8)
                    axes[i, 2].axis('off')

                    axes[i, 3].imshow(images_corrupted_core_hard[i].permute(1, 2, 0))
                    axes[i, 3].set_title(f"Core Hard {i}, Pred: {classes[pred_core_hard[i]]}\nLogits: {format_logits(logits_corrupted_core_hard[i])}", fontsize=8)
                    axes[i, 3].axis('off')

                    axes[i, 4].imshow(images_corrupted_spurious_hard[i].permute(1, 2, 0))
                    axes[i, 4].set_title(f"Spurious Hard {i}, Pred: {classes[pred_spur_hard[i]]}\nLogits: {format_logits(logits_corrupted_spurious_hard[i])}", fontsize=8)
                    axes[i, 4].axis('off')
                plt.tight_layout()

                if not os.path.exists(f"{path}/plot_logits"):
                    os.makedirs(f"{path}/plot_logits")

                plt.savefig(f"{path}/plot_logits/{split}_plot_core_spurious_accuracy_logits_sigma_{sigma}_artifact_{artifact_labels[k]}.png", dpi=400)
                plt.savefig(f"{path}/plot_logits/{split}_plot_core_spurious_accuracy_logits_sigma_{sigma}_artifact_{artifact_labels[k]}.jpeg", dpi=10)
                plt.savefig(f"{path}/plot_logits/{split}_plot_core_spurious_accuracy_logits_sigma_{sigma}_artifact_{artifact_labels[k]}.pdf" , dpi=400)
                plt.savefig(f"{path}/plot_logits/{split}_plot_core_spurious_accuracy_logits_sigma_{sigma}_artifact_{artifact_labels[k]}.svg" , dpi=400)

                if config.get('wandb_api_key', None):
                    path_to_save = f"{path}/plot_logits/{split}_plot_core_spurious_accuracy_logits_sigma_{sigma}_artifact_{artifact_labels[k]}.jpeg"
                    wandb.log({f"{split}_plot_core_spurious_accuracy_logits_sigma_{sigma}_artifact_{artifact_labels[k]}": wandb.Image(path_to_save)})
                plt.close()

                fig, axes = plt.subplots(n_samples_to_plot, 3, figsize=(20, 4 * n_samples_to_plot))
                plt.axis('off')
                plt.suptitle(f"Split: {split}, Artifact: {artifact_labels[k]}, Sigma: {sigma} -- Logits\nClasses: {classes}", fontsize=10)
                for i in range(n_samples_to_plot):
                    axes[i, 0].imshow(images_to_plot[i].permute(1, 2, 0))
                    axes[i, 0].set_title(f"Image {idxs[i]}\nTarget: {classes[target_to_plot[i]]}\nLogits: {format_logits(logits_image[i])}", fontsize=8)
                    axes[i, 0].axis('off')

                    axes[i, 1].imshow(images_corrupted_core_soft[i].permute(1, 2, 0))
                    axes[i, 1].set_title(f"Core Soft {i}, Pred: {classes[pred_core_soft[i]]}\nLogits: {format_logits(logits_corrupted_core_soft[i])}", fontsize=8)
                    axes[i, 1].axis('off')

                    axes[i, 2].imshow(images_corrupted_spurious_soft[i].permute(1, 2, 0))
                    axes[i, 2].set_title(f"Spurious Soft {i}, Pred: {classes[pred_spur_soft[i]]}\nLogits: {format_logits(logits_corrupted_spurious_soft[i])}", fontsize=8)
                    axes[i, 2].axis('off')
                plt.tight_layout()

                if not os.path.exists(f"{path}/plot_logits/soft_mask"):
                    os.makedirs(f"{path}/plot_logits/soft_mask")

                plt.savefig(f"{path}/plot_logits/soft_mask/{split}_plot_soft_mask_core_spurious_accuracy_logits_sigma_{sigma}_artifact_{artifact_labels[k]}.png", dpi=400)
                plt.savefig(f"{path}/plot_logits/soft_mask/{split}_plot_soft_mask_core_spurious_accuracy_logits_sigma_{sigma}_artifact_{artifact_labels[k]}.jpeg", dpi=10)
                plt.savefig(f"{path}/plot_logits/soft_mask/{split}_plot_soft_mask_core_spurious_accuracy_logits_sigma_{sigma}_artifact_{artifact_labels[k]}.pdf" , dpi=400)
                plt.savefig(f"{path}/plot_logits/soft_mask/{split}_plot_soft_mask_core_spurious_accuracy_logits_sigma_{sigma}_artifact_{artifact_labels[k]}.svg" , dpi=400)

                if config.get('wandb_api_key', None):
                    path_to_save = f"{path}/plot_logits/soft_mask/{split}_plot_soft_mask_core_spurious_accuracy_logits_sigma_{sigma}_artifact_{artifact_labels[k]}.jpeg"
                    wandb.log({f"{split}_plot_core_spurious_accuracy_logits_sigma_{sigma}_artifact_{artifact_labels[k]}": wandb.Image(path_to_save)})
                plt.close()

                fig, axes = plt.subplots(n_samples_to_plot, 3, figsize=(20, 4 * n_samples_to_plot))
                plt.axis('off')
                plt.suptitle(f"Split: {split}, Artifact: {artifact_labels[k]}, Sigma: {sigma} -- Logits\nClasses: {classes}", fontsize=10)
                for i in range(n_samples_to_plot):
                    axes[i, 0].imshow(images_to_plot[i].permute(1, 2, 0))
                    axes[i, 0].set_title(f"Image {idxs[i]}\nTarget: {classes[target_to_plot[i]]}\nLogits: {format_logits(logits_image[i])}", fontsize=8)
                    axes[i, 0].axis('off')

                    axes[i, 1].imshow(images_corrupted_core_hard[i].permute(1, 2, 0))
                    axes[i, 1].set_title(f"Core Hard {i}, Pred: {classes[pred_core_hard[i]]}\nLogits: {format_logits(logits_corrupted_core_hard[i])}", fontsize=8)
                    axes[i, 1].axis('off')

                    axes[i, 2].imshow(images_corrupted_spurious_hard[i].permute(1, 2, 0))
                    axes[i, 2].set_title(f"Spurious Hard {i}, Pred: {classes[pred_spur_hard[i]]}\nLogits: {format_logits(logits_corrupted_spurious_hard[i])}", fontsize=8)
                    axes[i, 2].axis('off')
                plt.tight_layout()

                if not os.path.exists(f"{path}/plot_logits/hard_mask"):
                    os.makedirs(f"{path}/plot_logits/hard_mask")

                plt.savefig(f"{path}/plot_logits/hard_mask/{split}_plot_hard_mask_core_spurious_accuracy_logits_sigma_{sigma}_artifact_{artifact_labels[k]}.png", dpi=400)
                plt.savefig(f"{path}/plot_logits/hard_mask/{split}_plot_hard_mask_core_spurious_accuracy_logits_sigma_{sigma}_artifact_{artifact_labels[k]}.jpeg", dpi=10)
                plt.savefig(f"{path}/plot_logits/hard_mask/{split}_plot_hard_mask_core_spurious_accuracy_logits_sigma_{sigma}_artifact_{artifact_labels[k]}.pdf" , dpi=400)
                plt.savefig(f"{path}/plot_logits/hard_mask/{split}_plot_hard_mask_core_spurious_accuracy_logits_sigma_{sigma}_artifact_{artifact_labels[k]}.svg" , dpi=400)

                if config.get('wandb_api_key', None):
                    path_to_save = f"{path}/plot_logits/hard_mask/{split}_plot_hard_mask_core_spurious_accuracy_logits_sigma_{sigma}_artifact_{artifact_labels[k]}.jpeg"
                    wandb.log({f"{split}_plot_core_spurious_accuracy_logits_sigma_{sigma}_artifact_{artifact_labels[k]}": wandb.Image(path_to_save)})
                plt.close()



    if config.get('wandb_api_key', None):
        wandb.log(results)

def format_logits(logits, precision=2, max_len=50):
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    logits_str = np.array2string(np.round(logits, precision), separator=', ')
    return "\n".join(textwrap.wrap(logits_str, width=max_len))


if __name__ == '__main__':
    main()