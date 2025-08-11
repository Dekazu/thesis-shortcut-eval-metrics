import os
import math
import gc
import numpy as np
import torch
from argparse import ArgumentParser
from collections import Counter
from torch.utils.data import DataLoader
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import pairwise_distances
import tqdm
import wandb

from datasets import load_dataset
from models import get_fn_model_loader
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from utils.helper import load_config

torch.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--config_file',
        default="config_files/evaluate_metrics_clean/waterbirds_clean/local/"
                "waterbirds_clean_resnet50d_p_artifact_landbirds0.05_"
                "p_artifact_waterbirds0.95_last_conv_baseline.yaml"
    )
    parser.add_argument('--before_correction', action="store_true")
    return parser.parse_args()


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    """
    Numerically‑stable softmax. Handles both binary (1D scores) and multi‑class (2D scores).
    """
    scores = np.asarray(scores)
    # If binary, convert shape (N,) → (N, 2) by treating scores as distance to class 1
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    # subtract max for stability
    m = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - m)
    denom = exp_scores.sum(axis=1, keepdims=True)
    # avoid divide‑by‑zero → fall back to uniform
    denom[denom == 0] = exp_scores.shape[1]
    return exp_scores / denom


def compute_mi_from_probs(prob_te: np.ndarray, y_te: np.ndarray) -> float:
    classes = np.unique(y_te)
    p_emp = Counter(y_te)
    N_te = len(y_te)

    mi_sum = 0.0
    for i in range(N_te):
        for j, cls in enumerate(classes):
            p_yf = prob_te[i, j]
            if p_yf <= 1e-12:
                continue
            p_y = p_emp[cls] / N_te
            mi_sum += p_yf * math.log(p_yf / p_y)
    return mi_sum / N_te


def compute_alignment_loss(features: np.ndarray,
                           y_labels: np.ndarray,
                           bias_labels: np.ndarray) -> float:
    unique_y = np.unique(y_labels)
    losses = []

    for y in unique_y:
        idx = np.where(y_labels == y)[0]
        if len(idx) < 2:
            continue

        feats = features[idx]
        biases = bias_labels[idx]
        uniq_bs = np.unique(biases)
        if len(uniq_bs) < 2:
            continue

        pair_ls = []
        for i in range(len(uniq_bs)):
            for j in range(i + 1, len(uniq_bs)):
                g1 = feats[biases == uniq_bs[i]]
                g2 = feats[biases == uniq_bs[j]]
                if g1.size == 0 or g2.size == 0:
                    continue
                dists = pairwise_distances(g1, g2, metric='euclidean') ** 2
                pair_ls.append(dists.mean())

        if pair_ls:
            losses.append(np.mean(pair_ls))

    return float(np.mean(losses)) if losses else 0.0


def extract_and_train_sgd_classifiers(model, dataloader, bias_labels, n_classes, device):
    sgd_y = SGDClassifier(loss="log_loss", max_iter=5, tol=None, random_state=0)
    sgd_z = SGDClassifier(loss="log_loss", max_iter=5, tol=None, random_state=1)
    sgd_joint = SGDClassifier(loss="log_loss", max_iter=5, tol=None, random_state=2)

    max_z = int(np.max(bias_labels))
    idx = 0

    for x_batch, y_batch in tqdm.tqdm(dataloader, desc="SGD training"):
        x_batch = x_batch.to(device)
        with torch.no_grad():
            _, feat = model.forward_with_features(x_batch)
        feat = feat.cpu().numpy()
        y = y_batch.cpu().numpy()
        z = np.array(bias_labels[idx:idx + len(y)])
        joint = y * (max_z + 1) + z

        sgd_y.partial_fit(feat, y, classes=np.arange(n_classes))
        sgd_z.partial_fit(feat, z, classes=np.arange(2))
        sgd_joint.partial_fit(feat, joint,
                              classes=np.arange(n_classes * (max_z + 1)))

        idx += len(y)
        del x_batch, y_batch, feat
        torch.cuda.empty_cache()

    return sgd_y, sgd_z, sgd_joint, max_z


def compute_feature_metrics_with_trained_classifiers(
    feature_batches,
    sgd_y, sgd_z, sgd_joint,
    max_z,
    bias_labels,
    prefix="", suffix=""
):
    # gather all features and labels
    all_feats = []
    all_y = []
    all_z = []
    idx = 0

    for feat_batch, y_batch, _ in feature_batches:
        z_batch = bias_labels[idx:idx + len(y_batch)]
        all_feats.append(feat_batch)
        all_y.append(y_batch)
        all_z.append(z_batch)
        idx += len(y_batch)

    all_feats = np.vstack(all_feats)
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)

    # normalized for alignment loss
    feats_norm = all_feats / (np.linalg.norm(all_feats, axis=1, keepdims=True) + 1e-12)

    # compute probabilities via stable softmax on raw decision scores
    prob_fy    = stable_softmax(sgd_y.decision_function(all_feats))
    prob_fz    = stable_softmax(sgd_z.decision_function(all_feats))
    prob_joint = stable_softmax(sgd_joint.decision_function(all_feats))

    joint_labels = all_y * (max_z + 1) + all_z

    mi_fy      = compute_mi_from_probs(prob_fy,    all_y)
    mi_fz      = compute_mi_from_probs(prob_fz,    all_z)
    mi_f_joint = compute_mi_from_probs(prob_joint, joint_labels)

    cmi_fz_y   = mi_f_joint - mi_fy
    cobias_val = mi_f_joint - mi_fy
    alignment_l= compute_alignment_loss(feats_norm, all_y, all_z)

    return {
        f"{prefix}mi_fy{suffix}":           mi_fy,
        f"{prefix}mi_fz{suffix}":           mi_fz,
        f"{prefix}joint_mi_f_yz{suffix}":   mi_f_joint,
        f"{prefix}cmi_fz_y{suffix}":        cmi_fz_y,
        f"{prefix}cobias{suffix}":          cobias_val,
        f"{prefix}alignment_loss{suffix}":  alignment_l,
    }


def evaluate_by_subset_attacked(config, before_correction):
    config_name = config["config_name"]
    print(f"Evaluating {config_name} on device...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_dataset(config, normalize_data=True)
    n_classes = len(dataset.classes)
    ckpt_path = (
        config['ckpt_path']
        if before_correction
        else f"{config['checkpoint_dir_corrected']}/{config_name}/last.ckpt"
    )

    model = get_fn_model_loader(config['model_name'])(
        n_class=n_classes,
        ckpt_path=ckpt_path,
        device=device
    )
    model = prepare_model_for_evaluation(model, dataset, device, config)

    sets = {
        'train': dataset.idxs_train,
        'val':   dataset.idxs_val,
        'test':  dataset.idxs_test,
    }

    # Step 1: train SGD classifiers
    train_idxs = sets['train']
    train_set = dataset.get_subset_by_idxs(train_idxs)
    bias_labels_train = [1 if x % 2 == 1 else 0 for x in dataset.groups[train_idxs]]
    train_dl = DataLoader(train_set, batch_size=config['batch_size'], shuffle=False)

    print("Training SGD classifiers on train set...")
    sgd_y, sgd_z, sgd_joint, max_z = extract_and_train_sgd_classifiers(
        model, train_dl, bias_labels_train, n_classes, device
    )

    # Step 2: evaluate on val and test
    for split in ['val', 'test']:
        print(f"Evaluating on {split} set...")
        split_idxs = sets[split]
        split_set = dataset.get_subset_by_idxs(split_idxs)
        bias_labels = [1 if x % 2 == 1 else 0 for x in dataset.groups[split_idxs]]
        dl = DataLoader(split_set, batch_size=config['batch_size'], shuffle=False)

        def feature_generator():
            for x_batch, y_batch in tqdm.tqdm(dl):
                x_batch = x_batch.to(device)
                with torch.no_grad():
                    out, feat = model.forward_with_features(x_batch)
                yield feat.cpu().numpy(), y_batch.cpu().numpy(), out.cpu().numpy()
                del x_batch, y_batch, out, feat
                torch.cuda.empty_cache()

        metrics = compute_feature_metrics_with_trained_classifiers(
            feature_generator(),
            sgd_y, sgd_z, sgd_joint, max_z,
            bias_labels=bias_labels,
            prefix=f"{split}_", suffix="_stream"
        )

        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
            if config.get('wandb_api_key', None):
                wandb.log({key: value})


def main():
    args = get_args()
    config = load_config(args.config_file)

    if config.get('wandb_api_key', None):
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
