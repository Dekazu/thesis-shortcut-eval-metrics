import os
from argparse import ArgumentParser
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
import wandb
import yaml
from tqdm import tqdm

from datasets import get_dataset
from experiments.evaluation.compute_metrics import compute_metrics
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_fn_model_loader

torch.random.manual_seed(0)


def get_args(fixed_arguments: List[str] = []):
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="vgg16")
    parser.add_argument("--ckpt_path",
                        # default="/home/fe/dreyer/projects/cxxai_paper_figures/figure_11_isic_plaster/code/models/isic2019-vgg16.pth")
                        # default="results/fixing/GClarc/last-v5.ckpt")
                        default=None)
    parser.add_argument("--dataset_name", default="isic")
    parser.add_argument("--fname_idx_train", default="data/idxs_train.npy")
    parser.add_argument("--data_path", default="/hardd/datasets/ISIC")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument('--config_file', default="config_files/fixing/local/vgg16_Vanilla_sgd_lr0.0005_ep10_local.yaml")
    parser.add_argument('--wandb_api_key', default="3e98f9f11a0e6b8baac5b064a66ef2af0c0bfed9")
    parser.add_argument('--wandb_project_name', default="crp-miccai-fixing")
    parser.add_argument('--wandb_id', default=None)

    args = parser.parse_args()

    with open(parser.parse_args().config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["wandb_id"] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    for k, v in config.items():
        if k not in fixed_arguments:
            setattr(args, k, v)

    return args


def main():
    args = get_args(fixed_arguments=["ckpt_path"])
    dataset_name = args.dataset_name
    model_name = args.model_name

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(id=args.wandb_id, project=args.wandb_project_name, resume=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = get_dataset(dataset_name)(path=args.data_path, normalize_data=True)

    n_classes = len(dataset.classes)

    ckpt_path = args.ckpt_path if args.ckpt_path else f"checkpoints/{os.path.basename(args.config_file)[:-5]}/last.ckpt"
    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path, device=device)
    model = prepare_model_for_evaluation(model, dataset, device, args)

    labels_set = ["BAND_AID", "SKIN_MARKER", "RULER", "CLEAN"]
    sample_sets = [dataset.BAND_AID, dataset.SKIN_MARKER, dataset.RULER, dataset.CLEAN]

    train_set = np.load(args.fname_idx_train)
    val_set = [i for i in range(len(dataset)) if i not in train_set]
    mask = T.Compose([T.CenterCrop((124, 124)), T.Resize((224, 224))])
    for split in ['train', 'val']:
        model_outs_all = []
        ys_all = []

        split_set = train_set if split == 'train' else val_set
        sample_sets_split = [[y for y in x if y in split_set] for x in sample_sets]

        for k, samples in enumerate(sample_sets_split):

            n_samples = min(len(samples), 2000)
            n_batches = int(np.ceil(n_samples / args.batch_size))

            y_pred = []
            y_target = []

            for i in tqdm(range(n_batches)):
                samples_batch = samples[i * args.batch_size:(i + 1) * args.batch_size]
                data = torch.stack([dataset[j][0] for j in samples_batch], dim=0).to(device)
                data_ = mask(data)
                out = model(data_).detach().cpu()

                # plt.imshow(1 / 255 * dataset.reverse_normalization(data_[0]).detach().cpu().permute((1, 2, 0)))
                # plt.show()
                # plt.imshow(masks[0, 0].detach().cpu())
                # plt.show()
                targets = torch.tensor([dataset[j][1] for j in samples_batch])

                y_pred.append(out)
                y_target.append(targets)

            y_pred = torch.cat(y_pred, 0)
            y_target = torch.cat(y_target, 0)

            model_outs_all.append(y_pred)
            ys_all.append(y_target)

            metrics = compute_metrics(y_pred, y_target, prefix=f"{split}_foreground_",
                                      suffix=f"_{labels_set[k].lower()}")
            # print(metrics)
            if args.wandb_api_key:
                wandb.log(metrics)

        model_outs_all = torch.cat(model_outs_all)
        ys_all = torch.cat(ys_all)

        metrics_all = compute_metrics(model_outs_all, ys_all, prefix=f"{split}_foreground_")
        if args.wandb_api_key:
            wandb.log(metrics_all)

            wandb.log({f"foreground_roc_curve_{split}": wandb.plot.roc_curve(ys_all, model_outs_all,
                                                                             labels=dataset.classes,
                                                                             title=f"Foreground ROC ({split})")})
            wandb.log(
                {f"foreground_pr_curve_{split}": wandb.plot.pr_curve(ys_all, model_outs_all, labels=dataset.classes,
                                                                     title=f"Foreground Precision/Recall ({split})")})


if __name__ == "__main__":
    main()
