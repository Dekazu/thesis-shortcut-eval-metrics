import os
import random
from argparse import ArgumentParser
from typing import List

import numpy as np
import torch
import torchvision
import wandb
import yaml
from tqdm import tqdm

from datasets import get_dataset
from experiments.evaluation.compute_metrics import compute_metrics
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_fn_model_loader

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)


def get_args(fixed_arguments: List[str] = []):
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="vgg16")
    parser.add_argument("--ckpt_path",
                        # default="/home/fe/dreyer/projects/cxxai_paper_figures/figure_11_isic_plaster/code/models/isic2019-vgg16.pth")
                        # default="results/fixing/GClarc/last-v5.ckpt")
                        default=None)
    parser.add_argument("--dataset_name", default="isic_hm")
    parser.add_argument("--data_path", default="/hardd/datasets/ISIC")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--artifact", default="band_aid")
    parser.add_argument('--config_file',
                        default="config_files/fixing_bone/local/vgg16_GLocClarc2_sgd_lr0.0001_lamb1000_small_l_local.yaml")
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
    args = get_args(fixed_arguments=[
        "ckpt_path",
        "artifact"
    ])
    dataset_name = args.dataset_name + "_hm"
    model_name = args.model_name

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(id=args.wandb_id, project=args.wandb_project_name, resume=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = get_dataset(dataset_name)(path=args.data_path, normalize_data=True, split="train", artifact=args.artifact)

    n_classes = len(dataset.classes)

    ckpt_path = args.ckpt_path if args.ckpt_path else f"checkpoints/{os.path.basename(args.config_file)[:-5]}/last.ckpt"
    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path, device=device)
    model = prepare_model_for_evaluation(model, dataset, device, args)

    gaussian = torchvision.transforms.GaussianBlur(kernel_size=141, sigma=7.0)
    flip = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
    ])

    ### COLLECT ARTIFACTS

    artifact_samples = dataset.artifact_ids

    print(f"There are {len(artifact_samples)} artifact samples")
    for k, samples in enumerate([artifact_samples]):

        n_samples = len(samples)
        n_batches = int(np.ceil(n_samples / args.batch_size))

        y_pred = []
        y_target = []
        for i in tqdm(range(n_batches)):
            samples_batch = samples[i * args.batch_size:(i + 1) * args.batch_size]
            data = torch.stack([dataset[j][0] for j in samples_batch], dim=0)
            mask = torch.stack([dataset[j][2] for j in samples_batch])
            mask = gaussian(mask.clamp(min=0)) ** 1.0
            mask = mask / mask.abs().flatten(start_dim=1).max(1)[0][:, None, None]
            mask = mask > 0.2
            mask = 1 - 1 * mask[:, None, :, :]
            data_ = data * mask + (1 - mask) * data.flatten(start_dim=2).median(2)[0][..., None, None]

            out = model(data_.to(device)).detach().cpu()

            targets = torch.tensor([dataset[j][1] for j in samples_batch])

            y_pred.append(out)
            y_target.append(targets)
            # plt.imshow(1 / 255 * dataset.reverse_normalization(data[0]).detach().cpu().permute((1, 2, 0)))
            # plt.show()
            #
            # plt.imshow(mask[0].detach().cpu())
            # plt.show()
            #
            # plt.imshow(1 / 255 * dataset.reverse_normalization(data[0] * mask[0][None].to(data)).detach().cpu().permute(
            #     (1, 2, 0)))
            # plt.show()
            #
            # plt.imshow(1 / 255 * dataset.reverse_normalization(data_[0]).detach().cpu().permute((1, 2, 0)))
            # plt.show()
            print("")

        y_pred = torch.cat(y_pred, 0)
        y_target = torch.cat(y_target, 0)

        metrics = compute_metrics(y_pred, y_target, None, prefix=f"auto-cleansed_{args.artifact}_")
        print(metrics)
        if args.wandb_api_key:
            wandb.log(metrics)


if __name__ == "__main__":
    main()
