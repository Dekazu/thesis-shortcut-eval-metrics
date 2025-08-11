import glob
import os
from argparse import ArgumentParser
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
import tqdm
import wandb
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from datasets import get_dataset, do_custom_split
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
    parser.add_argument("--data_path", default="/hardd/datasets/ISIC")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument('--config_file', default="config_files/fixing/local/vgg16_Vanilla_sgd_lr0.0005.yaml")
    parser.add_argument('--wandb_api_key', default="3e98f9f11a0e6b8baac5b064a66ef2af0c0bfed9")
    parser.add_argument('--wandb_project_name', default="crp-miccai-fixing")
    parser.add_argument('--segmentation_dir', default="crp-miccai-fixing")
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


transform_mask = T.Compose([
    T.Resize((224, 224), interpolation=T.functional.InterpolationMode.BICUBIC),
    T.ToTensor()
])


def load_mask(name_mask, transform_mask):
    mask = Image.open(name_mask)
    mask = transform_mask(mask)
    mask = (mask.mean(0) > .5).type(torch.uint8)

    return mask


def load_img(img_name, transform):
    img = Image.open(img_name)
    img = transform(img)

    return img


def main():
    args = get_args(
        fixed_arguments=[
            "ckpt_path"
        ])  #
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

    dataset = get_dataset(dataset_name)(path=args.data_path, normalize_data=True, split="train")
    dataset_train, dataset_val = do_custom_split(dataset, f"data/idxs_train.npy")

    segmenation_dir = args.segmentation_dir
    path_bandaid_masks = f"{segmenation_dir}/segmentation_masks"
    path_bandaid_imgs = f"{segmenation_dir}/imgs"

    all_imgs = [path.split("/")[-1] for path in glob.glob(f"{path_bandaid_imgs}/*.jpg")]
    all_masks = [path.split("/")[-1] for path in glob.glob(f"{path_bandaid_masks}/*.jpg")]

    all_masks = np.array([mask for mask in all_masks if mask in all_imgs])

    for split in [
        'train',
        'val'
    ]:

        outs_clean_all, outs_attacked_all = [], []
        ys_clean_all, ys_attacked_all = [], []

        dataset_split = dataset_train if split == 'train' else dataset_val

        dl = DataLoader(dataset_split, batch_size=args.batch_size, shuffle=False)

        for batch_x, batch_y in tqdm.tqdm(dl):
            pick = np.random.choice(range(len(all_masks)), len(batch_x))

            m = torch.stack(
                [load_mask(f"{path_bandaid_masks}/{all_masks[i]}", transform_mask)[None, :, :] for i in pick])
            batch_artifact = torch.stack(
                [load_img(f"{path_bandaid_imgs}/{all_masks[i]}", dataset_split.transform) for i in pick])

            batch_x_attacked = batch_x * (1 - m) + batch_artifact * m

            out_attacked = model(batch_x_attacked.to(device)).detach().cpu()
            out_clean = model(batch_x.to(device)).detach().cpu()

            outs_clean_all.append(out_clean)
            outs_attacked_all.append(out_attacked)
            ys_clean_all.append(batch_y)
            ys_attacked_all.append(batch_y)

        outs_clean_all = torch.cat(outs_clean_all)
        outs_attacked_all = torch.cat(outs_attacked_all)
        ys_clean_all = torch.cat(ys_clean_all)
        ys_attacked_all = torch.cat(ys_attacked_all)

        metrics_clean_all = compute_metrics(outs_clean_all, ys_clean_all, None, prefix=f"clean_{split}_")
        metrics_attacked_all = compute_metrics(outs_attacked_all, ys_attacked_all, None, prefix=f"attacked_{split}_")

        if args.wandb_api_key:
            wandb.log({**metrics_clean_all, **metrics_attacked_all})


if __name__ == "__main__":
    main()
