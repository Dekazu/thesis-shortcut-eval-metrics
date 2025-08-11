import copy
import glob
import os
from argparse import ArgumentParser
from typing import List

import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from datasets import get_dataset, do_custom_split
from experiments.evaluation.compute_metrics import compute_metrics, compute_model_scores
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

    for split in [
        'train',
        'val'
    ]:

        outs_clean_all, outs_attacked_all = [], []
        ys_clean_all, ys_attacked_all = [], []

        dataset_split = dataset_train if split == 'train' else dataset_val
        for attacked_class in [
            'MEL', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC'
        ]:
            dataset_clean, dataset_attacked = get_attacked_dataset(dataset_split,
                                                                   f"data/attacked_samples/{split}/{attacked_class}")

            print("size of sample set", len(dataset_attacked))

            dl_clean = DataLoader(dataset_clean, batch_size=args.batch_size, shuffle=False)
            dl_attacked = DataLoader(dataset_attacked, batch_size=args.batch_size, shuffle=False)

            model_outs_clean, y_true_clean = compute_model_scores(model, dl_clean, device)
            model_outs_attacked, y_true_attacked = compute_model_scores(model, dl_attacked, device)

            print(y_true_clean)
            metrics_clean = compute_metrics(model_outs_clean, y_true_clean, None, prefix=f"clean_{split}_",
                                            suffix=f"_{attacked_class.lower()}")
            metrics_attacked = compute_metrics(model_outs_attacked, y_true_attacked, None, prefix=f"attacked_{split}_",
                                               suffix=f"_{attacked_class.lower()}")

            outs_clean_all.append(model_outs_clean)
            outs_attacked_all.append(model_outs_attacked)
            ys_clean_all.append(y_true_clean)
            ys_attacked_all.append(y_true_attacked)

            if args.wandb_api_key:
                wandb.log({**metrics_clean, **metrics_attacked})

        outs_clean_all = torch.cat(outs_clean_all)
        outs_attacked_all = torch.cat(outs_attacked_all)
        ys_clean_all = torch.cat(ys_clean_all)
        ys_attacked_all = torch.cat(ys_attacked_all)

        metrics_clean_all = compute_metrics(outs_clean_all, ys_clean_all, None, prefix=f"clean_{split}_")
        metrics_attacked_all = compute_metrics(outs_attacked_all, ys_attacked_all, None, prefix=f"attacked_{split}_")

        if args.wandb_api_key:
            wandb.log({**metrics_clean_all, **metrics_attacked_all})


def get_attacked_dataset(dataset, path_samples_attacked):
    fnames_samples_ch = [fname.split('/')[-1][:-4] for fname in glob.glob(f"{path_samples_attacked}/*")]
    print(fnames_samples_ch)
    idxs_pois = [i for i, path in enumerate(dataset.ids.image.values) if path in fnames_samples_ch]
    dataset_clean = dataset.get_subset_by_idxs(idxs_pois)
    dataset_attacked = copy.deepcopy(dataset_clean)
    dataset_attacked.train_dir = path_samples_attacked
    return dataset_clean, dataset_attacked


if __name__ == "__main__":
    main()
