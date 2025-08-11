import os
from argparse import ArgumentParser
from typing import List

import numpy as np
import torch
import torchvision.transforms
import wandb
import yaml
from crp.attribution import CondAttribution
from crp.helper import get_layer_names
from tqdm import tqdm
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_canonizer, get_fn_model_loader

torch.random.manual_seed(0)


def get_args(fixed_arguments: List[str] = []):
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="vgg16")
    parser.add_argument("--ckpt_path",
                        # default="/home/fe/dreyer/projects/cxxai_paper_figures/figure_11_isic_plaster/code/models/isic2019-vgg16.pth")
                        # default="/hardd/projects/crp-miccai/checkpoints/vgg16_RRR/last.ckpt")
                        default=None)
    parser.add_argument("--dataset_name", default="isic")
    parser.add_argument("--data_path", default="/hardd/datasets/ISIC")
    parser.add_argument("--artifacts_file", default=None)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument('--config_file',
                        default="config_files/fixing_bone/local/vgg16_GLocClarc2_sgd_lr0.0001_lamb1000_small_l_local.yaml")
    parser.add_argument('--wandb_api_key', default="119fe3ea513aead1b4d78dfc1f90967d04b56cc1")
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
        "batch_size"
    ])
    dataset_name = args.dataset_name
    model_name = args.model_name

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(id=args.wandb_id, project=args.wandb_project_name, resume=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = {"p_artifact": args.p_artifact, 
              "attacked_classes": args.attacked_classes} if args.artifact == 'artificial' else {}
    dataset = get_dataset(dataset_name)(data_paths=args.data_paths,
                                        normalize_data=True,
                                        artifact_ids_file=args.artifacts_file,
                                        artifact=args.artifact,
                                        **kwargs)

    n_classes = len(dataset.classes)

    ckpt_path = args.ckpt_path if args.ckpt_path else f"checkpoints/{os.path.basename(args.config_file)[:-5]}/last.ckpt"
    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path, device=device)
    model = prepare_model_for_evaluation(model, dataset, device, args)

    attribution = CondAttribution(model)
    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)

    layer_names = get_layer_names(model, [torch.nn.Conv2d])

    labels_set = list(dataset.sample_ids_by_artifact.keys()) + ["CLEAN"]
    sample_sets = list(dataset.sample_ids_by_artifact.values()) + [dataset.CLEAN]
    scores = []

    train_set = np.load(args.fname_idx_train)
    val_set = [i for i in range(len(dataset)) if i not in train_set]

    for split in ['train', 'val']:

        split_set = train_set if split == 'train' else val_set
        sample_sets_split = [[y for y in x if y in split_set] for x in sample_sets]
        mask = torchvision.transforms.CenterCrop((124, 124))
        for k, samples in enumerate(sample_sets_split):

            n_samples = min(len(samples), 2000)
            n_batches = int(np.ceil(n_samples / args.batch_size))

            score = []
            for i in tqdm(range(n_batches)):
                samples_batch = samples[i * args.batch_size:(i + 1) * args.batch_size]
                data = torch.stack([dataset[j][0] for j in samples_batch], dim=0).to(device).requires_grad_()
                out = model(data).detach().cpu()
                condition = [{"y": c_id} for c_id in out.argmax(1)]

                attr = attribution(data, condition, composite, record_layer=layer_names, init_rel=1)

                inside = (mask(attr.heatmap)).abs().sum((1, 2)) / (
                        attr.heatmap.abs().sum((1, 2)) + 1e-10)
                score.extend(list(1 - inside.detach().cpu()))

            scores.append(np.mean(score))
            print(labels_set[k], scores[-1])
            if args.wandb_api_key:
                wandb.log({f"{split}_background_rel_{labels_set[k].lower()}": scores[-1]})

        for score in scores:
            wandb.log({f"{split}_background_rel": score})


if __name__ == "__main__":
    main()
