import os
from argparse import ArgumentParser
from typing import List

import numpy as np
import torch
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
    parser.add_argument("--dataset_name", default="isic_seg")
    parser.add_argument("--data_path", default="/hardd/datasets/ISIC")
    parser.add_argument("--fname_idx_train", default="data/idxs_train.npy")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument('--config_file', default="config_files/fixing/local/vgg16_Vanilla_sgd_lr0.0001_ep7_local.yaml")
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
    args = get_args(fixed_arguments=["ckpt_path", "dataset_name", "batch_size"])
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

    attribution = CondAttribution(model)
    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)

    layer_names = get_layer_names(model, [torch.nn.Conv2d])

    labels_set = ["BAND_AID", "SKIN_MARKER", "RULER", "CLEAN"]
    sample_sets = [dataset.BAND_AID, dataset.SKIN_MARKER, dataset.RULER, dataset.CLEAN]
    scores = []

    train_set = np.load(args.fname_idx_train)
    val_set = [i for i in range(len(dataset)) if i not in train_set]

    for split in ['train', 'val']:

        split_set = train_set if split == 'train' else val_set
        sample_sets_split = [[y for y in x if y in split_set] for x in sample_sets]

        for k, samples in enumerate(sample_sets_split):

            n_samples = len(samples)
            n_batches = int(np.ceil(n_samples / args.batch_size))

            score = []
            for i in tqdm(range(n_batches)):
                samples_batch = samples[i * args.batch_size:(i + 1) * args.batch_size]
                data = torch.stack([dataset[j][0] for j in samples_batch], dim=0).to(device).requires_grad_()
                out = model(data).detach().cpu()
                condition = [{"y": c_id} for c_id in out.argmax(1)]

                attr = attribution(data, condition, composite, record_layer=layer_names, init_rel=1)

                masks = [dataset.get_mask(ind).to(device) for ind in samples_batch]
                inside = (attr.heatmap * torch.cat(masks, 0)).abs().sum((1, 2)) / (
                        attr.heatmap.abs().sum((1, 2)) + 1e-10)
                score.extend(list(1 - inside.detach().cpu()))

                # plt.imshow(1 / 255 * dataset.reverse_normalization(data[0]).detach().cpu().permute((1, 2, 0)))
                # plt.show()
                # hm = attr.heatmap[0].detach().cpu()
                # hm = hm / hm.abs().max()
                # plt.imshow(hm, cmap="bwr", vmin=-1, vmax=1)
                # plt.show()
                # print("debug")

            scores.append(np.mean(score))
            print(labels_set[k], scores[-1])
            if args.wandb_api_key:
                wandb.log({f"{split}_background_rel_{labels_set[k].lower()}": scores[-1]})

        for score in scores:
            wandb.log({f"{split}_background_rel": score})


if __name__ == "__main__":
    main()
