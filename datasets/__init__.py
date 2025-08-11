import logging
from typing import Callable

from datasets.isic.isic import NORM_PARAMS_ISIC, ISICDataset, get_isic_dataset
from datasets.isic.isic_attacked import get_isic_attacked_dataset
from datasets.isic.isic_attacked_hm import get_isic_attacked_hm_dataset
from datasets.isic.isic_hm import get_isic_hm_dataset
from datasets.waterbirds.waterbirds import WaterbirdsDataset, get_waterbirds_dataset
from datasets.waterbirds.waterbirds_hm import get_waterbirds_hm_dataset
from datasets.imagenet.imagenet import NORM_PARAMS_IMAGENET, ImageNetDataset, get_imagenet_dataset
from datasets.imagenet.imagenet_attacked import get_imagenet_attacked_dataset
from datasets.imagenet.imagenet_attacked_hm import get_imagenet_attacked_hm_dataset
from utils.artificial_artifact import get_artifact_kwargs

logger = logging.getLogger(__name__)

DATASETS = {
    "isic": get_isic_dataset,
    "isic_hm": get_isic_hm_dataset,
    "isic_attacked": get_isic_attacked_dataset,
    "isic_attacked_hm": get_isic_attacked_hm_dataset,
    "waterbirds": get_waterbirds_dataset,
    "waterbirds_hm": get_waterbirds_hm_dataset,
    "imagenet": get_imagenet_dataset,
    "imagenet_attacked": get_imagenet_attacked_dataset,
    "imagenet_attacked_hm": get_imagenet_attacked_hm_dataset,

}

DATASET_CLASSES = {
    "isic": ISICDataset,
    "isic_attacked": ISICDataset,
    "waterbirds": WaterbirdsDataset,
    "imagenet": ImageNetDataset,
    "imagenet_attacked": ImageNetDataset,
}

DATASET_NORM_PARAMS = {
    # (means, vars)
    "imagenet": NORM_PARAMS_IMAGENET,
    "isic": NORM_PARAMS_ISIC,
    "isic_attacked": NORM_PARAMS_ISIC,
}


def get_dataset(dataset_name: str) -> Callable:
    """
    Get dataset by name.
    :param dataset_name: Name of the dataset.
    :return: Dataset.

    """
    if dataset_name in DATASETS:
        dataset = DATASETS[dataset_name]
        logger.info(f"Loading {dataset_name}")
        return dataset
    else:
        raise KeyError(f"DATASET {dataset_name} not defined.")
    
def get_dataset_kwargs(config):
    if "isic_attacked" in config['dataset_name']:
        dataset_specific_kwargs = {
            "p_artifact_na": config.get("p_artifact_na", 0.0),
        }
    elif "waterbirds" in config['dataset_name']:
        dataset_specific_kwargs = {
            "cub_places_path": config["cub_places_path"],
            "dilation_size": config.get("dilation_size", 3),
            "p_artifact_landbirds": config.get("p_artifact_landbirds", 0.0),
            "p_artifact_waterbirds": config.get("p_artifact_waterbirds", 0.0),
        }
    elif "imagenet" in config['dataset_name']:
        dataset_specific_kwargs = { 
            "label_map_path": config["label_map_path"],
            "classes": config.get("classes", None),
            "only_val": config.get("only_val", False),
    }
    else:
        dataset_specific_kwargs = {}
    return dataset_specific_kwargs

def load_dataset(config, normalize_data=True, hm=False):
    dataset_name = config['dataset_name']
    dataset_name = f"{dataset_name}_hm" if hm else dataset_name
    data_paths = config['data_paths']
    img_size = config.get("img_size", 224)
    binary_target = config.get('binary_target', None)
    attacked_classes = config.get("attacked_classes", [])
    p_artifact = config.get("p_artifact", None)
    artifact_type = config.get("artifact_type", None)
    artifact_ids_file=config.get('artifacts_file', None)
    p_backdoor = config.get("p_backdoor", 0)
    artifact_kwargs = get_artifact_kwargs(config)
    dataset_specific_kwargs = get_dataset_kwargs(config)
    
    if hm:
        if "isic_attacked" in dataset_name:
            dataset_specific_kwargs["gt_segmentation_path"] = config["gt_segmentation_path"]

        if "imagenet" in dataset_name:
            dataset_specific_kwargs["bbox_path"] = config["bbox_path"]

        if "attacked" in dataset_name:
            # specify whether to use GT or predicted masks
            source_maks = config.get("source_maks", "gt")
            assert source_maks in ["gt", "hm", "bin"], f"Unknown mask source: {source_maks}"
            dataset_specific_kwargs["source_masks"] = source_maks

        else:
            # specify artifact
            dataset_specific_kwargs["artifact"] = config["artifact"]

    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=normalize_data,
                                        image_size=img_size,
                                        binary_target=binary_target,
                                        attacked_classes=attacked_classes,
                                        p_artifact=p_artifact,
                                        p_backdoor=p_backdoor,
                                        artifact_type=artifact_type,
                                        artifact_ids_file=artifact_ids_file,
                                        **artifact_kwargs, **dataset_specific_kwargs)
    return dataset