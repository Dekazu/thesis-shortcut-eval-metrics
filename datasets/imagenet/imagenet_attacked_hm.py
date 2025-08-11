import random
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
import xml.etree.ElementTree as ET
import xmltodict
import cv2

from datasets.imagenet.imagenet import imagenet_augmentation, NORM_PARAMS_IMAGENET
from datasets.imagenet.imagenet_attacked import ImageNetAttackedDataset


def get_imagenet_attacked_hm_dataset(data_paths, 
                                     normalize_data=True,
                                     image_size=224,
                                     label_map_path=None,
                                     classes=None,
                                     attacked_classes=[],
                                     p_artifact=0.5,
                                     artifact_type="ch_text",
                                     bbox_path=None,
                                     **kwargs):
    print("in get dataset_hm", label_map_path, kwargs)
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]
    if normalize_data:
        fns_transform.append(T.Normalize(*NORM_PARAMS_IMAGENET))
    transform = T.Compose(fns_transform)
    return ImageNetAttackedHmDataset(
        data_paths=data_paths,
        transform=transform,
        augmentation=imagenet_augmentation,
        attacked_classes=attacked_classes,
        p_artifact=p_artifact,
        artifact_type=artifact_type,
        label_map_path=label_map_path,
        classes=classes,
        image_size=image_size,
        bbox_path=bbox_path,
        **kwargs
    )


class ImageNetAttackedHmDataset(ImageNetAttackedDataset):
    def __init__(self,
                 data_paths,
                 transform,
                 augmentation,
                 attacked_classes=[],
                 p_artifact=0.5,
                 artifact_type="ch_text",
                 label_map_path=None,
                 classes=None,
                 image_size=224,
                 only_val=False,
                 bbox_path=None,
                 **artifact_kwargs):
        super().__init__(data_paths=data_paths, transform=transform, augmentation=augmentation, attacked_classes=attacked_classes, p_artifact=p_artifact,
                         artifact_type=artifact_type, label_map_path = label_map_path, classes=classes, image_size=image_size, only_val=only_val, p_backdoor=artifact_kwargs.get('p_backdoor', 0))
        self.bbox_path = f'{bbox_path}/imagenet/'
        assert self.bbox_path is not None, "bbox_path must be provided"
        assert os.path.isdir(self.bbox_path), "The bbox_path does not exist or is not a directory."
        # (self.only_val, self.idxs_train, self.idxs_val, and self.idxs_test are assumed to be set in the parent.)

    def load_bbox_mask(self, annotation_path, orig_wnid):
        """
        Loads a bounding-box mask from an XML annotation.
        Parses the XML (using xmltodict and ElementTree) and creates a binary mask with ones in the bounding-box regions.
        """
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        xmlstr = ET.tostring(root, encoding="utf-8", method="xml")
        annotation = dict(xmltodict.parse(xmlstr))['annotation']
        width = int(annotation["size"]["width"])
        height = int(annotation["size"]["height"])
        mask = np.zeros((height, width), dtype=np.uint8)
        objects = annotation.get("object", [])
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            bndbox = obj['bndbox']
            xmin = int(bndbox['xmin'])
            ymin = int(bndbox['ymin'])
            xmax = int(bndbox['xmax'])
            ymax = int(bndbox['ymax'])
            mask[ymin:ymax, xmin:xmax] = 1
        # Resize mask to (image_size, image_size)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = torch.tensor(mask, dtype=torch.uint8)
        return mask

    def get_segmentation_mask(self, idx):
        """
        Returns a binary segmentation mask computed from the bounding-box XML annotation
        for the sample at index `idx`.
        
        If self.only_val is True, the annotation file is assumed to be under:
            bbox_path/val/{stem}.xml
        Otherwise, if the sample index is in self.idxs_train or self.idxs_val, the annotation is assumed to be in:
            bbox_path/{wnid}/{stem}.xml
        And if the sample index is in self.idxs_test, it is assumed to be in:
            bbox_path/val/{stem}.xml
        """
        # Our self.samples is assumed to be stored as (image_path, wnid)
        path, old_wnid, wnid = self.samples[idx]
        stem = os.path.splitext(os.path.basename(path))[0]
        if self.only_val:
            ann_path = os.path.join(self.bbox_path, "val", f"{stem}.xml")
        else:
            if idx in self.idxs_val:
                ann_path = os.path.join(self.bbox_path, f"train/{old_wnid}/" f"{stem}.xml")
            elif idx in self.idxs_test:
                ann_path = os.path.join(self.bbox_path, "val", f"{stem}.xml")
            else:
                ann_path = os.path.join(self.bbox_path, f"train/{old_wnid}", f"{stem}.xml")
        if os.path.exists(ann_path):
            return self.load_bbox_mask(ann_path, wnid)
        else:
            return np.nan # The mask is not available. We have to skip it

    def __getitem__(self, idx):
        # Basic __getitem__ returns image and target label.
        path, _, wnid = self.samples[idx]
        x = Image.open(path).convert('RGB')
        if hasattr(self, "transform_resize"):
            x = self.transform_resize(x)
        # Optionally apply artifact logic.
        if self.artifact_labels[idx]:
            x, artifact_mask = self.add_artifact(x, idx)
            mask = artifact_mask
        else:
            mask = torch.zeros((self.image_size, self.image_size), dtype=torch.uint8)
        x = self.transform(x) if self.transform else x
        x = self.augmentation(x) if self.do_augmentation and self.augmentation else x
        y = self.label_map[wnid]["label"]
        return x, y, mask.type(torch.uint8)
