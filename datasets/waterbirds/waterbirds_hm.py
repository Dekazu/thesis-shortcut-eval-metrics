import random
from pathlib import Path

import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from datasets.base_dataset import BaseDataset
from utils.localization import binarize_heatmaps
from datasets.waterbirds.waterbirds import WaterbirdsDataset, create_waterbirds_dataset
import cv2


def get_waterbirds_hm_dataset(data_paths, p_artifact_landbirds, p_artifact_waterbirds, cub_places_path, normalize_data=True, image_size=224, **kwargs):
    output_subfolder, transform, p_artifacts, cub_places_path, image_size, kwargs = create_waterbirds_dataset(data_paths, p_artifact_landbirds,
                                                                                        p_artifact_waterbirds, cub_places_path, normalize_data, image_size, 
                                                                                        **kwargs)

    return WaterbirdsHMDataset(output_subfolder, transform=transform, p_artifacts=p_artifacts, cub_places_path=cub_places_path,
                            image_size=image_size, augmentation=None, **kwargs)


class WaterbirdsHMDataset(WaterbirdsDataset):
    def __init__(self, path, cub_places_path, p_artifacts, transform, image_size, augmentation, **kwargs):

        super().__init__(path, cub_places_path, p_artifacts, transform, image_size, augmentation, **kwargs)

        self.dilation_size = kwargs.get('dilation_size', 3)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        img = Image.open(self.path / row['img_filename'])
        img = img.convert('RGB')

        # Get Mask
        seg_path = f"{self.cub_places_path[0]}/CUB_200_2011/segmentations"
        mask_path = os.path.join(seg_path, row['img_filename'].replace('.jpg', '.png'))
        mask_img = Image.open(mask_path)
        mask_img = mask_img.convert('L')
        
        # Image to tensor
        tensor_transform = T.ToTensor()
        mask_img_tensor = tensor_transform(mask_img)
        
        mask_np = mask_img_tensor.numpy()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 3x3-Kernel

        # Führe die Dilation durch
        dilated_mask_np = cv2.dilate(mask_np, kernel, iterations=5)

        # Konvertiere die erweiterte Maske zurück zu einem PyTorch-Tensor
        mask_img_tensor = torch.from_numpy(dilated_mask_np).float()

        # Binarize mask via finding a threshold
        mask_img_tensor = binarize_heatmaps(mask_img_tensor)

        mask_img_tensor = T.Resize((self.image_size, self.image_size), interpolation=T.functional.InterpolationMode.BICUBIC)(mask_img_tensor)
        # mask_img_tensor = mask_img_tensor.squeeze(0) # Remove channel dimension (1, 224, 224) -> (224, 224)

        mask = torch.ones_like(mask_img_tensor) - mask_img_tensor # Background is 1, bird is 0
        #mask = mask_img_tensor

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(row['y']), mask.squeeze(0)
    
    def get_segmentation_mask(self, i):
        _, _, mask = self.__getitem__(i)

        mask = torch.ones_like(mask) - mask # Background is 0, bird is 1
        return mask

