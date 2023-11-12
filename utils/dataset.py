import os
from typing import Callable

import numpy as np
import skimage
import torch
from lxml import etree
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2


class MoNuSegDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: v2.Compose = None,
        target_transform: Callable = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ]) if transform is None else transform
        self.target_transform = target_transform
        self.images = []
        self.annotations = []
        self._init_dataset()

    def _init_dataset(self) -> None:
        for file in os.listdir(os.path.join(self.root, 'Tissue Images')):
            if file.endswith('.tif'):
                self.images.append(os.path.join(
                    self.root, 'Tissue Images', file))
                xml_file = file.replace('.tif', '.xml')
                self.annotations.append(os.path.join(
                    self.root, 'Annotations', xml_file))

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image_shape = image.height, image.width
        annotation = self._annotation_to_target(
            image_shape, self.annotations[idx])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            annotation = self.target_transform(annotation)
        return image, annotation

    def __len__(self):
        return len(self.images)

    def _annotation_to_target(
        self,
        image_shape: tuple,  # (rows, cols), origin is top left corner.
        xml_path
    ) -> dict:
        with open(xml_path, 'r', encoding='utf-8') as f:
            tree = etree.parse(f)
        root = tree.getroot()
        regions = root.find('Annotation').find('Regions').findall('Region')
        mask = np.zeros(image_shape, dtype=bool)
        for region in regions:
            vertices = region.find('Vertices').findall('Vertex')
            x = []
            y = []
            for vertex in vertices:
                x.append(float(vertex.attrib['X']))
                y.append(float(vertex.attrib['Y']))
            poly = np.array([x, y]).T
            mask = self._get_mask(mask, poly, image_shape)

        mask = torch.from_numpy(mask).to(torch.int64)
        target = tv_tensors.Mask(mask)
        return target

    def _get_mask(
        self,
        mask: np.ndarray,
        polys: np.ndarray,
        image_shape: tuple  # (rows, cols), origin is top left corner.
    ) -> np.ndarray[bool]:
        new_mask = skimage.draw.polygon2mask(image_shape, polys)
        mask = np.logical_or(mask, new_mask)
        return mask
