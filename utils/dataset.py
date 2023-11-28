import os
from typing import Callable

import numpy as np
import torch
import cv2
from lxml import etree
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2


class MoNuSegDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: v2.Compose = None,
        train: bool = True
    ) -> None:
        super().__init__()
        self.root = root
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ]) if transform is None else transform
        self.images = []
        self.annotations = []
        self.train = train
        self._init_dataset()

    def _init_dataset(self) -> None:
        if self.train:
            for file in os.listdir(os.path.join(self.root, 'Tissue Images')):
                if file.endswith('.tif'):
                    self.images.append(os.path.join(
                        self.root, 'Tissue Images', file))
                    xml_file = file.replace('.tif', '.xml')
                    self.annotations.append(os.path.join(
                        self.root, 'Annotations', xml_file))
        else:
            for file in os.listdir(self.root):
                if file.endswith('.tif'):
                    self.images.append(os.path.join(self.root, file))
                    xml_file = file.replace('.tif', '.xml')
                    self.annotations.append(os.path.join(self.root, xml_file))

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image_shape = image.height, image.width
        annotation = self._annotation_to_target(
            image_shape, self.annotations[idx])
        if self.transform:
            image, annotation = self.transform(image, annotation)
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
        contours = []
        for region in root.findall('Annotation/Regions/Region/Vertices'):
            coords = []
            for vertex in region:
                coords.append((float(vertex.attrib['X']),
                               float(vertex.attrib['Y'])))
            contour = np.array(coords, dtype=np.int32)
            contours.append(contour)
        mask = self._get_mask(contours, image_shape)

        mask = torch.from_numpy(mask).to(torch.int64)
        target = tv_tensors.Mask(mask)
        return target

    def _get_mask(
        self,
        contours: list[np.ndarray],
        image_shape: tuple
    ) -> np.ndarray[bool]:
        rendered_annotations = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(rendered_annotations, contours, -1, (0, 255, 0))
        for contour in contours:
            cv2.fillPoly(rendered_annotations, pts=np.array(
                [contour], np.int32), color=(255, 255, 255))
        rendered_annotations = rendered_annotations.astype(bool)
        return rendered_annotations


class MultiEpochsDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.sampler) if self.batch_sampler is None else len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
