import os

import fastai
import numpy as np
from sympy import per
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from fastai.data.all import DataLoaders
from fastai.callback.all import (CSVLogger, MixedPrecision, SaveModelCallback,
                                 ShowGraphCallback)
from fastai.vision.all import (BCEWithLogitsLossFlat, DataLoader,
                               SegmentationDataLoaders, resnet34, unet_learner, ResNet34_Weights)
from fastai.vision.learner import create_unet_model, Learner
from PIL import Image
from sklearn import base
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torchvision.transforms import v2
from tqdm.auto import tqdm

from utils.dataset import MoNuSegDataset, MultiEpochsDataLoader

# Set to False if you don't want to use CUDA
ROOT_DIR = "./data/MoNuSeg 2018 Training Data/MoNuSeg 2018 Training Data"
USE_CUDA = torch.cuda.is_available() and True
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
MODELS_DIR = "./models/"
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 10
NUM_FOLDS = 5
PERSISTENT_WORKERS = True


def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(180),
        v2.RandomResizedCrop(
            256, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    valid_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(256, interpolation=v2.InterpolationMode.BILINEAR,
                  antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[
                     0.229, 0.224, 0.225])
    ])
    train_data = MoNuSegDataset(ROOT_DIR, transform=train_transform)
    valid_data = MoNuSegDataset(ROOT_DIR, transform=valid_transform)

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True)

    for fold, (train_idx, valid_idx) in tqdm(
            enumerate(kf.split(train_data.images, train_data.annotations)),
            desc="Folds", total=NUM_FOLDS):
        train_ds = Subset(train_data, train_idx)
        valid_ds = Subset(valid_data, valid_idx)

        train_dl = MultiEpochsDataLoader(train_ds, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=NUM_WORKERS,
                                         pin_memory=True, sampler=None,
                                         persistent_workers=PERSISTENT_WORKERS)
        valid_dl = MultiEpochsDataLoader(valid_ds, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=NUM_WORKERS,
                                         pin_memory=True, sampler=None,
                                         persistent_workers=PERSISTENT_WORKERS)

        dataloader = DataLoaders(train_dl, valid_dl)

        model = create_unet_model(
            resnet34, n_out=1, img_size=(256, 256), pretrained=True, weights=ResNet34_Weights.DEFAULT)

        learn = Learner(dls=dataloader, model=model, metrics=fastai.metrics.Dice(), cbs=[
            MixedPrecision(),
            CSVLogger(fname=f"{MODELS_DIR}/fold_{fold}.csv"),
            SaveModelCallback(fname=f"fold_{fold}_best"),
            ShowGraphCallback()
        ], loss_func=BCEWithLogitsLossFlat())

        learn.fine_tune(NUM_EPOCHS, base_lr=3e-5,  # determined from lr_find()
                        freeze_epochs=5, pct_start=0.2)
        learn.save(f"fold_{fold}_final")

        train_dl._iterator._shutdown_workers()
        valid_dl._iterator._shutdown_workers()
        del train_dl
        del valid_dl


if __name__ == "__main__":
    main()
