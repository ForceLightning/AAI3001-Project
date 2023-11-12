import os

import fastai
import numpy as np
from sklearn import base
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from fastai.callback.all import (CSVLogger, MixedPrecision, SaveModelCallback,
                                 ShowGraphCallback)
from fastai.vision.all import SegmentationDataLoaders, resnet34, unet_learner, DataLoader
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torchvision.transforms import v2
from tqdm.auto import tqdm

from utils.dataset import MoNuSegDataset
from utils.focaldiceloss import CombinedLoss

# Set to False if you don't want to use CUDA
ROOT_DIR = "./data/MoNuSeg 2018 Training Data/MoNuSeg 2018 Training Data"
USE_CUDA = torch.cuda.is_available() and True
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
MODELS_DIR = "./models/"
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 10
NUM_FOLDS = 5


def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(180),
        v2.RandomResizedCrop(
            256, interpolation=v2.InterpolationMode.NEAREST, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    valid_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomResizedCrop(
            256, interpolation=v2.InterpolationMode.NEAREST, antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[
                     0.229, 0.224, 0.225])
    ])
    train_data = MoNuSegDataset(ROOT_DIR, transform=train_transform,
                                target_transform=train_transform)
    valid_data = MoNuSegDataset(
        ROOT_DIR, transform=valid_transform, target_transform=valid_transform)

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True)

    for fold, (train_idx, valid_idx) in tqdm(
            enumerate(kf.split(train_data.images, train_data.annotations)),
            desc="Folds", total=NUM_FOLDS):
        train_ds = Subset(train_data, train_idx)
        valid_ds = Subset(valid_data, valid_idx)

        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, device=DEVICE)
        valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True, device=DEVICE)

        dataloader = SegmentationDataLoaders(train_dl, valid_dl)
        learn = unet_learner(
            dls=dataloader, arch=resnet34, metrics=fastai.metrics.JaccardCoeff(), cbs=[
                MixedPrecision(),
                CSVLogger(fname=f"{MODELS_DIR}/fold_{fold}.csv"),
                SaveModelCallback(fname=f"fold_{fold}"),
                ShowGraphCallback()
            ], normalize=False, pretrained=True, n_out=2, loss_func=CombinedLoss())
        learn.fine_tune(NUM_EPOCHS, base_lr=1e-3, freeze_epochs=3)
        learn.recorder.plot_loss()
        learn.recoder.plot_metrics()
        learn.save(f"fold_{fold}_final.pth")


if __name__ == "__main__":
    main()
