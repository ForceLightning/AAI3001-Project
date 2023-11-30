"""
This file runs the validation of the model on the validation sets (from K-Fold) and the test set.
"""
import os

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from fastai.vision.all import resnet50
from fastai.vision.learner import create_unet_model
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from torchvision.transforms import v2

from utils.dataset import MoNuSegDataset
from utils.validation import k_fold_inference, k_fold_roc_curve

VALID_DIR = os.path.join(os.path.dirname(os.path.realpath(
    __file__)), "data", "MoNuSeg 2018 Training Data", "MoNuSeg 2018 Training Data")
TEST_DIR = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "data", "MoNuSegTestData")
MODELS_DIR = "./models/"
BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_FOLDS = 5


def main():
    sns.set_theme("paper", "whitegrid")
    valid_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(256, interpolation=v2.InterpolationMode.BILINEAR,
                  antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[
                     0.229, 0.224, 0.225])
    ])

    valid_data = MoNuSegDataset(VALID_DIR, transform=valid_transform)
    test_data = MoNuSegDataset(
        TEST_DIR, transform=valid_transform, train=False)

    model = create_unet_model(
        resnet50, n_out=1, img_size=(256, 256), pretrained=False
    )

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    _, _ = k_fold_inference(
        model=model,
        test_ds=valid_data,
        model_dir=MODELS_DIR,
        weights_name="fold_{idx}_best",
        k=NUM_FOLDS,
        target_names=["background", "nuclei"],
        splitter=kf,
        dl_batch_size=BATCH_SIZE,
        dl_num_workers=NUM_WORKERS,
        dl_persistent_workers=True,
        output_save_format="fold_{idx}_valid_output.npz"
    )

    model_outputs = []
    for i in range(NUM_FOLDS):
        output = np.load(os.path.join(
            f"{MODELS_DIR}", f"fold_{i}_valid_output.npz"))
        model_outputs.append(output)
        report = classification_report(
            y_true=output["y"].reshape(-1), y_pred=output["preds"].reshape(-1),
            target_names=["background", "nuclei"], output_dict=True)
        report = pd.DataFrame(report)
        print(report)

    k_fold_roc_curve(
        model_outputs=model_outputs,
        model_name="ResNet50 (Valididation Set)",
        num_classes=1,
        average="macro"
    )

    _, _ = k_fold_inference(
        model=model,
        test_ds=test_data,
        model_dir=MODELS_DIR,
        weights_name="fold_{idx}_best",
        k=1,
        target_names=["background", "nuclei"],
        dl_batch_size=BATCH_SIZE,
        dl_num_workers=NUM_WORKERS,
        dl_persistent_workers=True,
        output_save_format="fold_{idx}_test_output.npz"
    )

    model_outputs = []
    for i in range(NUM_FOLDS):
        output = np.load(os.path.join(
            f"{MODELS_DIR}", f"fold_{i}_test_output.npz"))
        model_outputs.append(output)
        report = classification_report(
            y_true=output["y"].reshape(-1), y_pred=output["preds"].reshape(-1),
            target_names=["background", "nuclei"], output_dict=True)
        report = pd.DataFrame(report)
        print(report)

    k_fold_roc_curve(
        model_outputs=model_outputs,
        model_name="ResNet50 (Test Set)",
        num_classes=1,
        average="macro"
    )


if __name__ == "__main__":
    main()
