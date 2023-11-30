"""
Runs the whitebox evaluation on the test set.
"""
import csv
import gc
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms.functional as F
from fastai.callback.all import MixedPrecision
from fastai.data.all import DataLoaders
from fastai.learner import CastToTensor
from fastai.vision.all import DataLoader, resnet50
from fastai.vision.learner import Learner, create_unet_model
from PIL import Image
from torch import nn
from torchvision.transforms import v2
from torchvision.utils import draw_segmentation_masks
from tqdm.auto import tqdm

from utils.dataset import MoNuSegDataset
from utils.lossmetrics import BinaryDice, CombinedBCEDiceLoss, PixelAccuracy

VALID_DIR = os.path.join(
    os.path.dirname(__file__), "data", "MoNuSeg 2018 Training Data", "MoNuSeg 2018 Training Data")
TEST_DIR = os.path.join(os.path.dirname(__file__), "data", "MoNuSegTestData")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
USE_CUDA = torch.cuda.is_available() and True
BATCH_SIZE = 1  # untested with other batch sizes
NUM_WORKERS = 0
NUM_FOLDS = 5
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
EPSILONS = [0.0, 1e-2, 1.5e-2, 3e-2, 6e-2, 1e-1, 3e-1]


def get_fold_adversarial_img(
    output_dir: str | os.PathLike,
    fold_idx: int,
    img_idx: int,
    eps_idx: int
) -> str | os.PathLike:
    """Gets the path to the adversarial image for a given fold, image, and epsilon.

    Args:
        output_dir (str): The output directory.
        fold_idx (int): The fold index.
        img_idx (int): The image index.
        eps_idx (int): The epsilon index.

    Returns:
        str: The path to the adversarial image.
    """
    return os.path.join(output_dir, f"fold_{fold_idx}", f"img_{img_idx}_eps_{eps_idx}.png")


def show(
    imgs: List[torch.Tensor] | torch.Tensor,
    labels: List[str],
    title: str = "",
    show_fig: bool = True
) -> None | Tuple[plt.Figure, plt.Axes]:
    """Shows a list of images.

    Args:
        imgs (List[torch.Tensor] | torch.Tensor): Image or list of images.
        labels (List[str]): List of labels.
        title (str, optional): Title string. Defaults to "".
        show_fig (bool, optional): If true, shows the figure.
            Disable when many calls are made to this function. Defaults to True.

    Returns:
        None | Tuple[plt.Figure, plt.Axes]: If show_fig is False, returns the figure and axes.
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    fig.set_size_inches(10 * len(imgs), 10)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[0, i].set_title(labels[i])
    fig.suptitle(title)
    if show_fig:
        plt.show()
        return
    return fig, axs


def main():
    valid_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(256, interpolation=v2.InterpolationMode.BILINEAR,
                  antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[
                     0.229, 0.224, 0.225])
    ])

    test_data = MoNuSegDataset(
        TEST_DIR, transform=valid_transform, train=False)
    test_dl_fastai = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                pin_memory=True, sampler=None, shuffle=False, persistent_workers=True,
                                device=DEVICE)

    with open(os.path.join(OUTPUT_DIR, "whitebox_test.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "img", "eps", "acc", "dice"])
        for fold_num in tqdm(range(NUM_FOLDS), desc="Folds"):
            model = create_unet_model(
                resnet50, n_out=1, img_size=(256, 256), pretrained=False
            )
            if USE_CUDA:
                model.cuda()

            del model.layers[-1]

            learn = Learner(dls=DataLoaders(test_dl_fastai), model=model, metrics=BinaryDice(),
                            cbs=[MixedPrecision(), CastToTensor()],
                            loss_func=CombinedBCEDiceLoss())
            learn.load(f"fold_{fold_num}_best")
            model.eval()

            for img_idx in range(len(test_data)):
                images_and_masks = []
                for eps_idx, eps in enumerate(EPSILONS):
                    img_path = get_fold_adversarial_img(
                        OUTPUT_DIR, fold_num, img_idx, eps_idx)
                    img = Image.open(img_path)
                    mask = test_data[img_idx][1].unsqueeze(0).to(DEVICE)
                    img_tensor = valid_transform(img).unsqueeze(0).to(DEVICE)
                    proba = nn.Sigmoid()(model(img_tensor))
                    pred = proba > 0.5
                    inter = (pred * mask).sum().item()
                    union = (pred + mask).sum().item()
                    dice = 2 * inter / union if union > 0 else 0.0
                    pred = pred.to(torch.bool).to(DEVICE)
                    acc = PixelAccuracy(pred.to(torch.int32), mask)
                    image_and_mask = draw_segmentation_masks(
                        v2.ToImage()(img), pred.squeeze(0, 1), alpha=1.0, colors=["red"]
                    )
                    images_and_masks.append(image_and_mask)
                    print(
                        f"Fold {fold_num}, img {img_idx}, eps {eps:.2e}, acc {acc * 100:4.2f}%, dice {dice:.4f}")
                    writer.writerow([fold_num, img_idx, eps, acc, dice])

                fig, _ = show(images_and_masks, [r"$\varepsilon = %.2e$" % (
                    eps) for eps in EPSILONS], title=f"Fold {fold_num}, img {img_idx}", show_fig=False)
                fig.savefig(os.path.join(
                    OUTPUT_DIR, f"fold_{fold_num}_img_{img_idx}.png"))
                plt.close()

    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()

    sns.set("paper", style="whitegrid")

    df = pd.read_csv(os.path.join(OUTPUT_DIR, "whitebox_test.csv"), header=0)
    ax = sns.lineplot(data=df, x="eps", y="acc", hue="fold", legend="full")
    ax.set_xlabel(r"$L^\infty$ perturbation $\varepsilon$")
    ax.set_ylabel("Accuracy")
    ax.set_title(r"Accuracy vs. $L^\infty$ perturbation")

    plt.savefig(os.path.join(OUTPUT_DIR, "whitebox_test_acc.png"))
    plt.show()

    ax = sns.lineplot(data=df, x="eps", y="dice", hue="fold", legend="full")
    ax.set_xlabel(r"$L^\infty$ perturbation $\varepsilon$")
    ax.set_ylabel("Dice score")
    ax.set_title(r"Dice score vs. $L^\infty$ perturbation")

    plt.savefig(os.path.join(OUTPUT_DIR, "whitebox_test_dice.png"))
    plt.show()


if __name__ == "__main__":
    main()
