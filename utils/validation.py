"""
This module contains the utility functions for validating the model.
"""
import os
import gc
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import Dataset, Subset
from sklearn.metrics import (auc, average_precision_score, classification_report,
                             precision_recall_curve, roc_auc_score)
from sklearn.metrics import roc_curve as sk_roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from fastai.vision.all import Learner
from fastai.data.core import DataLoaders, TensorBase, DataLoader
from fastai.callback.all import MixedPrecision

from utils.lossmetrics import BinaryDice, CombinedBCEDiceLoss


def k_fold_inference(
    model: nn.Module,
    test_ds: Dataset | Subset,
    model_dir: str = "./models/",
    weights_name: str = "fold_{idx}_best",
    k: int = 10,
    target_names: Optional[List[str]] = None,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"),
    splitter: Optional[Union[KFold, StratifiedKFold]] = None,
    dl_batch_size: int = 4,
    dl_shuffle: bool = False,
    dl_pin_memory: bool = True,
    dl_num_workers: int = 4,
    dl_persistent_workers: bool = True,
    output_save_format: str = "fold_{idx}_output.npz"
) -> Tuple[List[Union[dict, str]], List[dict]]:
    """Performs K-Fold inference on the given model.

    Args:
        model (nn.Module): Model to perform inference on.
        test_ds (Dataset | Subset): Dataset to perform inference on.
        model_dir (str, optional): Saved model checkpoints directory. Defaults to "./models/".
        weights_name (str, optional): Format of checkpoint files. Defaults to "fold_{idx}_best".
        k (int, optional): Number of folds. Defaults to 10.
        target_names (Optional[List[str]], optional): Target classification names. Defaults to None.
        device (torch.device, optional): Device to perform inferrence on. Defaults to torch.device( "cuda" if torch.cuda.is_available() else "cpu").
        splitter (Optional[Union[KFold, StratifiedKFold]], optional): Splits the dataset into k parts. Defaults to None.
        dl_batch_size (int, optional): Batch size for DataLoader. Defaults to 4.
        dl_shuffle (bool, optional): Determines whether the DataLoader will shuffle. Defaults to False.
        dl_pin_memory (bool, optional): Determines whether the DataLoader will pin data to device memory. Defaults to True.
        dl_num_workers (int, optional): Number of DataLoader workers. Defaults to 4.
        dl_persistent_workers (bool, optional): Determines whether DataLoader workers will persist. Defaults to True.
        output_save_format (str, optional): Model output save format. Defaults to "fold_{idx}_output.npz".

    Returns:
        Tuple[List[Union[dict, str]], List[dict]]: Model outputs and classification reports.
    """
    if target_names is None:
        target_names = ["Background", "Nucleus"]

    model_outputs = []
    classification_reports = []
    iterator = range(k) if splitter is None else enumerate(
        splitter.split(np.zeros(len(test_ds.annotations)), test_ds.annotations))
    for fold in tqdm(iterator, position=0, desc="Folds", total=k):
        if splitter is not None:
            idx = fold[0]
            valid_idx = fold[1][1]
            fold_ds = Subset(test_ds, valid_idx)
            test_dl = DataLoader(
                fold_ds, batch_size=dl_batch_size, shuffle=dl_shuffle, num_workers=dl_num_workers,
                pin_memory=dl_pin_memory, sampler=None, persistent_workers=dl_persistent_workers,
                device=device
            )
        else:
            idx = fold
            test_dl = DataLoader(
                test_ds, batch_size=dl_batch_size, shuffle=dl_shuffle, num_workers=dl_num_workers,
                pin_memory=dl_pin_memory, sampler=None, persistent_workers=dl_persistent_workers,
                device=device
            )
        test_dls = DataLoaders(test_dl)
        learner = Learner(dls=test_dls, model=model, metrics=BinaryDice(), cbs=[
            MixedPrecision()
        ], loss_func=CombinedBCEDiceLoss(alpha=0.5, reduction="mean"))
        learner.load(weights_name.format(idx=idx))

        proba, target = learner.get_preds(dl=test_dls)
        # flatten both proba and target
        proba, target = map(_contiguous, (proba, target))
        proba = proba.view(-1, proba.shape[-1])
        target = target.view(-1, target.shape[-1])
        preds = proba > 0.5
        proba = proba.cpu().numpy()
        target = target.cpu().numpy()
        preds = preds.cpu().numpy()

        try:
            classification_reports.append(classification_report(
                target,
                preds,
                target_names=target_names,
                output_dict=True,
                zero_division=0
            ))
        except ValueError:
            classification_reports.append(None)

        model_outputs.append({
            "proba": proba,
            "preds": preds,
            "y": target
        })

        if output_save_format is not None:
            np.savez_compressed(
                os.path.join(model_dir, output_save_format.format(idx=idx)),
                proba=proba,
                preds=preds,
                y=target
            )
            model_outputs[-1] = os.path.join(
                model_dir, output_save_format.format(idx=idx)
            )

        # Prevent CUDA OOM
        del learner, proba, target, preds
        del test_dl, test_dls
        gc.collect()

        with torch.no_grad():
            torch.cuda.empty_cache()

    return model_outputs, classification_reports


def k_fold_roc_curve(
    model_outputs: Dict[str, Union[torch.Tensor, np.ndarray, np.lib.npyio.NpzFile]],
    model_name: str = "ResNet-50",
    num_classes: int = 2,
    average: str = "macro",
    legend_key: str = "Fold",
    show_mean_and_std: bool = True
) -> None:
    """Generates ROC and PRC curves for the given model outputs.

    Args:
        model_outputs (Dict[str, Union[torch.Tensor, np.ndarray, np.lib.npyio.NpzFile]]): Model outputs.
        model_name (str, optional): Name of model. Defaults to "ResNet-50".
        num_classes (int, optional): Number of classes. Defaults to 2.
        average (str, optional): Unused, averaging type. Defaults to "macro".
        legend_key (str, optional): Iterated type of model to put on the legend. Defaults to "Fold".
        show_mean_and_std (bool, optional): Whether to show mean and std dev on chart. Defaults to True.
    """
    fig, ax = plt.subplots(2, 1, figsize=(8.27, 11.69), dpi=100)
    tprs, aurocs, tpr_threshes = [], [], []
    fpr_mean = np.linspace(0, 1, 1000)
    precisions, auprcs, recall_threshes = [], [], []
    recall_mean = np.linspace(0, 1, 1000)
    for fold_idx, fold in enumerate(tqdm(model_outputs)):
        intermediate_tprs, intermediate_tpr_threshes = [], []
        intermediate_precs, intermediate_rec_threshes = [], []
        for i in range(num_classes):
            y = fold["y"].reshape(-1)
            proba = fold["proba"].reshape(-1)
            # ROC Curve
            fpr, tpr, tpr_thresh = sk_roc_curve(
                y,
                proba
            )

            intermediate_tpr_threshes.append(
                tpr_thresh[np.abs(tpr - 0.85).argmin()]
            )

            tpr_interp = np.interp(fpr_mean, fpr, tpr)
            tpr_interp[0] = 0.0
            intermediate_tprs.append(tpr_interp)

            # PRC
            precision, recall, recall_thresh = precision_recall_curve(
                y,
                proba
            )
            prec_interp = np.interp(recall_mean, recall[::-1], precision[::-1])
            intermediate_precs.append(prec_interp)
            intermediate_rec_threshes.append(
                recall_thresh[np.abs(recall - 0.85).argmin()]
            )

        auroc = roc_auc_score(y, proba, average=average)
        auprc = average_precision_score(
            y, proba, average=average)

        tprs.append(np.mean(intermediate_tprs, axis=0))
        aurocs.append(auroc)
        tpr_threshes.append(np.mean(intermediate_tpr_threshes))
        precisions.append(np.mean(intermediate_precs, axis=0))
        auprcs.append(auprc)
        recall_threshes.append(np.mean(intermediate_rec_threshes))

        ax[0].plot(fpr_mean, tprs[-1],
                   label=f"ROC {legend_key} {fold_idx + 1} (AUC = {aurocs[-1]:.2f})", alpha=.3)

        ax[1].plot(recall_mean, precisions[-1],
                   label=f"PRC {legend_key} {fold_idx + 1} (AUPRC = {auprcs[-1]:.2f})", alpha=.3)

    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title(f"Receiver Operating Characteristics Curve ({model_name})")
    ax[0].set_ylim(-0.1, 1.1)

    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title(f"Precision Recall Curve ({model_name})")
    ax[1].set_ylim(-0.1, 1.1)

    if show_mean_and_std:
        # ROC
        tpr_mean = np.mean(tprs, axis=0)
        tpr_mean[-1] = 1.0
        auroc_mean = auc(fpr_mean, tpr_mean)
        auroc_std = np.std(aurocs)
        ax[0].plot(
            fpr_mean, tpr_mean,
            label=r"Mean ROC (AUC = $%.2f \pm %.2f$)" % (
                auroc_mean, auroc_std),
            lw=2, alpha=.8
        )
        tpr_std = np.std(tprs, axis=0)
        ax[0].fill_between(
            fpr_mean, np.maximum(tpr_mean - tpr_std, 0),
            np.minimum(tpr_mean + tpr_std, 1), alpha=.2,
            label=r"$\pm$ 1 std. dev.", color="grey"
        )

        # PRC
        prec_mean = np.mean(precisions, axis=0)
        auprc_mean = auc(recall_mean, prec_mean)
        auprc_std = np.std(auprcs)
        ax[1].plot(
            recall_mean, prec_mean,
            label=r"Mean PRC (AUC = $%.2f \pm %.2f$)" % (
                auprc_mean, auprc_std),
            lw=2, alpha=.8
        )
        prec_std = np.std(precisions, axis=0)
        ax[1].fill_between(
            recall_mean, np.maximum(prec_mean - prec_std, 0),
            np.minimum(prec_mean + prec_std, 1), alpha=.2,
            label=r"$\pm$ 1 std. dev.", color="grey"
        )

    fig.suptitle(f"ROC and PRC Curves for {model_name}, average={average}")
    ax[0].legend()
    ax[1].legend()
    plt.show()


def _contiguous(x: torch.Tensor, axis: int = 1) -> TensorBase:
    """Returns a contiguous tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Contiguous tensor.
    """
    return TensorBase(x.transpose(axis, -1).contiguous()) if isinstance(x, torch.Tensor) else x
