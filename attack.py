import os
import torch
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import fastai
from utils.dataset import MoNuSegDataset, MultiEpochsDataLoader
from fastai.callback.all import (
    CSVLogger,
    MixedPrecision,
    SaveModelCallback,
    ShowGraphCallback,
)
from fastai.vision.all import (
    BCEWithLogitsLossFlat,
    DataLoader,
    ResNet34_Weights,
    ResNet50_Weights,
    SegmentationDataLoaders,
    resnet34,
    resnet50,
    unet_learner,
    DynamicUnet,
)
from fastai.vision.learner import Learner, create_unet_model
from torchvision.transforms import v2

# for blackbox attack
from torchvision import models
from foolbox import PyTorchModel, accuracy, samples, attacks
from foolbox.attacks.boundary_attack import BoundaryAttack
from foolbox.attacks import FGSM
from foolbox.criteria import Misclassification, TargetedMisclassification
import foolbox as fb


# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set variables
ROOT_DIR = "./data/MoNuSegTestData"
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 10
NUM_FOLDS = 5
PERSISTENT_WORKERS = True

from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


if __name__ == "__main__":
    # Set transforms - preprocessing
    valid_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(256, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # load dataset
    test_data = MoNuSegDataset(ROOT_DIR, transform=valid_transform)

    # load dataloader
    test_dl = MultiEpochsDataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        sampler=None,
        persistent_workers=PERSISTENT_WORKERS,
    )

    # model_path = "/models/fold_0_best"

    # load model
    # model = create_unet_model(
    #     resnet34,
    #     n_out=1,
    #     img_size=(256, 256),
    #     pretrained=True,
    #     weights=ResNet34_Weights.DEFAULT,
    # )
    # learn = Learner(
    #     dls=test_dl,
    #     model=model,
    #     metrics=fastai.metrics.Dice(),
    #     cbs=[
    #         MixedPrecision(),
    #     ],
    #     loss_func=BCEWithLogitsLossFlat(),
    # )
    # learn.load("fold_0_best")
    # Get the first image and its corresponding segmentation mask
    image, _ = test_data.__getitem__(0)
    image = image.unsqueeze(0).to(DEVICE)

    # load model
    model = create_unet_model(
        resnet50,
        n_out=1,
        img_size=(256, 256),
        pretrained=True,
        weights=ResNet50_Weights.DEFAULT,
    )
    learn = Learner(
        dls=test_dl,
        model=model,
        metrics=fastai.metrics.Dice(),
        cbs=[
            MixedPrecision(),
        ],
        loss_func=BCEWithLogitsLossFlat(),
    )
    learn.load("fold_0_best")

    model.eval().to(DEVICE)

    del model.layers[-1]
    # model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(DEVICE)
    # model.eval().to(DEVICE)
    # run normal model
    outputs = model(image)
    # labels, preds = outputs.data
    labels, preds = torch.max(outputs.data, 1)
    labels = labels.long()
    # print()

    # load foolbox model
    foolbox_model = PyTorchModel(
        model, bounds=(-100, 100), device=DEVICE, preprocessing=None
    )

    # images, labels =  samples(dataset=image, data_format="channels_first", bounds=(0, 255))
    # apply the attack
    attack = FGSM()
    # images, labels = fb.utils.samples(fmodel=foolbox_model, dataset=test_data, data_format="channels_first")
    # attack = BoundaryAttack(step_adaptation=0.5, steps=250)
    epsilons = [np.arange(0.1, 1.1, 0.2)]
    adversarial_images = []

    # attack =fb.attacks.deepfool.LinfDeepFoolAttack()

    criterion = Misclassification(labels=labels.unsqueeze(0))
    # criterion = BCEWithLogitsLossFlat()
    adversarial_images = attack(
        model=foolbox_model, inputs=image, criterion=criterion, epsilons=epsilons
    )

    model_predictions = model(adversarial_images[0][0])
    attack_labels, attack_preds = torch.max(model_predictions.data, 1)

    print(
        "real label: {}, label prediction; {}".format(attack_labels[0], attack_preds[0])
    )

    # for epsilon in epsilons:
    #     advs, _, success = attack.run(model=foolbox_model, inputs=image, criterion=BCEWithLogitsLossFlat(), epsilon=epsilon)
    #     adversarial_images.append(advs)
    #     # Visualize the input image and segmentation result

    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    # plt.imshow(image.cpu().numpy().transpose(1, 2, 0))
    image = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    plt.imshow(image)
    plt.title("Input Image")

    plt.subplot(1, 3, 2)
    adv_img = adversarial_images[0][0].squeeze(0).cpu().numpy().transpose(1, 2, 0)
    plt.imshow(adv_img, cmap="gray")
    plt.title("Ground Truth")

    plt.subplot(1, 3, 3)
    diff = image - adv_img
    plt.imshow(diff)
    plt.title("Segmentation Result")

    # plt.show()

    pred = (attack_preds > 0.5).to(torch.bool)
    pred = torch.tensor(pred)
    temp_image = Image.open(
        ".\data\MoNuSegTestData\Tissue Images\TCGA-18-5592-01Z-00-DX1.tif"
    ).convert("RGB")
    # temp_image = torch.tensor(temp_image,dtype=torch.uint8)
    valid_transform2 = v2.Compose(
        [
            v2.ToImage(),
        ]
    )

    temp_image = v2.ToImage()(temp_image)

    print(temp_image.shape)
    print(pred.shape)
    show(draw_segmentation_masks(temp_image, pred, alpha=0.5, colors=["green"]))

    # for epsilon, advs in zip(epsilons, adversarial_images):
    #     # Evaluate the adversarial examples using your model
    #     success_rate = np.mean(np.argmax(foolbox_model(advs), axis=-1) != true_labels)
    #     print(f"Attack success rate with epsilon {epsilon}: {success_rate * 100:.2f}%")

    # Get the first image and its corresponding segmentation mask
    # for i in range(test_data.__len__()):
    #     image, annotation = test_data.__getitem__(i)
    #     image_tensor = valid_transform(image).unsqueeze(0).to(DEVICE)
    #     # Preprocess and move to device
    #     preds, _ = learn.get_preds(dl=test_dl)

    #     # Visualize the input image and segmentation result
    #     plt.figure(figsize=(15, 10))
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(image.cpu().numpy().transpose(1, 2, 0))
    #     plt.title("Input Image")

    #     plt.subplot(1, 3, 2)
    #     plt.imshow(preds[i][0].cpu().numpy(), cmap="gray")
    #     plt.title("Segmentation Result")

    #     plt.subplot(1, 3, 3)
    #     plt.imshow(annotation.cpu().numpy(), cmap="gray")
    #     plt.title("Ground Truth")

    #     plt.show()
