import os
import torch
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import fastai
from utils.dataset import MoNuSegDataset, MultiEpochsDataLoader  
from fastai.callback.all import (CSVLogger, MixedPrecision, SaveModelCallback,
                                 ShowGraphCallback)
from fastai.vision.all import (BCEWithLogitsLossFlat, DataLoader,
                               ResNet50_Weights, SegmentationDataLoaders,
                               resnet50, unet_learner, DynamicUnet)
from fastai.vision.learner import Learner, create_unet_model
from torchvision.transforms import v2, ToPILImage
import torch
import torch.nn.functional as F
from utils.lossmetrics import PixelAccuracy


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = "./data/MoNuSegTestData"
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 10
NUM_FOLDS = 5
PERSISTENT_WORKERS = True


def fgsm_attack(model, images, labels, epsilon):

    images.requires_grad = True

    outputs = model(images).to(DEVICE)
    labels = labels.to(DEVICE)

    criterion = BCEWithLogitsLossFlat()
    loss = criterion(outputs, labels)

    model.zero_grad()

    loss.backward()

    perturbed_images = images + epsilon * images.grad.sign()

    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images

if __name__ == "__main__":
    valid_transform = valid_transform = v2.Compose([
			v2.ToImage(),
			v2.ToDtype(torch.float32, scale=True),
			v2.Resize(256, interpolation=v2.InterpolationMode.BILINEAR,
					antialias=True),
			v2.Normalize(mean=[0.485, 0.456, 0.406], std=[
						0.229, 0.224, 0.225])
		])
    test_data = MoNuSegDataset(ROOT_DIR, transform=valid_transform)
    test_dl = MultiEpochsDataLoader(test_data, batch_size=BATCH_SIZE,
											shuffle=False, num_workers=NUM_WORKERS,
											pin_memory=True, sampler=None,
											persistent_workers=PERSISTENT_WORKERS)

    model = create_unet_model(resnet50, n_out=1, img_size=(256, 256), pretrained=True, weights=ResNet50_Weights.DEFAULT)
    learn = Learner(dls=test_dl, model=model, metrics=fastai.metrics.Dice(), cbs=[MixedPrecision(),], loss_func=BCEWithLogitsLossFlat())
    learn.load("fold_0_best")

    model.eval().to(DEVICE)

    image, annotation = test_data.__getitem__(0)
    image_tensor = valid_transform(image).unsqueeze(0).to(DEVICE)

    preds, _ = learn.get_preds(dl=test_dl)
    print(preds.size())
    image_index = 0

    # epsilons = [0.2 * i for i in range(6)]

    # for epsilon in epsilons:
    #     perturbed_image = fgsm_attack(model, image_tensor, annotation, epsilon)
    #     print("Epsilon value:", epsilon)
    #     # Evaluate the model on the perturbed image
    #     perturbed_outputs = model(perturbed_image)
    #     perturbed_prediction = perturbed_outputs.cpu().detach().numpy()

    #     if not os.path.exists("./perturbed_images"):
    #         os.makedirs("./perturbed_images")


    #     # # Display the segmentation result on the perturbed image
    #     # plt.figure(figsize=(8, 4))

    #     # plt.subplot(1, 2, 1)
    #     # plt.imshow(annotation, cmap="gray")
    #     # plt.title("Original Segmentation")

    #     # plt.subplot(1, 2, 2)
    #     # plt.imshow(perturbed_prediction[0][0], cmap="gray")
    #     # plt.title("Perturbed Segmentation")

    #     # plt.show()

    epsilons = [100 * i for i in range(6)]
    
    # plt.figure(figsize=(15, 10))

    for i, epsilon in enumerate(epsilons, 1):
        perturbed_image = fgsm_attack(model, image_tensor, annotation, epsilon)
        print("Epsilon value:", epsilon)

        # Evaluate the model on the perturbed image
        perturbed_outputs = model(perturbed_image)
        perturbed_prediction = perturbed_outputs #.cpu().detach().numpy()
        pa = PixelAccuracy(prediction_mask=perturbed_prediction, target_mask=annotation, device=DEVICE)
        acc = pa.pixel_accuracy()
        print("Accuracy: " , acc)

    #     plt.subplot(2, 3, i)
    #     plt.imshow(perturbed_prediction[0][0], cmap="gray")
    #     plt.title("Epsilon: {}".format(epsilon))

    # plt.show()