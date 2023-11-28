# import os
# import torch
# import numpy as np
# import fastai
# import matplotlib.pyplot as plt
# import foolbox
# from torchvision.transforms import functional as F
# from PIL import Image
# from foolbox import PyTorchModel, attacks
# from utils.dataset import MoNuSegDataset, MultiEpochsDataLoader
# from fastai.vision.all import (BCEWithLogitsLossFlat, DataLoader, ResNet50_Weights,
#                                resnet50, unet_learner, DynamicUnet, MixedPrecision)
# from fastai.vision.learner import Learner, create_unet_model
# from torchvision.transforms import v2

# # Set device
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ROOT_DIR = "MoNuSegTestData"
# BATCH_SIZE = 8
# NUM_WORKERS = 4
# PERSISTENT_WORKERS = True
# TARGET_CLASS = 1

# def visualize_segmentation(image, segmentation, ground_truth):
#     plt.figure(figsize=(15, 10))
#     plt.subplot(1, 3, 1)
#     plt.imshow(image.cpu().numpy().transpose(1, 2, 0))
#     plt.title("Input Image")

#     plt.subplot(1, 3, 2)
#     plt.imshow(segmentation.cpu().numpy(), cmap="gray")
#     plt.title("Segmentation Result")

#     plt.subplot(1, 3, 3)
#     plt.imshow(ground_truth.cpu().numpy(), cmap="gray")
#     plt.title("Ground Truth")

#     plt.show()

# if __name__ == "__main__":
#     valid_transform = v2.Compose([
#         v2.ToImage(),
#         v2.ToDtype(torch.float32, scale=True),
#         v2.Resize(256, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
#         v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     test_data = MoNuSegDataset(ROOT_DIR, transform=valid_transform)
#     test_dl = MultiEpochsDataLoader(test_data, batch_size=BATCH_SIZE,
#                                     shuffle=False, num_workers=NUM_WORKERS,
#                                     pin_memory=True, sampler=None,
#                                     persistent_workers=PERSISTENT_WORKERS)

#     model = create_unet_model(resnet50, n_out=1, img_size=(256, 256), pretrained=True, weights=ResNet50_Weights.DEFAULT)
#     learn = Learner(dls=test_dl, model=model, metrics=fastai.metrics.Dice(),
#                     cbs=[MixedPrecision(),], loss_func=BCEWithLogitsLossFlat())
#     learn.load("fold_0_best")

#     model = model.eval().to(DEVICE)
#     fmodel = PyTorchModel(model, bounds=(0, 1))

#     attack = attacks.LinfBasicIterativeAttack()

#     criterion = foolbox.criteria.TargetedMisclassification(np.zeros((256, 256)))

#     #loop thru all data
#     for i in range(test_data.__len__()):
#         image, annotation = test_data.__getitem__(i)
#         transformed_image = valid_transform(image).unsqueeze(0).to(DEVICE)
#         transformed_mask = valid_transform(annotation).unsqueeze(0).to(DEVICE)

#         epsilon_values = 0.2

#         adversarial_image = attack(fmodel, transformed_image, labels=torch.tensor([0]), epsilons=epsilon_values, criterion=criterion)

#         original_segmentation, _ = learn.get_preds(dl=test_dl)
#         adversarial_segmentation, _ = learn.get_preds(dl=test_dl, inputs=adversarial_image)

#         visualize_segmentation(transformed_image.squeeze(), original_segmentation[i][0], transformed_mask.squeeze())
#         visualize_segmentation(adversarial_image.squeeze(), adversarial_segmentation[i][0], transformed_mask.squeeze())



import os
import torch
import eagerpy as ep
import matplotlib.pyplot as plt
import fastai
from torchvision.transforms import functional as F
from PIL import Image
from foolbox import PyTorchModel, attacks
from utils.dataset import MoNuSegDataset, MultiEpochsDataLoader
from fastai.vision.all import (
    BCEWithLogitsLossFlat, DataLoader, ResNet50_Weights,
    resnet50, unet_learner, DynamicUnet, MixedPrecision
)
from fastai.vision.learner import Learner, create_unet_model
from torchvision.transforms import v2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = "MoNuSegTestData"
BATCH_SIZE = 8
NUM_WORKERS = 4
PERSISTENT_WORKERS = True

def boundary_attack(model, images, labels, epsilon):
    model = PyTorchModel(model, bounds=(0, 1), device=DEVICE, preprocessing=None)
    attack = attacks.BoundaryAttack()

    images = ep.astensor(images).to(DEVICE)
    labels = ep.astensor(labels).to(DEVICE)

    _, _, adv_images = attack(model, images, labels, epsilon=epsilon)

    return adv_images.tensor

if __name__ == "__main__":
    valid_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(256, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

    epsilons = [0.2 * i for i in range(6)]

    plt.figure(figsize=(15, 10))

    for i, epsilon in enumerate(epsilons, 1):
        perturbed_image = boundary_attack(model, image_tensor, annotation, epsilon)

        print("Epsilon value:", epsilon)

        # Evaluate the model on the perturbed image
        perturbed_outputs = model(perturbed_image)
        perturbed_prediction = perturbed_outputs.cpu().detach().numpy()

        plt.subplot(2, 3, i)
        plt.imshow(perturbed_prediction[0][0], cmap="gray")
        plt.title("Epsilon: {}".format(epsilon))

    plt.show()
