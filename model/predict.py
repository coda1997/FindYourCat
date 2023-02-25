import os
from zipfile import ZipFile
from pathlib import Path
import torchvision
from torchvision import transforms
from torchvision import datasets
import torch

import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from PIL import Image

from typing import Tuple
from typing import List


def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device) -> torch.Tensor:

    pred_probs = []
    model.eval()

    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)



def load_custom_images() -> Path:
    custom_data_dir = custom_data_path
    return custom_data_dir


def predict_custom_images(model: torch.nn.Module,
                          data: datasets,
                          device: torch.device) -> None:
    data_list = []

    for sample in list(data):
        if len(data_list) < 10:
            data_list.append(sample)

    pred_probs = make_predictions(
        model=model,
        data=data_list,
        device=device
    )

    pred_classes = pred_probs.argmax(dim=1).numpy()
    print(pred_classes)

    return pred_classes


class ImageFolderCustom(Dataset):

    def __init__(self, targ_dir: str, transform: transforms):

        self.paths = list(Path(targ_dir).glob("*.jpg"))
        self.transform = transform

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        return self.transform(img)


categories=["butterfly","cat","chicken","cow","dog","elephant","horse","sheep","spider","squirrel"]

device = "cuda" if torch.cuda.is_available() else "cpu"

model=torch.load('animal_model.pt').to(device)
model.eval()

custom_data_dir = load_custom_images()
custom_data_transform = transforms.Compose(
    [transforms.Resize(size=(224, 224)),
    transforms.ToTensor()]
)

custom_data = ImageFolderCustom(
    targ_dir=custom_data_dir,
    transform=custom_data_transform
)

predict_custom_images(
    model=model,
    data=custom_data,
    device=device)