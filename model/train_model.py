import os
from pathlib import Path
import torchvision
from torchvision import transforms
from torchvision import datasets

import random
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from typing import Tuple
from typing import Dict
from typing import List
from timeit import default_timer as timer

from tqdm.auto import tqdm


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device) -> Tuple[float, float]:

    model.to(device).train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(y_pred, dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


def val_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device) -> Tuple[float, float, torch.Tensor]:

    model.eval()
    val_loss, val_acc = 0, 0
    y_preds = []

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            val_pred_logits = model(X)
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

            val_pred_labels = torch.argmax(val_pred_logits, dim=1)
            val_acc += ((val_pred_labels == y).sum().item() / len(val_pred_labels))
            y_preds.append(val_pred_labels.cpu())

    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)

    y_pred_tensor = torch.cat(y_preds)

    return val_loss, val_acc, y_pred_tensor


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device) -> Tuple[Dict, torch.Tensor]:

    results = {"train_loss": [],
             "train_acc": [],
             "val_loss": [],
             "val_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_acc, y_preds = val_step(
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train acc: {train_acc:.3f}, Val loss: {val_loss:.3f}, Val acc: {val_acc:.3f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    return results, y_preds

if __name__ == '__main__':

    image_path = Path("Animal_10")
    input_folder = image_path / "raw_img"

    train_dir = input_folder / "Dataset" / "train"
    val_dir = input_folder / "Dataset" / "val"

    train_transform = transforms.Compose(
        [transforms.Resize(size=(224, 224)),
         transforms.ToTensor()]
    )

    val_transform = transforms.Compose(
        [transforms.Resize(size=(224, 224)),
         transforms.ToTensor()]
    )

    train_data = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    val_data = datasets.ImageFolder(
        root=val_dir,
        transform=val_transform
    )

    BATCH_SIZE = 8

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    image_batch, label_batch = next(iter(train_dataloader))

    weights = torchvision.models.VGG16_Weights.DEFAULT
    auto_transforms = weights.transforms()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torchvision.models.vgg16(weights=weights).to(device)
    for param in model.features.parameters():
        param.requires_grad = False

    output_shape = len(train_data.classes)

    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, len(train_data.classes))])
    model.classifier = nn.Sequential(*features)

    torch.cuda.manual_seed(42)
    torch.manual_seed(42)

    EPOCHS = 15

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=0.001
    )

    start_time = timer()
    print(model)
    model_results, preds = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        device=device
    )

    trace_module=torch.jit.trace(model,torch.rand(1,3,224,224).to(device))
    trace_module.save("vgg16_model.pt")

    end_time = timer()
    print(f"Total learning time: {(end_time - start_time):.3f}")