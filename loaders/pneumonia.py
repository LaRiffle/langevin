import os
import random

import numpy as np
import pandas as pd
import torch as th
from PIL import Image
from torch import manual_seed
from torchvision import transforms


def single_channel_loader(filename):
    """Converts `filename` to a grayscale PIL Image"""
    with open(filename, "rb") as f:
        img = Image.open(f).convert("RGB")
        return img.copy()


class PneumoniaDataset(th.utils.data.Dataset):
    def __init__(
        self,
        label_path="./data/pneumonia/Labels.csv",
        train=False,
        transform=None,
        batch_size=8,
        seed=1,
    ):
        super().__init__()
        random.seed(seed)
        manual_seed(seed)
        self.train = train
        self.labels = pd.read_csv(label_path)
        self.labels = self.labels[self.labels["Dataset_type"] == ("TRAIN" if train else "TEST")]
        self.transform = transform
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size)) - 1

    def __getitem__(self, index):
        imgs = []
        labels = []
        for i in range(self.batch_size):
            idx = index * self.batch_size + i
            row = self.labels.iloc[idx]
            label = th.tensor(row["Numeric_Label"]).long()
            path = "train" if self.train else "test"
            folder = {0: "normal", 1: "bacterial pneumonia", 2: "viral pneumonia"}[
                row["Numeric_Label"]
            ]
            full_path = os.path.join("./data/pneumonia", path, folder, row["X_ray_image_name"])
            img = single_channel_loader(full_path)
            if self.transform:
                img = self.transform(img)

            label = label.view(1)
            img = img.view(1, *img.shape)

            imgs.append(img)
            labels.append(label)

        imgs, labels = th.cat(imgs, 0), th.cat(labels, 0)
        return imgs, labels


def pneumonia(args):

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_loader = PneumoniaDataset(train=True, transform=transform, batch_size=args.batch_size)
    test_loader = PneumoniaDataset(train=False, transform=transform, batch_size=args.batch_size)

    args.n_train = len(train_loader.labels)
    args.n_test = len(test_loader.labels)

    return train_loader, test_loader
