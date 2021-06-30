import os
import random

import albumentations as a
import numpy as np
import pandas as pd
import torch as th
from PIL import Image
from torch import manual_seed
from torchvision import transforms
from tqdm import tqdm


def single_channel_loader(filename):
    """Converts `filename` to a grayscale PIL Image"""
    with open(filename, "rb") as f:
        img = Image.open(f).convert("RGB")
        return img.copy()


class PneumoniaDataset(th.utils.data.Dataset):
    def __init__(
        self,
        label_path="../PriMIA/data/Labels.csv",
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
            full_path = os.path.join("../PriMIA/data", path, folder, row["X_ray_image_name"])
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
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = PneumoniaDataset(train=True, transform=transform, batch_size=args.batch_size)

    val_mean_std = calc_mean_std(dataset)
    mean, std = val_mean_std
    transform = create_albu_transform(args, mean, std)

    train_loader = PneumoniaDataset(train=True, transform=transform, batch_size=args.batch_size)
    test_loader = PneumoniaDataset(train=False, transform=transform, batch_size=args.batch_size)

    return train_loader, test_loader


class AlbumentationsTorchTransform:
    def __init__(self, transform, **kwargs):
        # print("init albu transform wrapper")
        self.transform = transform
        self.kwargs = kwargs

    def __call__(self, img):
        # print("call albu transform wrapper")
        if Image.isImageType(img):
            img = np.array(img)
        elif th.is_tensor(img):
            img = img.cpu().numpy()
        img = self.transform(image=img, **self.kwargs)["image"]
        # if img.max() > 1:
        #     img = a.augmentations.functional.to_float(img, max_value=255)
        img = th.from_numpy(img)
        if img.shape[-1] < img.shape[0]:
            img = img.permute(2, 0, 1)
        return img


def create_albu_transform(args, mean, std):
    train_tf = transforms.RandomAffine(
        degrees=args.rotation,
        translate=(args.translate, args.translate),
        scale=(1.0 - args.scale, 1.0 + args.scale),
        shear=args.shear,
        #    fillcolor=0,
    )
    start_transformations = [
        a.Resize(args.inference_resolution, args.inference_resolution),
        a.RandomCrop(args.train_resolution, args.train_resolution),
    ]
    if args.clahe:
        start_transformations.extend(
            [
                a.FromFloat(dtype="uint8", max_value=1.0),
                a.CLAHE(always_apply=True, clip_limit=(1, 1)),
            ]
        )
    train_tf_albu = [
        a.VerticalFlip(p=args.individual_albu_probs),
    ]
    if args.randomgamma:
        train_tf_albu.append(a.RandomGamma(p=args.individual_albu_probs))
    if args.randombrightness:
        train_tf_albu.append(a.RandomBrightness(p=args.individual_albu_probs))
    if args.blur:
        train_tf_albu.append(a.Blur(p=args.individual_albu_probs))
    if args.elastic:
        train_tf_albu.append(a.ElasticTransform(p=args.individual_albu_probs))
    if args.optical_distortion:
        train_tf_albu.append(a.OpticalDistortion(p=args.individual_albu_probs))
    if args.grid_distortion:
        train_tf_albu.append(a.GridDistortion(p=args.individual_albu_probs))
    if args.grid_shuffle:
        train_tf_albu.append(a.RandomGridShuffle(p=args.individual_albu_probs))
    if args.hsv:
        train_tf_albu.append(a.HueSaturationValue(p=args.individual_albu_probs))
    if args.invert:
        train_tf_albu.append(a.InvertImg(p=args.individual_albu_probs))
    if args.cutout:
        train_tf_albu.append(
            a.Cutout(num_holes=5, max_h_size=80, max_w_size=80, p=args.individual_albu_probs)
        )
    if args.shadow:
        assert args.pretrained, "RandomShadows needs 3 channels"
        train_tf_albu.append(a.RandomShadow(p=args.individual_albu_probs))
    if args.fog:
        assert args.pretrained, "RandomFog needs 3 channels"
        train_tf_albu.append(a.RandomFog(p=args.individual_albu_probs))
    if args.sun_flare:
        assert args.pretrained, "RandomSunFlare needs 3 channels"
        train_tf_albu.append(a.RandomSunFlare(p=args.individual_albu_probs))
    if args.solarize:
        train_tf_albu.append(a.Solarize(p=args.individual_albu_probs))
    if args.equalize:
        train_tf_albu.append(a.Equalize(p=args.individual_albu_probs))
    if args.grid_dropout:
        train_tf_albu.append(a.GridDropout(p=args.individual_albu_probs))
    train_tf_albu.append(a.GaussNoise(var_limit=args.noise_std ** 2, p=args.noise_prob))
    end_transformations = [
        a.ToFloat(max_value=255.0),
        a.Normalize(mean, std, max_pixel_value=1.0),
    ]
    if not args.pretrained:
        end_transformations.append(a.Lambda(image=lambda x, **kwargs: x[:, :, np.newaxis]))
    train_tf_albu = AlbumentationsTorchTransform(
        a.Compose(
            [
                a.Compose(start_transformations),
                a.Compose(train_tf_albu, p=args.albu_prob),
                a.Compose(end_transformations),
            ]
        )
    )
    return transforms.Compose(
        [
            train_tf,
            train_tf_albu,
        ]
    )


def calc_mean_std(dataset, save_folder=None):
    """
    Calculates the mean and standard deviation of `dataset` and
    saves them to `save_folder`.
    Needs a dataset where all images have the same size
    """
    accumulated_data = []
    for d in tqdm(dataset, total=len(dataset), leave=False, desc="accumulate data in dataset"):
        if type(d) is tuple or type(d) is list:
            d = d[0]
        accumulated_data.append(d)
    if isinstance(dataset, th.utils.data.Dataset):
        accumulated_data = th.stack(accumulated_data)
    elif isinstance(dataset, th.utils.data.DataLoader):
        accumulated_data = th.cat(accumulated_data)
    else:
        raise NotImplementedError("don't know how to process this data input class")
    if accumulated_data.shape[1] in [1, 3]:  # ugly hack
        dims = (0, *range(2, len(accumulated_data.shape)))
    else:
        dims = (*range(len(accumulated_data.shape)),)
    std, mean = th.std_mean(accumulated_data, dim=dims)
    return mean, std
