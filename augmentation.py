import random

from torchvision import transforms


def flip_rotate(n_transformations: int) -> list:
    transformation_list = [
        transforms.RandomRotation(0),  # Identity
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),
    ]
    for _ in range(n_transformations - 2):
        transformation_list.append(transforms.RandomRotation(180))
    transformation_list = transformation_list[:n_transformations]
    random.shuffle(transformation_list)
    return transformation_list


# TODO: check out albumentation/pneumonia
