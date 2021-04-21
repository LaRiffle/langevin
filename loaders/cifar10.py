import torch
import torchvision
from torchvision import transforms


def cifar10(args):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10("./data/", train=True, download=True, transform=transform),
        batch_size=args.batch_size,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10("./data/", train=False, download=True, transform=transform),
        batch_size=args.test_batch_size,
    )

    return train_loader, test_loader
