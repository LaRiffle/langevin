import torch
import torchvision
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Empty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def alexnet(args):
    assert (
        args.dataset == "cifar10"
    ), "Alexnet does not support this dataset for the moment."  # nosec

    alexnet = torchvision.models.alexnet(pretrained=True)

    if not args.full_train:
        for param in alexnet.parameters():
            param.requires_grad = False

    alexnet.avgpool = Empty()

    alexnet.classifier = nn.Linear(256, 10)

    # Adaptation to support CIFAR10
    alexnet.features[0].padding = (10, 10)
    alexnet.features[3].padding = (1, 1)
    alexnet.features[12] = Empty()  # remove last MaxPool

    alexnet.to(args.device)

    if args.full_train:
        parameters = alexnet.parameters()
    else:
        parameters = alexnet.classifier.parameters()

    return alexnet, parameters
