import torchvision
from torch import nn


def resnet(args):
    resnet = torchvision.models.resnet18(pretrained=True)

    if not args.full_train:
        for param in resnet.parameters():
            param.requires_grad = False

    assert (
        args.dataset == "pneumonia"
    ), "Resnet does not support this dataset for the moment."  # nosec

    resnet.fc = nn.Linear(512, 3)

    resnet.to(args.device)

    if args.full_train:
        parameters = resnet.parameters()
    else:
        parameters = resnet.fc.parameters()

    return resnet, parameters
