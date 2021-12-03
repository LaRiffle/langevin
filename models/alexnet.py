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

    # alexnet = torchvision.models.alexnet(pretrained=True)

    class AlexNet(nn.Module):
        def __init__(self, num_classes=100):
            super(AlexNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    alexnet = AlexNet()

    path = "data/20211007-alexnet-cifar100-epoch-100.pt"
    state_dict = torch.load(path, map_location=args.device)
    alexnet.load_state_dict(state_dict)

    for param in alexnet.features.parameters():
        param.requires_grad = False

    # alexnet.avgpool = Empty()
    #
    # alexnet.classifier = nn.Linear(256, 10)
    #
    # # Adaptation to support CIFAR10
    # alexnet.features[0].padding = (10, 10)
    # alexnet.features[3].padding = (1, 1)
    # alexnet.features[12] = Empty()  # remove last MaxPool

    alexnet.to(args.device)

    parameters = alexnet.classifier.parameters()

    return alexnet, parameters
