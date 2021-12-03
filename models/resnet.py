import torchvision
import torch as th
from torch import nn


class Empty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def resnet(args):
    resnet = torchvision.models.resnet18(pretrained=False)

    # these ops are necessary to match the architecture of the model stored
    resnet.maxpool, resnet.relu = resnet.relu, resnet.maxpool
    resnet.fc = nn.Linear(in_features=512, out_features=100)

    path = "data/resnet-cifar100-best-model.pt"
    state_dict = th.load(path, map_location=args.device)
    resnet.load_state_dict(state_dict)

    resnet.fc = Empty()

    for param in resnet.parameters():
        param.requires_grad = False

    classifier = nn.Linear(in_features=512, out_features=args.out_features)
    # Put the appropriate Langevin initialization of the weights: theta_0 ~ proj( N(0, 2 sigma^2 / lambda) )
    std = 2 * args.sigma ** 2 / args.lambd
    nn.init.normal_(classifier.weight, mean=0.0, std=std)
    nn.init.normal_(classifier.bias, mean=0.0, std=std)

    resnet.to(args.device)
    classifier.to(args.device)

    resnet.eval()

    return resnet, classifier
