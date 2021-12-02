import torchvision
import torch as th
from torch import nn


def resnet(args):
    resnet = torchvision.models.resnet18(pretrained=False)

    # these ops are necessary to match the architecture of the model stored
    resnet.maxpool, resnet.relu = resnet.relu, resnet.maxpool
    resnet.fc = nn.Linear(in_features=512, out_features=100)

    path = "data/resnet-cifar100-best-model.pt"
    state_dict = th.load(path, map_location=args.device)
    resnet.load_state_dict(state_dict)

    for param in resnet.parameters():
        param.requires_grad = False

    ## Change the last layer to be ready for CIFAR10
    resnet.fc = nn.Linear(in_features=512, out_features=args.out_features)
    ## Put the appropriate Langevin initialization of the weights: theta_0 ~ proj( N(0, 2 sigma^2 / lambda) )
    std = 2 * args.sigma ** 2 / args.lambd
    nn.init.normal_(resnet.fc.weight, mean=0.0, std=std)
    nn.init.normal_(resnet.fc.bias, mean=0.0, std=std)

    resnet.to(args.device)

    parameters = resnet.fc.parameters()

    return resnet, parameters
