import torchvision
import torch as th
from torch import nn


class Empty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def resnet(args, finetuning=False):
    """
    Build a backbone model based on resnet18 and a head/classifier.

    Args:
        args:
        finetuning: when True, add in the classifier the last block of the resnet to allow for
            more finetuning. This only works without DP or with Rényi as implemented in Opacus.
    """
    pre_trained_from_imagenet = args.dataset == "pneumonia"

    resnet = torchvision.models.resnet18(pretrained=pre_trained_from_imagenet)

    # these ops are necessary to match the architecture of the model stored
    resnet.maxpool, resnet.relu = resnet.relu, resnet.maxpool
    resnet.fc = nn.Linear(in_features=512, out_features=100)

    if not pre_trained_from_imagenet:
        path = "data/resnet-cifar100-best-model.pt"
        state_dict = th.load(path, map_location=args.device)
        resnet.load_state_dict(state_dict)

    if finetuning:
        resnet.fc = nn.Linear(in_features=512, out_features=args.out_features)

        resnet_modules = list(resnet.children())

        backbone = nn.Sequential(*resnet_modules[:-3])
        head = nn.Sequential(*resnet_modules[-3:-1], nn.Flatten(), nn.Linear(512, 10))

        if args.dp == "renyi":
            from opacus.validators.module_validator import ModuleValidator

            head = ModuleValidator.fix(head)

        backbone.to(args.device)
        head.to(args.device)

        backbone = backbone.eval()
        head = head.train()

        return backbone, head
    else:
        resnet.fc = Empty()

        for param in resnet.parameters():
            param.requires_grad = False

        classifier = nn.Linear(in_features=512, out_features=args.out_features)

        if args.dp == "langevin":
            # Put the appropriate Langevin initialization of the weights: theta_0 ~ proj( N(0, 2 sigma^2 / lambda) )
            std = 2 * args.sigma ** 2 / args.lambd
            nn.init.normal_(classifier.weight, mean=0.0, std=std)
            nn.init.normal_(classifier.bias, mean=0.0, std=std)

        resnet.to(args.device)
        classifier.to(args.device)

        resnet.eval()

        return resnet, classifier
