import torch.optim as optim
import torchvision
from torch import nn

from loaders import pneumonia
from procedure.test import test
from procedure.train import sgd_train


def resnet(args):

    train_loader, test_loader = pneumonia(args)

    resnet = torchvision.models.resnet18(pretrained=True)

    if not args.full_train:
        for param in resnet.parameters():
            param.requires_grad = False

    resnet.fc = nn.Linear(512, 3)

    resnet.to(args.device)

    if args.full_train:
        parameters = resnet.parameters()
    else:
        parameters = resnet.fc.parameters()

    criterion = nn.CrossEntropyLoss()

    if args.optim == "sgd":
        optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum)
    elif args.optim == "adam":
        optimizer = optim.Adam(parameters, lr=args.lr, betas=(args.beta1, args.beta2))

    if args.scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(args.epochs):
        sgd_train(args, resnet, train_loader, criterion, optimizer, epoch)
        test(args, resnet, test_loader)
        if args.scheduler:
            scheduler.step()
