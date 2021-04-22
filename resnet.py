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

    criterion = nn.CrossEntropyLoss()
    # FOR COMPLETE TRAINING optimizer = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.SGD(resnet.parameters(), lr=args.lr)

    # NOT SUPPORTED scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for i in range(args.epochs):
        sgd_train(args, resnet, train_loader, criterion, optimizer, i)
        test(args, resnet, test_loader)
