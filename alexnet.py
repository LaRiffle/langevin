from dataclasses import dataclass
from test import test

import torch.optim as optim
import torchvision
from torch import nn

from loaders import cifar10
from train import sgd_train


@dataclass
class Arguments:
    epochs = 30
    lr = 0.001
    batch_size = 32
    test_batch_size = 128
    log_interval = 50 * 4
    noisy_training = True
    sigma = 0.01
    device = "cpu"


args = Arguments()

train_loader, test_loader = cifar10(args)

alexnet = torchvision.models.alexnet(pretrained=True)
for param in alexnet.parameters():
    param.requires_grad = False


class Empty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


alexnet.avgpool = Empty()

alexnet.classifier = nn.Linear(256, 10)


# Adaptation to support CIFAR10
alexnet.features[0].padding = (10, 10)
alexnet.features[3].padding = (1, 1)
alexnet.features[12] = Empty()  # remove last MaxPool

alexnet.to(args.device)

criterion = nn.CrossEntropyLoss()
# FOR COMPLETE TRAINING optimizer = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.SGD(alexnet.classifier.parameters(), lr=args.lr)

# NOT SUPPORTED scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for i in range(args.epochs):
    sgd_train(args, alexnet, train_loader, criterion, optimizer, i)
    test(args, alexnet, test_loader)
