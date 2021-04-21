from dataclasses import dataclass
from test import test

import torch.optim as optim
import torchvision
from torch import nn

from loaders import pneumonia
from train import sgd_train


@dataclass
class Arguments:
    epochs = 30
    lr = 0.01
    batch_size = 32
    test_batch_size = 128
    log_interval = 10
    noisy_training = False
    sigma = 0.01
    device = "cpu"


args = Arguments()

train_loader, test_loader = pneumonia(args)

resnet = torchvision.models.resnet18(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False


resnet.fc = nn.Linear(512, 3)

resnet.to(args.device)

criterion = nn.CrossEntropyLoss()
# FOR COMPLETE TRAINING optimizer = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.SGD(resnet.fc.parameters(), lr=args.lr)

# NOT SUPPORTED scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for i in range(args.epochs):
    sgd_train(args, resnet, train_loader, criterion, optimizer, i)
    test(args, resnet, test_loader)
