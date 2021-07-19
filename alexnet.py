import torch
import torch.optim as optim
import torchvision
from opacus import PrivacyEngine
from torch import nn
from torchvision import transforms

from loaders import cifar10
from procedure.test import test
from procedure.train import sgd_train, sgd_train_augmented

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#
# @dataclass
# class Arguments:
#     epochs = 30
#     lr = 0.001
#     batch_size = 32
#     test_batch_size = 128
#     log_interval = 50 * 4
#     noisy_training = True
#     sigma = 0.01
#     device = device


def alexnet(args):

    assert args.optim == "sgd"  # nosec
    assert args.scheduler is False  # nosec

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

    # TODO: decouple models and DP training/augmentation?
    if not args.disable_dp:

        # TODO: careful with augmentation
        sample_rate = args.batch_size / len(train_loader.dataset)
        privacy_engine = PrivacyEngine(
            alexnet,
            sample_rate=sample_rate,
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
            secure_rng=False,
        )
        if args.fixed_seed:
            privacy_engine._set_seed(10)
        privacy_engine.attach(optimizer)
    # NOT SUPPORTED scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # TODO: Fancier transformations

    transformation_list = [
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),
    ]
    for _ in range(args.data_aug_factor - 2):
        transformation_list.append(transforms.RandomRotation(180))
    transformation_list = transformation_list[: args.data_aug_factor]

    for i in range(args.epochs):

        if args.data_aug_factor > 1:
            sgd_train_augmented(
                args, alexnet, train_loader, criterion, optimizer, i, transformation_list
            )
        else:
            sgd_train(args, alexnet, train_loader, criterion, optimizer, i)
        test(args, alexnet, test_loader)

    if not args.disable_dp:
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
        print(f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}")
