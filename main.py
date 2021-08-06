import argparse

import torch
from torch import nn
import torch.optim as optim

from loaders import cifar10, pneumonia
from models.alexnet import alexnet
from models.resnet import resnet
from procedure.test import test
from procedure.train import sgd_train


def run(args):
    print(f"Training over {args.epochs} epochs")
    print("model:\t\t", args.model)
    print("dataset:\t", args.dataset)
    print("batch_size:\t", args.batch_size)

    if args.dataset == "pneumonia":
        train_loader, test_loader = pneumonia(args)
    elif args.dataset == "cifar10":
        train_loader, test_loader = cifar10(args)
    else:
        raise ValueError("Unknown dataset")

    criterion = nn.CrossEntropyLoss()

    if args.model == "resnet18":
        model, parameters = resnet(args)
    elif args.model == "alexnet":
        model, parameters = alexnet(args)
    else:
        raise ValueError("Unknown model")

    optimizer_kwargs = dict(lr=args.lr, weight_decay=args.l2)
    if args.optim == "sgd":
        optimizer = optim.SGD(parameters, momentum=args.momentum, **optimizer_kwargs)
    elif args.optim == "adam":
        optimizer = optim.Adam(parameters, betas=(args.beta1, args.beta2), **optimizer_kwargs)

    if args.scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(args.epochs):
        sgd_train(args, model, train_loader, criterion, optimizer, epoch)
        test(args, model, test_loader)
        if args.scheduler:
            scheduler.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        help="model to use for inference (resnet18, alexnet)",
        default="resnet18",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset to use (pneumonia, cifar10)",
        default="pneumonia",
    )

    parser.add_argument(
        "--full_train",
        help="Train *all* the layers",
        action="store_true",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="size of the batch to use. Default 128",
        default=128,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        help="size of the batch to use for testing. Default: as batch_size",
        default=None,
    )

    parser.add_argument(
        "--l2",
        type=float,
        help="[not with --full_train] L2 regularization to make the logistic regression strongly convex. Default 0",
        default=0,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="[needs --train] number of epochs to train on. Default 30",
        default=30,
    )

    parser.add_argument(
        "--optim",
        type=str,
        help="optimizer to use (sgd, adam)",
        default="sgd",
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="[needs --train] learning rate of the SGD. Default 0.001",
        default=0.001,
    )

    parser.add_argument(
        "--beta1",
        type=float,
        help="[needs --optim adam] first beta parameter for Adam optimizer. Default 0.9",
        default=0.9,
    )

    parser.add_argument(
        "--beta2",
        type=float,
        help="[needs --optim adam] first beta parameter for Adam optimizer. Default 0.999",
        default=0.999,
    )

    parser.add_argument(
        "--momentum",
        type=float,
        help="[needs --train] momentum of the SGD. Default 0",
        default=0,
    )

    parser.add_argument(
        "--scheduler",
        help="Use a scheduler for the learning rate",
        action="store_true",
    )

    parser.add_argument(
        "--step_size",
        type=float,
        help="[needs --scheduler] Period of learning rate decay. Default 10",
        default=10,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        help="[needs --scheduler] Multiplicative factor of learning rate decay. Default 0.5",
        default=0.5,
    )

    parser.add_argument(
        "--langevin",
        help="Activate Langevin DP SGD",
        action="store_true",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        help="[needs --langevin] noise for the Langevin DP. Default 0.01",
        default=0.01,
    )

    parser.add_argument(
        "--verbose",
        help="show extra information and metrics",
        action="store_true",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        help="[needs --test or --train] log intermediate metrics every n batches. Default 10",
        default=10,
    )

    cmd_args = parser.parse_args()

    if cmd_args.langevin:
        if cmd_args.momentum != 0:
            print("WARNING: With DPSGD, momentum should be 0!")

    if cmd_args.optim == "adam":
        if cmd_args.momentum != 0:
            raise ValueError("With Adam optimizer, momentum should not be set.")
    else:
        if cmd_args.beta1 != 0.9 or cmd_args.beta2 != 0.999:
            raise ValueError("Don't set the betas if optim is not 'adam'.")

    class Arguments:
        model = cmd_args.model.lower()
        dataset = cmd_args.dataset.lower()

        full_train = cmd_args.full_train

        batch_size = cmd_args.batch_size
        # Defaults to the train batch_size
        test_batch_size = cmd_args.test_batch_size or cmd_args.batch_size

        l2 = cmd_args.l2

        epochs = cmd_args.epochs

        optim = cmd_args.optim
        lr = cmd_args.lr
        momentum = cmd_args.momentum
        beta1 = cmd_args.beta1
        beta2 = cmd_args.beta2

        scheduler = cmd_args.scheduler
        step_size = cmd_args.step_size
        gamma = cmd_args.gamma

        langevin = cmd_args.langevin
        sigma = cmd_args.sigma

        verbose = cmd_args.verbose
        log_interval = cmd_args.log_interval

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = Arguments()

    run(args)
