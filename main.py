import argparse

import torch

from alexnet import alexnet
from resnet import resnet


def run(args):
    print(f"Training over {args.epochs} epochs")
    print("model:\t\t", args.model)
    print("dataset:\t", args.dataset)
    print("batch_size:\t", args.batch_size)

    if args.model == "resnet18":
        resnet(args)
    elif args.model == "alexnet":
        alexnet(args)
    else:
        raise ValueError("")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        help="model to use for inference (alexnet, resnet18)",
        default="resnet18",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset to use (cifar10, pneumonia)",
        default="pneumonia",
    )

    # parser.add_argument(
    #     "--test",
    #     help="run testing on the complete test dataset",
    #     action="store_true",
    # )
    #
    # parser.add_argument(
    #     "--train",
    #     help="Fine tune for n epochs",
    #     action="store_true",
    # )

    parser.add_argument(
        "--full_train",
        help="Train *all* the layers",
        action="store_true",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="size of the batch to use",
        default=64,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        help="size of the batch to use",
        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="[needs --train] number of epochs to train on",
        default=30,
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="[needs --train] learning rate of the SGD",
        default=0.001,
    )

    parser.add_argument(
        "--momentum",
        type=float,
        help="[needs --train] momentum of the SGD",
        default=0,
    )

    parser.add_argument(
        "--langevin",
        help="Activate Langevin DP SGD",
        action="store_true",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        help="[needs --langevin] noise for the Langevin DP",
        default=0.01,
    )

    parser.add_argument("--verbose", help="show extra information and metrics", action="store_true")

    parser.add_argument(
        "--log_interval",
        type=int,
        help="[needs --test or --train] log intermediate metrics every n batches",
        default=10,
    )

    cmd_args = parser.parse_args()

    if cmd_args.langevin:
        if cmd_args.momentum != 0:
            print("WARNING: With DPSGD, momentum should be 0!")

    class Arguments:
        model = cmd_args.model.lower()
        dataset = cmd_args.dataset.lower()

        full_train = cmd_args.full_train

        # train = cmd_args.train or cmd_args.full_train
        # test = cmd_args.test or train

        batch_size = cmd_args.batch_size
        # Defaults to the train batch_size
        test_batch_size = cmd_args.test_batch_size or cmd_args.batch_size

        epochs = cmd_args.epochs
        lr = cmd_args.lr
        momentum = cmd_args.momentum

        langevin = cmd_args.langevin
        sigma = cmd_args.sigma

        verbose = cmd_args.verbose
        log_interval = cmd_args.log_interval

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = Arguments()

    run(args)
