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
        help="model to use for inference (alexnet, resnet18). Default resnet18.",
        default="resnet18",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset to use (cifar10, pneumonia). Default pneumonia.",
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
        help="size of the batch to use. Default 64.",
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
        help="[needs --train] number of epochs to train on. Default 30.",
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
        help="[needs --train] learning rate of the SGD. Default 0.001.",
        default=0.001,
    )

    parser.add_argument(
        "--momentum",
        type=float,
        help="[needs --train] momentum of the SGD. Default 0.",
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
        help="[needs --scheduler] Period of learning rate decay. Default 10.",
        default=10,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        help="[needs --scheduler] Multiplicative factor of learning rate decay. Default: 0.5",
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
        help="[needs --langevin] noise for the Langevin DP. Default 0.01.",
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

    if cmd_args.optim == "adam":
        if cmd_args.momentum != 0:
            raise ValueError("With Adam optimizer, momentum should not be set.")

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

        optim = cmd_args.optim
        lr = cmd_args.lr
        momentum = cmd_args.momentum

        scheduler = cmd_args.scheduler
        step_size = cmd_args.step_size
        gamma = cmd_args.gamma

        langevin = cmd_args.langevin
        sigma = cmd_args.sigma

        verbose = cmd_args.verbose
        log_interval = cmd_args.log_interval

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        pretrained = True
        batch_size = 200  # FIXME conflict with above
        train_resolution = 224
        inference_resolution = 224
        lr = 0.00025320576414208793  # FIXME conflict with above
        end_lr = 0.00005226049769018076 # 0.00025226049769018076
        restarts = 1
        beta1 = 0.580178104854167
        beta2 = 0.9307811218548168
        weight_decay = 3.6897202836578385e-12
        optim = "adam"  # FIXME conflict with above
        rotation = 26
        translate = 0.0
        scale = 0.36890351894258405
        shear = 8
        noise_std = 0.0367014748759099
        noise_prob = 0.7878737537443424
        albu_prob = 0.19488241828717623
        individual_albu_probs = 0.43795095886930185
        clahe = False
        randomgamma = True
        randombrightness = False
        blur = False
        elastic = False
        optical_distortion = False
        grid_distortion = False
        grid_shuffle = False
        hsv = False
        invert = True
        cutout = False
        shadow = True
        fog = False
        sun_flare = False
        solarize = True
        equalize = True
        grid_dropout = False

    args = Arguments()

    run(args)
