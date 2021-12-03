import argparse
import datetime

from opacus import PrivacyEngine
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from loaders import cifar10, pneumonia
from models.alexnet import alexnet
from models.resnet import resnet
from procedure.test import test
from procedure.train import sgd_train


def run(args):
    print(f"Training over {args.epochs} epochs")
    print("with DP:\t", args.dp)
    print("model:\t\t", args.model)
    print("dataset:\t", args.dataset)
    print("batch_size:\t", args.batch_size)

    if args.dataset == "pneumonia":
        train_loader, test_loader = pneumonia(args)
    elif args.dataset == "cifar10":
        train_loader, test_loader = cifar10(args)
    else:
        raise ValueError("Unknown dataset")

    if args.model == "resnet18":
        model, parameters = resnet(args)
    elif args.model == "alexnet":
        raise NotImplementedError
        model, parameters = alexnet(args)
    else:
        raise ValueError("Unknown model")

    optimizer_kwargs = dict(lr=args.lr, weight_decay=args.lambd)
    if args.optim == "sgd":
        optimizer = optim.SGD(parameters, **optimizer_kwargs)
    elif args.optim == "adam":
        raise NotImplementedError
        optimizer = optim.Adam(parameters, betas=(args.beta1, args.beta2), **optimizer_kwargs)

    if args.dp == "renyi":
        privacy_engine = PrivacyEngine(
            model,
            sample_rate=0.01,
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),  # [10, 100],
            noise_multiplier=3.0,  # Control the noise
            max_grad_norm=1.0,
        )
        privacy_engine.attach(optimizer)

        # TODO: support Opacus 1.0
        # privacy_engine = PrivacyEngine()
        # model, optimizer, data_loader = privacy_engine.make_private(
        #     module=model.fc,
        #     optimizer=optimizer,
        #     data_loader=train_loader,
        #     noise_multiplier=1.1,
        #     max_grad_norm=1.0,
        # )
    else:
        privacy_engine = None

    writer = SummaryWriter()

    accuracies = []
    for epoch in range(args.epochs):
        sgd_train(args, model, train_loader, optimizer, privacy_engine, epoch)
        accuracy = test(args, model, test_loader)
        accuracies.append(accuracy)
        writer.add_scalar("accuracy/accuracy", accuracy, epoch)

    print(type(max(accuracies)))
    writer.add_hparams(args.hparam_dict, {"accuracy/accuracy": max(accuracies)})
    writer.close()


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
        help="dataset to use (cifar10, pneumonia).",
        default="cifar10",
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
        help="[needs --train] learning rate of the SGD. Default ???",
        default=0.0088,
    )

    parser.add_argument(
        "--langevin",
        help="Use Langevin DP SGD",
        action="store_true",
    )

    parser.add_argument(
        "--renyi",
        help="Use Renyi DP SGD and Opacus",
        action="store_true",
    )

    parser.add_argument(
        "--delta",
        type=float,
        help="delta constant in the DP budget. Default 1e-5",
        default=1e-5,
    )

    parser.add_argument(
        "--lambd",
        type=float,
        help="L2 regularization to make the logistic regression strongly convex. Default 0.01",
        default=0.01,
    )

    parser.add_argument(
        "--beta",
        type=float,
        help="[needs --optim adam] Smoothness constant estimation. Default 110.",
        default=110,
    )

    parser.add_argument(
        "--sigma",
        type=float,
        help="Noise for the Langevin DP. Default 0.001",
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Seed. Default 1",
        default=1,
    )

    parser.add_argument(
        "--verbose",
        help="show extra information and metrics",
        action="store_true",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        help="Log intermediate metrics every n batches. Default 100",
        default=100,
    )

    cmd_args = parser.parse_args()

    today = datetime.datetime.today()

    class Arguments:
        seed = 1
        date = f"{today.year}-{'0' if today.month < 10 else ''}{today.month}-{'0' if today.day < 10 else ''}{today.day}"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = cmd_args.model.lower()
        dataset = cmd_args.dataset.lower()
        n_train = -1  # size of the train data (is set later when loading the data)
        n_test = -1  # size of the test data (is set later when loading the data)
        if dataset == "cifar10":
            out_features = 10
        elif dataset == "pneumonia":
            out_features = 3
        else:
            raise ValueError(f"Dataset {dataset} is not recognized.")

        batch_size = cmd_args.batch_size
        test_batch_size = cmd_args.test_batch_size or cmd_args.batch_size
        epochs = cmd_args.epochs
        optim = cmd_args.optim
        lr = cmd_args.lr

        if cmd_args.langevin:
            dp = "langevin"
        elif cmd_args.renyi:
            dp = "renyi"
        else:
            dp = False

        delta = cmd_args.delta
        alphas = range(1, 2000)
        L = 1  # sensitivity of the total gradients: \sum_x_i \grad
        lambd = cmd_args.lambd  # λ-strong convexity of the loss function
        beta = cmd_args.beta  # β-smoothness of the loss function
        sigma = cmd_args.sigma  # Gaussian noise is N(0, 2.σ^2/λ)
        eta = lr

        verbose = cmd_args.verbose
        log_interval = cmd_args.log_interval

    args = Arguments()

    training_arguments = {
        key: getattr(args, key) for key in dir(Arguments) if not key.startswith("_")
    }

    args.hparam_dict = training_arguments
    args.metric_dict = {}
    print(training_arguments)

    run(args)
