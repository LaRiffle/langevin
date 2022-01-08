import argparse
import datetime
import math

from opacus import PrivacyEngine
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from loaders import cifar10, pneumonia
from models.alexnet import alexnet
from models.resnet import resnet
from procedure.test import test
from procedure.train import sgd_train
from procedure.features import compute_features


def run(args):
    if not args.silent:
        print(f"Training over {args.epochs} epochs")
        print("with DP:\t", args.dp)
        print("model:\t\t", args.model)
        print("dataset:\t", args.dataset)
        print("batch_size:\t", args.batch_size)

    if args.model == "resnet18":
        feature_extractor, classifier = resnet(args)
    elif args.model == "alexnet":
        feature_extractor, classifier = alexnet(args)
    else:
        raise ValueError(f"Unknown model {args.model}")

    if args.dataset == "pneumonia":
        train_loader, test_loader = pneumonia(args)
    elif args.dataset == "cifar10":
        train_loader, test_loader = cifar10(args)
    else:
        raise ValueError("Unknown dataset")

    # Compute the features once to be more efficient, load beta as well if needed
    train_loader, test_loader = compute_features(args, feature_extractor, train_loader, test_loader)

    # Auto set the lr to 1 / beta if needed (beta can only be accessed now)
    if args.lr == -1:
        multiplier = 10 ** 4
        args.lr = int(1 / args.beta * multiplier) / multiplier
        if not args.silent and not args.decreasing:
            print(f"AUTO: lr set to 1 / beta = {args.lr}")

    # args.decreasing overwrites the value of args.lr if it was provided
    if args.decreasing:
        multiplier = 10 ** 4
        args.lr = int(1 / (2 * args.beta) * multiplier) / multiplier
        if not args.silent:
            print(f"AUTO: lr set to 1 / (2 * beta) = {args.lr}")

    training_arguments = {
        key: getattr(args, key) for key in dir(type(args)) if not key.startswith("_")
    }
    args.hparam_dict = training_arguments
    args.metric_dict = {}
    if not args.silent:
        print(training_arguments)

    optimizer_kwargs = dict(lr=args.lr, weight_decay=args.lambd)
    if args.optim == "sgd":
        optimizer = optim.SGD(classifier.parameters(), **optimizer_kwargs)
    elif args.optim == "adam":
        raise NotImplementedError
        optimizer = optim.Adam(
            classifier.parameters(), betas=(args.beta1, args.beta2), **optimizer_kwargs
        )

    if args.dp == "renyi":
        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=classifier,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.L,
        )
    elif args.dp == "langevin":
        # The Opacus privacy engine is used to clip properly the gradients, but no noise is added
        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=classifier,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=0,  # The noise is added separately using the Langevin approach
            max_grad_norm=args.L,
        )
    else:
        privacy_engine = None

    writer = SummaryWriter()

    accuracies = []
    alphas = []
    for epoch in range(args.epochs):
        epsilon, alpha = sgd_train(args, classifier, train_loader, optimizer, privacy_engine, epoch)
        accuracy = test(args, classifier, test_loader, epoch)
        accuracies.append(accuracy)
        alphas.append(alpha)
        writer.add_scalar("accuracy/accuracy", accuracy, epoch)

    best_accuracy = round(max(accuracies), 2)
    if not args.silent:
        print("Best accuracy", best_accuracy)
    writer.close()

    if args.parameter_info:
        accuracies = torch.tensor(accuracies)
        best_idx = torch.argmax(accuracies)
        alpha = alphas[best_idx]

        beta_per_lambda = args.beta / args.lambd
        eps_n2_per_alpha_d = (1 * args.n_train ** 2) / (alpha * args.out_features)

        print("beta", args.beta)
        print("beta / lambda", beta_per_lambda)
        print("eps n^2 / (alpha d)", eps_n2_per_alpha_d)

        print("regime n^4", eps_n2_per_alpha_d / beta_per_lambda ** 2)

    return args.epochs, epsilon, best_accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        help="Model to use for inference (resnet18, alexnet)",
        default="resnet18",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to use (cifar10, pneumonia).",
        default="cifar10",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="Size of the batch to use. Default 128",
        default=128,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        help="Size of the batch to use for testing. Default: as batch_size",
        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train on. Default 30",
        default=30,
    )

    parser.add_argument(
        "--optim",
        type=str,
        help="Optimizer to use (sgd, adam)",
        default="sgd",
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate of the SGD. Default 1 / beta.",
        default=-1,
    )

    parser.add_argument(
        "--decreasing",
        help="Use a decreasing learning rate in 1 / (2 beta + lambda k / 2).",
        action="store_true",
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
        help="[needs --langevin] Smoothness constant estimation. Default AUTO-COMPUTED.",
        default=-1,
    )

    parser.add_argument(
        "--sigma",
        type=float,
        help="[needs --langevin] Gaussian noise variance defined as std = sqrt(2.σ^2/λ). Default 0.002",
        default=0.002,
    )

    parser.add_argument(
        "--noise_multiplier",
        type=float,
        help="[needs --renyi] Gaussian noise variance defined as std = noise_multiplier * max_grad_norm. Default 1.2",
        default=1.2,
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        help="Maximum gradient norm per sample. Default 20",
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Seed. Default 1",
        default=1,
    )

    parser.add_argument(
        "--silent",
        help="Hide display information",
        action="store_true",
    )

    parser.add_argument(
        "--compute_features_force",
        help="Force computation of the features even if already computed",
        action="store_true",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        help="Log intermediate metrics every n batches. Default 1000",
        default=1000,
    )

    parser.add_argument(
        "--parameter_info",
        help="Print extra information about the parameter values.",
        action="store_true",
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
        decreasing = cmd_args.decreasing

        if cmd_args.langevin:
            dp = "langevin"
        elif cmd_args.renyi:
            dp = "renyi"
        else:
            dp = False

        delta = cmd_args.delta
        alphas = range(1, 2000)
        L = cmd_args.max_grad_norm  # sensitivity of the loss function wrt one sample
        lambd = cmd_args.lambd  # λ-strong convexity of the loss function
        beta = cmd_args.beta  # β-smoothness of the loss function
        sigma = cmd_args.sigma  # [Langevin] Gaussian noise is N(0, 2.σ^2/λ)
        noise_multiplier = (
            cmd_args.noise_multiplier
        )  # [Renyi] Gaussian noise std is noise_multiplier * L
        k = 0  # Number of batches processed

        silent = cmd_args.silent
        parameter_info = cmd_args.parameter_info
        log_interval = cmd_args.log_interval
        compute_features_force = cmd_args.compute_features_force  # Recompute the features

    args = Arguments()

    torch.manual_seed(args.seed)
    run(args)
