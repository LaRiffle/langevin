import datetime

import torch

from main import run

today = datetime.datetime.today()

SEARCH_PARAMETER = None


class Arguments:
    seed = 1
    date = f"{today.year}-{'0' if today.month < 10 else ''}{today.month}-{'0' if today.day < 10 else ''}{today.day}"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = "resnet18"
    dataset = "cifar10"
    n_train = -1  # size of the train data (is set later when loading the data)
    n_test = -1  # size of the test data (is set later when loading the data)
    if dataset == "cifar10":
        out_features = 10
    elif dataset == "pneumonia":
        out_features = 3

    batch_size = 128
    test_batch_size = batch_size
    epochs = 30
    optim = "sgd"
    lr = 0.0088

    dp = "langevin"

    delta = 1e-5
    alphas = range(1, 2000)
    L = SEARCH_PARAMETER  # sensitivity of the loss function wrt one sample
    lambd = 0.01  # λ-strong convexity of the loss function
    beta = 110  # β-smoothness of the loss function
    sigma = SEARCH_PARAMETER  # [Langevin] Gaussian noise is N(0, 2.σ^2/λ)
    noise_multiplier = 1.2  # [Renyi] Gaussian noise std is noise_multiplier * L
    eta = lr

    verbose = False
    log_interval = 1000
    compute_features_force = False  # Recompute the features


args = Arguments()

args.sigma = 0.0005
args.L = 5

for i in range(4):
    print("\nRUN", args.sigma, args.L)
    run(args)

    args.sigma *= 2
    args.L *= 2
