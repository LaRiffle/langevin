import copy
import datetime

import torch

from main import run

today = datetime.datetime.today()

hyperparams_config = {
    "lr": [0.01, 0.018],
    "lambd": [0.001, 0.002],
    "epochs": [30],
    "sigma": [0.001, 0.002, 0.003],
}

# Max LR
# cifar10 <> resnet18 : 0.0182
# cifar10 <> alexnet  : 0.0038
# pneumo  <> resnet18 : 0.0067


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
    lr = -1
    decreasing = True

    # "langevin", "renyi" or False
    dp = False

    delta = 1e-5
    alphas = range(1, 2000)
    L = 20  # sensitivity of the loss function wrt one sample
    lambd = 0.01  # λ-strong convexity of the loss function
    beta = -1  # β-smoothness of the loss function
    sigma = 0.002  # [Langevin] Gaussian noise is N(0, 2.σ^2/λ)
    noise_multiplier = 1.2  # [Renyi] Gaussian noise std is noise_multiplier * L

    silent = True
    log_interval = 1000
    compute_features_force = False  # Recompute the features


args = Arguments()

training_arguments = {key: getattr(args, key) for key in dir(Arguments) if not key.startswith("_")}
print("Base arguments")
print(training_arguments)


results = []


def explore_hyperparams(args, hyperparams, setting=""):
    hyperparam = list(hyperparams.keys())[0]
    values = hyperparams[hyperparam]
    hyperparams.pop(hyperparam)

    for value in values:
        new_setting = setting + f"{hyperparam}: {value}, "
        setattr(args, hyperparam, value)
        if len(hyperparams):
            new_hyperparams = copy.deepcopy(hyperparams)
            explore_hyperparams(args, new_hyperparams, new_setting)
        else:
            print(new_setting)
            if args.dp == "langevin":
                method = "DP-SGLD (Ours)"
            elif args.dp == "renyi":
                method = "DP-SGD"
            else:
                method = "No DP"

            dataset = args.dataset.upper() if "cifar" in args.dataset else args.dataset.capitalize()

            model = args.model.capitalize()

            epochs, epsilon, accuracy = run(args)
            if isinstance(epsilon, torch.Tensor):
                epsilon = epsilon.item()
            epsilon = round(epsilon, 2) if epsilon is not None else "-"
            print(f"{method} & {dataset} & {model} & {epochs} & {epsilon} & {accuracy}")
            results.append([method, dataset, model, epochs, epsilon, accuracy])


explore_hyperparams(args, hyperparams_config)

print(results)
