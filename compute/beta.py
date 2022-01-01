import numpy as np
import torch as th


def compute_beta(args, feature_train_loader):
    """
    Compute the smoothness parameter beta for Langevin DP

    Args:
        args: miscellaneous parameters
        feature_train_loader: beta is computed on the features of the train set
    """
    max_eigen_vals = []
    for batch_idx, (features, target) in enumerate(feature_train_loader):
        X = features
        n = X.shape[0]  # the batch size
        M = 1 / (2 * n) * X @ X.T
        eigen_vals = th.linalg.eigvalsh(M)
        max_eigen_val = eigen_vals.max()
        max_eigen_vals.append(max_eigen_val.cpu().numpy())

    if not args.silent:
        print("mean", np.mean(max_eigen_vals))
        print("median", np.median(max_eigen_vals))
        print("max", np.max(max_eigen_vals))
        print("min", np.min(max_eigen_vals))
        print("std", np.std(max_eigen_vals))

    beta = np.mean(max_eigen_vals)
    return beta
