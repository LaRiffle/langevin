import numpy as np
import torch as th


def get_privacy_spent(args, epoch):
    """
    Computes the (epsilon, delta) privacy budget spent so far for Langevin DP.

    This method converts from an (alpha, epsilon)-DP guarantee for all alphas
    that the PrivacyEngine was initialized with. It returns the optimal alpha
    together with the best epsilon.

    Condition TODO : ℓ(θ; x) be an L-Lipschitz, λ-strongly convex, and β-smooth loss function on closed convex set C,
    Condition : assert eta < 1/beta
    Condition : initialization of the weight : theta_0 ~ proj( N(0, 2 sigma^2 / lambda) )
    """
    K = epoch
    L = args.L
    n = args.n_train
    beta = args.beta
    delta = args.delta
    eta = args.lr
    lambd = args.lambd
    sigma = args.sigma
    alphas = args.alphas

    assert eta < 1 / beta, "This condition is necessary for DP-SGLD."
    # I don't think this is relevant => epsilon_renyi = alpha * Sg ** 2 / (lambd * sigma**2 * n**2 * (1 - eta * beta)**2)
    epsilons = []
    for alpha in alphas:
        epsilon_renyi = (
            (4 * alpha * L ** 2)
            / (lambd * sigma ** 2 * n ** 2)
            * (1 - np.exp(-lambd * eta * K / 2).item())
        )
        epsilon_dwork = epsilon_renyi + th.log(th.tensor(1 / delta)) / (alpha - 1)
        epsilons.append(epsilon_dwork)

    epsilons = th.tensor(epsilons)
    best_idx = th.argmin(epsilons)
    best_epsilon = epsilons[best_idx]
    best_alpha = alphas[best_idx]
    return best_epsilon, best_alpha
