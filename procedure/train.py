import numpy as np
import torch
from torch import nn

from compute.privacy import get_privacy_spent


def sgd_train(args, classifier, train_loader, optimizer, privacy_engine, epoch):
    classifier.train()

    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        output = classifier(data)

        loss = loss_fn(output, target)
        loss.backward()

        # The gradients are clipped properly thanks to the privacy engine and the
        # wrapper around the optimizer
        optimizer.step()

        # Add the proper noise depending on the DP method
        if args.dp == "langevin":
            factor = torch.tensor(2 * args.lr * args.sigma ** 2).sqrt()
            with torch.no_grad():
                for param in classifier.parameters():
                    param += (factor * torch.randn(param.shape)).to(args.device)

        args.k += 1
        if args.decreasing:
            args.lr = 1 / (2 * args.beta + args.lambd * args.k / 2)

        losses.append(loss.item())

        if not args.silent and (batch_idx + 1) % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    args.n_train,
                    100.0 * batch_idx * args.batch_size / args.n_train,
                    loss.item(),
                )
            )

    if args.dp == "langevin":
        epsilon, best_alpha = get_privacy_spent(args, epoch + 1)
    elif args.dp == "renyi":
        alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(
            delta=args.delta, alphas=alphas
        )

    if not args.silent:
        if args.dp:
            print(
                f"Epoch {epoch} : "
                f"Train  Loss: {np.mean(losses):.6f} "
                f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
            )
        else:
            print(f"Epoch {epoch} : Train  Loss: {np.mean(losses):.6f}")

    return (epsilon, best_alpha) if args.dp else (None, None)
