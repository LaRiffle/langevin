import numpy as np
import torch
from torch import nn

from privacy.compute import get_privacy_spent


def sgd_train(args, model, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        output = model(data)

        loss = loss_fn(output, target)
        loss.backward()

        # Clip the gradient to ensure L-lipschitz
        if args.dp is not False:
            for param in model.fc.parameters():
                clipped_grad = torch.clip(param.grad, -args.L, args.L)
                param.grad = clipped_grad

        optimizer.step()

        # Add the proper noise depending on the DP method
        if args.dp == "langevin":
            factor = torch.tensor(2 * args.lr * args.sigma ** 2).sqrt()
            with torch.no_grad():
                for param in model.fc.parameters():
                    param += (factor * torch.randn(param.shape)).to(args.device)

        elif args.dp == "renyi":
            pass

        losses.append(loss.item())
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    if args.dp == "langevin":
        epsilon, best_alpha = get_privacy_spent(args, epoch)
    elif args.dp == "renyi":
        # TODO: support Opacus 1.0
        # epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(delta=args.delta, alphas=args.alphas)
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)

    print(
        f"Train Epoch: {epoch} \t"
        f"Loss: {np.mean(losses):.6f} "
        f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
        if args.dp
        else ""
    )
