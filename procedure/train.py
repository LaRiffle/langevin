import numpy as np
import time

import torch


def sgd_train(args, model, train_loader, criterion, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        start_time = time.time()
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()

        if args.langevin:
            factor = torch.tensor(2 * args.lr * args.sigma ** 2).sqrt()

            with torch.no_grad():
                head = model.fc if hasattr(model, "fc") else model.classifier
                for param in head.parameters():
                    param += (factor * torch.randn(param.shape)).to(args.device)

        tot_time = time.time() - start_time

        if batch_idx % args.log_interval == 0:
            n_items = (
                len(train_loader.dataset)
                if hasattr(train_loader, "dataset")
                else len(train_loader.labels)
            )
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s ({:.3f}s/item) [{:.3f}]".format(  # noqa
                    epoch,
                    batch_idx * args.batch_size,
                    n_items,
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    tot_time,
                    tot_time / args.batch_size,
                    args.batch_size,
                )
            )


class LearningRateScheduler:
    """
    Available schedule plans:
    log_linear : Linear interpolation with log learning rate scale
    log_cosine : Cosine interpolation with log learning rate scale
    """

    def __init__(
        self,
        total_epochs: int,
        log_start_lr: float,
        log_end_lr: float,
        schedule_plan: str = "log_linear",
        restarts: int = None,
    ):
        if restarts == 0:
            restarts = None
        self.total_epochs = total_epochs if not restarts else total_epochs / (restarts + 1)
        if schedule_plan == "log_linear":
            self.calc_lr = lambda epoch: np.power(
                10,
                ((log_end_lr - log_start_lr) / self.total_epochs) * epoch + log_start_lr,
            )
        elif schedule_plan == "log_cosine":
            self.calc_lr = lambda epoch: np.power(
                10,
                (np.cos(np.pi * (epoch / self.total_epochs)) / 2.0 + 0.5)
                * abs(log_start_lr - log_end_lr)
                + log_end_lr,
            )
        else:
            raise NotImplementedError(
                "Requested learning rate schedule {} not implemented".format(schedule_plan)
            )

    def get_lr(self, epoch: int):
        epoch = epoch % self.total_epochs
        if (type(epoch) is int and epoch > self.total_epochs) or (
            type(epoch) is np.ndarray and np.max(epoch) > self.total_epochs
        ):
            raise AssertionError("Requested epoch out of precalculated schedule")
        return self.calc_lr(epoch)

    def adjust_learning_rate(self, optimizer: torch.optim.Optimizer, epoch: int):
        new_lr = self.get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
