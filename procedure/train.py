import random
import time
from typing import Iterable

import einops
import numpy as np
import torch
from loguru import logger


def sgd_train_augmented(
    args, model, train_loader, criterion, optimizer, epoch, transformation_list
):
    """
    Handmade DPSGD with microbatching
    Pyvacy is a good start: https://github.com/ChrisWaites/pyvacy/blob/master/pyvacy/optim/dp_optimizer.py
    """
    model.train()

    # TODO: To improve efficiency
    # - pack different microbatches on the GPU as the same time (but you have to keep track of the norms)
    # - implement the speed trick
    # - plug into Opacus
    # - use Jax?

    # TODO: try foreach
    # TODO: later, try torch.vmap

    logger.info(f"N params: {len([p for (_, p) in model.named_parameters() if p.requires_grad])}")

    for batch_idx, (data, target) in enumerate(train_loader):
        start_time = time.time()

        # Initialize/clean-up accumulated microbatch grads
        # TODO: check init and update!
        # TODO: check scaling

        for param in [p for (_, p) in model.named_parameters() if p.requires_grad]:
            param.accumulated_microbatch_grad = torch.zeros_like(param.data)

        for image, label in zip(data, target):

            # Build a microbatch with a single augmented example
            image_list = [t(image) for t in transformation_list]
            microbatch_data = einops.rearrange(image_list, "b h w c -> b h w c")

            # The label is the same for the whole microbatch
            microbatch_target = einops.rearrange([label for _ in transformation_list], "b -> b")

            # Forward and backward pass as usual
            microbatch_data, microbatch_target = microbatch_data.to(
                args.device
            ), microbatch_target.to(args.device)
            optimizer.zero_grad()
            output = model(microbatch_data)
            loss = criterion(output, microbatch_target)
            loss.backward()

            # Calculate the clipping coefficient for the microbatch gradients
            microbatch_sq_norm = 0.0
            for param in [p for (_, p) in model.named_parameters() if p.requires_grad]:
                microbatch_sq_norm += param.grad.data.norm(2).item() ** 2
            microbatch_grad_norm = microbatch_sq_norm ** 0.5
            clip_coef = min(args.max_per_sample_grad_norm / (microbatch_grad_norm + 1e-6), 1.0)

            # Clip and save the gradients for later
            for param in [p for (_, p) in model.named_parameters() if p.requires_grad]:
                param.accumulated_microbatch_grad.add_(param.grad.data.mul(clip_coef))

        # Take the accumulated gradients and put them into the regular gradient field
        # + add noise and scale
        for param in [p for (_, p) in model.named_parameters() if p.requires_grad]:
            param.grad.data = param.accumulated_microbatch_grad.clone()
            param.grad.data.add_(
                args.max_per_sample_grad_norm * args.sigma * torch.randn_like(param.grad.data)
            )
            # We added `args.batch_size` microbatches, where each had gradients below `args.max_per_sample_grad_norm`
            param.grad.data.mul_(1 / args.batch_size)

        # Finally, take a vanilla optimizer step
        optimizer.step()

        tot_time = time.time() - start_time

        if batch_idx % args.log_interval == 0:
            n_items = (
                len(train_loader.dataset)
                if hasattr(train_loader, "dataset")
                else len(train_loader.labels)
            )
            logger.info(
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


def sgd_train_augmented_cheap(
    args, model, train_loader, criterion, optimizer, epoch, transformation_list
):
    """
    Cheap data augmentation trick? (instead of implementing microbatching)
    Accumulate the gradients inplace and let Opacus do the hard work.
    """
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        start_time = time.time()

        optimizer.zero_grad()
        target = target.to(args.device)

        # Run k backward passes before we clip/noise/step
        for i, transformation in enumerate(transformation_list):
            if batch_idx == 0:
                print(f"Transformation {i}")

            transformed_data = (transformation(data)).to(args.device)

            output = model(transformed_data)

            # Accumulate the loss for each microbatch
            loss = (1 / len(transformation_list)) * criterion(output, target)
            loss.backward()
            # BUG: Opacus doesn't accumulate the p.grad_sample properly
            # We would need something like virtual_step()

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


def sgd_train(args, model, train_loader, criterion, optimizer, epoch, transformation_list):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        start_time = time.time()

        # Simple data augmentation (not using the dataloader)
        # TODO: fix large pool of transformations and sample (feed a generator + max_nb of samples)
        transformation = random.choice(transformation_list)
        data, target = (transformation(data)).to(args.device), target.to(args.device)
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
