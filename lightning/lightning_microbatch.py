"""
Prototype for a faster/distributed microbatched SGD
Using Alexnet + CIFAR10 for now
"""


import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class DPAlexnet(pl.LightningModule):
    # TODO: this could be a black-box around any DP module
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.alexnet(pretrained=True)
        # for param in self.alexnet.parameters():
        #     param.requires_grad = False

        class Empty(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        # Adaptation to support CIFAR10
        self.model.avgpool = Empty()
        self.model.classifier = nn.Linear(256, 10)
        self.model.features[0].padding = (10, 10)
        self.model.features[3].padding = (1, 1)
        self.model.features[12] = Empty()  # remove last MaxPool

    def forward(self, x):
        return self.model(x)

    # TODO: Lightning has hooks for batches (for the augmentation), but they don't support multi GPU yet

    def microbatch_augment(minibatch_data, minibatch_target):
        # TODO: randomize?
        image_list = []
        target_list = []
        for image, label in zip(minibatch_data, minibatch_target):
            image_list += [t(image) for t in self.transformation_list]
            target_list += [[label for _ in self.transformation_list]]

        # Concatenate along batch dimension
        microbatch_data = einops.rearrange(image_list, "b h w c -> b h w c")
        microbatch_target = einops.rearrange(target_list, "b h w c -> b h w c")

        return microbatch_data, microbatch_target

    def training_step(self, batch, batch_idx):

        # No per sample gradients for now
        assert len(batch[0]) == 1

        minibatch_data, minibatch_target = batch
        microbatch_data, microbatch_target = self.microbatch_augment(
            minibatch_data, minibatch_target
        )

        microbatch_output = self.model(microbatch_data)

        loss = nn.CrossEntropyLoss(microbatch_output, microbatch_target)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        print(list(self.named_parameters())[0:2])
        for param in [p for (_, p) in self.named_parameters() if p.requires_grad]:
            print(param)
            print(param.grad)
            # print(param.grad.data.norm(2).item() ** 2)
            break
        optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# TODO: What if we have two nested training loops? Batch + microbatch
# Two ideas:
# 1. Batch of size 1 and replace the outer loop by gradient accumulation
# 2. Fit multiple microbatches in the same GPU, clip per microbatch. Amplify in the per_loading step.
# ----> problem for solution 2.: no per-gradient computation in this case

# FIXME: Hmmm, hard to do multiple backwards (solution 1) for a single optimizer step.

# TODO: check manual optimization! https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html#gradient-accumulation


def main(args):
    print(args.batch_size)
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset)

    # init model
    autoencoder = LitAutoEncoder()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(autoencoder, train_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    main(args)
