import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
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
