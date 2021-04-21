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

        if args.noisy_training:
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
