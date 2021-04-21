import time

import torch


def test(args, model, test_loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            start_time = time.time()
            output = model(data)
            pred = output.argmax(dim=1)
            tot_time = time.time() - start_time
            correct += pred.eq(target.view_as(pred)).sum()

    n_items = (
        len(test_loader.dataset) if hasattr(test_loader, "dataset") else len(test_loader.labels)
    )
    print(
        "\nTest set: Accuracy: {}/{} ({:.0f}%) \tTime /item: {:.4f}s \t [{:.3f}]\n".format(
            correct.item(),
            n_items,
            100.0 * correct.item() / n_items,
            tot_time / args.test_batch_size,
            args.test_batch_size,
        )
    )
