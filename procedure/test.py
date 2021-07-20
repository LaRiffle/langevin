import time

import torch


def test(args, model, test_loader):
    model.eval()
    correct = 0
    start_time = time.time()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()

    tot_time = time.time() - start_time
    n_items = (
        len(test_loader.dataset) if hasattr(test_loader, "dataset") else len(test_loader.labels)
    )
    print(
        "\nTest set: Accuracy: {}/{} ({:.0f}%) \tTime: {:.4f}s ({:.4f}s/item) \t [{:.3f}]\n".format(
            correct.item(),
            n_items,
            100.0 * correct.item() / n_items,
            tot_time,
            tot_time / n_items,
            args.test_batch_size,
        )
    )
