import torch
import torch.nn.functional as F


def test(args, classifier, test_loader):
    classifier.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = classifier(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= args.n_test

    accuracy = 100.0 * correct / args.n_test

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n".format(
            test_loss, correct, args.n_test, accuracy
        )
    )

    return accuracy
