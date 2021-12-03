import tqdm


def compute_features(args, feature_extractor, train_loader, test_loader):

    print("Compute training features")
    feature_train_loader = []
    for (data, target) in tqdm.tqdm(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        features = feature_extractor(data)
        feature_train_loader.append((features, target))

    print("Compute test features")
    feature_test_loader = []
    for (data, target) in tqdm.tqdm(test_loader):
        data, target = data.to(args.device), target.to(args.device)
        features = feature_extractor(data)
        feature_test_loader.append((features, target))

    return feature_train_loader, feature_test_loader
