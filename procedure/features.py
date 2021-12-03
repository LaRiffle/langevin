import pickle
import tqdm


def compute_features(args, feature_extractor, train_loader, test_loader):

    feature_key = f"{args.date}-features-{args.model}-{args.dataset}"
    file_path = f"./data/tmp/{feature_key}.txt"

    if not args.compute_features_force:
        try:
            with open(file_path, "rb") as file:
                feature_train_loader, feature_test_loader = pickle.load(file)
            print('Features loaded!')
            return feature_train_loader, feature_test_loader
        except FileNotFoundError:
            pass

    print("Compute training features...")
    feature_train_loader = []
    for (data, target) in tqdm.tqdm(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        features = feature_extractor(data)
        feature_train_loader.append((features, target))

    print("Compute test features...")
    feature_test_loader = []
    for (data, target) in tqdm.tqdm(test_loader):
        data, target = data.to(args.device), target.to(args.device)
        features = feature_extractor(data)
        feature_test_loader.append((features, target))

    with open(file_path, "wb") as file:
        pickle.dump((feature_train_loader, feature_test_loader), file)

    return feature_train_loader, feature_test_loader
