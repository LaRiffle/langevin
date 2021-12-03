import pickle
import tqdm

import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader


class FeatureDataset(Dataset):
    def __init__(self, args=None, feature_extractor=None, data_loader=None):
        self.batch_size = None
        self.features = []
        self.targets = []

        if data_loader is not None:
            for (data, target) in tqdm.tqdm(data_loader):
                if self.batch_size is None:
                    self.batch_size = len(data)

                data, target = data.to(args.device), target.to(args.device)
                features = feature_extractor(data)

                self.features.append(features)
                self.targets.append(target)

            self.features = torch.cat(self.features, dim=0)
            self.targets = torch.cat(self.targets, dim=0)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: list):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.features[idx, :]
        targets = self.targets[idx]

        return features, targets


def compute_features(
    args, feature_extractor: Module, train_loader: DataLoader, test_loader: DataLoader
):
    """
    Pre-compute the feature using the feature extractor base model and returns new
    DataLoaders. This allows to speed up the computation, since the features are
    only computed once.
    In addition, there is a caching mechanism which stores the features between several
    computations with the same setting
    """

    feature_key = f"{args.date}-features-{args.model}-{args.dataset}"
    file_path = f"./data/tmp/{feature_key}.txt"

    if not args.compute_features_force:
        try:
            with open(file_path, "rb") as file:
                train_features, train_targets, test_features, test_targets = pickle.load(file)

                feature_train_dataset = FeatureDataset()
                feature_train_dataset.features = train_features
                feature_train_dataset.targets = train_targets
                feature_train_loader = DataLoader(
                    feature_train_dataset, batch_size=args.batch_size, shuffle=True
                )

                feature_test_dataset = FeatureDataset()
                feature_test_dataset.features = test_features
                feature_test_dataset.targets = test_targets
                feature_test_loader = DataLoader(
                    feature_test_dataset, batch_size=args.test_batch_size, shuffle=True
                )

            print("Features loaded!")
            return feature_train_loader, feature_test_loader
        except FileNotFoundError:
            pass

    print("Compute training features...")
    feature_train_dataset = FeatureDataset(args, feature_extractor, train_loader)
    feature_train_loader = DataLoader(
        feature_train_dataset, batch_size=args.batch_size, shuffle=True
    )

    print("Compute test features...")
    feature_test_dataset = FeatureDataset(args, feature_extractor, test_loader)
    feature_test_loader = DataLoader(
        feature_test_dataset, batch_size=args.test_batch_size, shuffle=True
    )

    with open(file_path, "wb") as file:
        pickle.dump(
            (
                feature_train_dataset.features,
                feature_train_dataset.targets,
                feature_test_dataset.features,
                feature_test_dataset.targets,
            ),
            file,
        )

    return feature_train_loader, feature_test_loader
