import os
import dataclasses
from dataclasses import dataclass

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, EMNIST
from torchvision.transforms import Compose, Lambda, ToTensor, Grayscale, CenterCrop

from typing import Optional, List

from torchvision.transforms.transforms import Grayscale


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class ExperimentConfig:
    experiment_id: str
    target_prune_proportion: float
    iters: int
    n_epochs: int
    initialisation: str
    train_dataset: str
    eval_datasets: List[str]
    split_classes: str
    noise: bool


@dataclass
class ExperimentResult:
    prune_proportion: float
    accuracy: float
    iteration: int
    dataset: str
    classes: Optional[List[int]] = None


def create_log(experiment_id):

    if not os.path.exists('experiments'):
        os.makedirs('experiments')

    with open(f'experiments/{experiment_id}.csv', 'w') as f:
        config_names = list(ExperimentConfig.__annotations__.keys())
        experiment_names = list(ExperimentResult.__annotations__.keys())
        f.write(','.join(config_names + [n for n in experiment_names if n != 'config']))
        f.write('\n')


def write_result(experiment_config, experiment_result):
    with open(f'experiments/{experiment_config.experiment_id}.csv', 'a') as f:
        experiment_data = dataclasses.asdict(experiment_config)
        experiment_data['eval_datasets'] = ' '.join(experiment_data['eval_datasets'])
        experiment_data = list(experiment_data.values())
        result_data = [item for name, item in dataclasses.asdict(experiment_result).items()]
        f.write(','.join(map(str, experiment_data + result_data)))
        f.write('\n')


def log_instance(config, dataset, classes, pruned_proportion, i, accuracy):
    print(f"Proportion pruned: {pruned_proportion}")

    result = ExperimentResult(
        prune_proportion=pruned_proportion,
        accuracy=accuracy,
        iteration=i,
        classes=' '.join(map(str, classes)) if classes else classes,
        dataset=dataset
    )
    write_result(config, result)
    return pruned_proportion


class GaussianNoise:
    """https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745"""
    def __init__(self, mean=0, std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


def load_data(
    root: str,
    dataset: str,
    batch_size: int = 128,
    download: bool = True,
    classes: Optional[List[int]] = None,
    noise: bool = False
):
    if noise:
        ToFlatTensor = Compose([ToTensor(), GaussianNoise(), Lambda(lambda x: x.ravel())])
        ToGrayscaleFlatTensor = Compose([
            ToTensor(),
            Grayscale(),
            CenterCrop(28),
            GaussianNoise(),
            Lambda(lambda x: x.ravel())
        ])
    else:
        ToFlatTensor = Compose([ToTensor(), Lambda(lambda x: x.ravel())])
        ToGrayscaleFlatTensor = Compose([
            ToTensor(),
            Grayscale(),
            CenterCrop(28),
            Lambda(lambda x: x.ravel())
        ])

    if dataset == 'MNIST':
        train = MNIST(root, train=True, download=download, transform=ToFlatTensor)
        test = MNIST(root, train=False, download=download, transform=ToFlatTensor)
    elif dataset == 'FashionMNIST':
        train = FashionMNIST(root, train=True, download=download, transform=ToFlatTensor)
        test = FashionMNIST(root, train=False, download=download, transform=ToFlatTensor)
    elif dataset == 'CIFAR10':
        train = CIFAR10(root, train=True, download=download, transform=ToGrayscaleFlatTensor)
        test = CIFAR10(root, train=False, download=download, transform=ToGrayscaleFlatTensor)
    elif dataset == 'SumOfInputs':
        train = SumOfInputs(train=True)
        test = SumOfInputs(train=False)
    elif dataset == 'MaxGroup':
        train = MaxGroup(train=True)
        test = MaxGroup(train=False)
    else:
        raise NotImplementedError(f"Unknown dataset {dataset}")

    if classes is not None:
        train_idx = np.isin(train.targets, classes)
        test_idx = np.isin(test.targets, classes)
        
        train.data = train.data[train_idx]
        test.data = test.data[test_idx]
        train.targets = train.targets[train_idx]
        test.targets = test.targets[test_idx]

    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


def calc_accuracy(net, test):
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total

    return accuracy


class SumOfInputs(torch.utils.data.Dataset):
    def __init__(self, train):
        self.train = train

    def __len__(self):
        return 50000 if self.train else 10000

    def __getitem__(self, idx):
        x = torch.rand(28 * 28, dtype=torch.float32) * 10 / (28 * 28)
        y = torch.floor(torch.fmod(x.sum(), 10)).long()
        assert y < 10
        return x, y


class MaxGroup(torch.utils.data.Dataset):
    def __init__(self, train):
        self.train = train

    def __len__(self):
        return 50000 if self.train else 10000

    def __getitem__(self, idx):
        x = torch.rand(28 * 28, dtype=torch.float32) * 10 / (28 * 28)
        y = x.numpy().copy()
        intervals = list(range(0, len(x), len(x) // 10))
        y = np.array([y[idx1:idx2].sum() for idx1, idx2 in zip(intervals[:-1], intervals[1:])])
        y = np.argmax(y)
        return x, torch.tensor(y)
