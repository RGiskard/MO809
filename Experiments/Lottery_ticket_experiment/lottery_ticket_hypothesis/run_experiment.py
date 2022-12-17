import torch
from torch import nn
import torch.optim as optim

from tqdm import tqdm

from models import LeNetMLP, AlexNet
from pruning import prune_net, get_pruned_proportion, reset_network
from utils import (
    ExperimentConfig, create_log, load_data, calc_accuracy, log_instance
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_network(net, train, test, epochs):
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=1.2e-3)#, lr=1.2e-3)
    loss_fn = nn.CrossEntropyLoss()
    prog_bar = tqdm(list(range(epochs)))
    for epoch in prog_bar:
        for data in train:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            preds = net(inputs)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

        accuracy = calc_accuracy(net, test)
        prog_bar.set_postfix({"Accuracy": accuracy})

    return accuracy


def run_train_and_prune_loop(config,alexnet = False):
    if config.split_classes:
        classes = [0, 1, 2, 3, 4]
    else:
        classes = None
    
    train, test = load_data('data', dataset=config.train_dataset, classes=classes, noise=config.noise)

    eval_datasets = {}
    for dataset_name in config.eval_datasets:
        eval_datasets[dataset_name] = load_data('data', classes=classes, dataset=dataset_name, noise=config.noise)
    
    layers = [28*28, 300, 100, 10]
    net = LeNetMLP(layers)
    initial_net = LeNetMLP(layers)
    initial_net.load_state_dict(net.state_dict())

    per_round_prune_proportion = 1 - (1 - config.target_prune_proportion)**(1/config.iters)
    for i in range(config.iters):
        accuracy = train_network(net, train, test, config.n_epochs)
        pruned_proportion = get_pruned_proportion(net)
        prune_net(net, per_round_prune_proportion) 
        reset_network(config, layers, net, initial_net)
        log_instance(config, config.train_dataset, classes, pruned_proportion, i, accuracy)

        for dataset_name, eval_dataset in eval_datasets.items():
            train_eval, test_eval = eval_dataset
            accuracy = train_network(net, train_eval, test_eval, config.n_epochs)
            reset_network(config, layers, net, initial_net)
            log_instance(config, dataset_name, classes, pruned_proportion, i, accuracy)
            print(f"Accuracy on {dataset_name} = {accuracy}")


if __name__=="__main__":
    experiment_name = 'SGD Normal LR'
    create_log(experiment_name)
    for mode in ('same', 'random'):
        config = ExperimentConfig(
            experiment_id=experiment_name,
            initialisation=mode,
            target_prune_proportion=0.99,
            iters=7,
            n_epochs=10,
            split_classes=None,
            train_dataset='MNIST',
            eval_datasets=['FashionMNIST', 'CIFAR10', 'MaxGroup'],
            noise=True
        )
        run_train_and_prune_loop(config)
