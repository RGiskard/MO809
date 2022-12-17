from torch.nn.utils.prune import l1_unstructured
from models import LeNetMLP
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prune_layer(layer, amount):
    l1_unstructured(layer, "weight", amount)
    l1_unstructured(layer, "bias", amount)


def prune_net(net, amount):
    for idx, layer in enumerate(net.layers):
        if idx == len(net.layers) - 1:
            prune_layer(layer, amount/2) # We prune the last layer half as much
        else:
            prune_layer(layer, amount)


def copy_net(source, target):
    with torch.no_grad():
        for target_layer, source_layer in zip(target.layers, source.layers):
            target_layer.weight_orig.copy_(source_layer.weight)
            target_layer.bias_orig.copy_(source_layer.bias)


def reset_network(config, layers, net, initial_net):
    if config.initialisation=='same':
        print("Setting same initialisation")
        copy_net(initial_net, net)
    elif config.initialisation=='random':
        print("Setting random initialisation")
        copy_net(LeNetMLP(layers), net)  # Set random initialisation
    else:
        raise ValueError(f"incorrect initialisation value '{config.initialisation}'")


def get_pruned_proportion(net):
    total = 0
    pruned = 0

    for layer in net.layers:
        if not hasattr(layer, 'weight_mask'):
            return 0
        total += layer.weight_mask.numel() + layer.bias_mask.numel()
        pruned += (layer.weight_mask == 0).sum() + (layer.bias_mask == 0).sum()

    return (pruned / total).item()

