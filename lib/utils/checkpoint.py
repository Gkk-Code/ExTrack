"""Load trusted local training checkpoints, including optimizer metadata."""

import torch


def load_training_checkpoint(path, map_location='cpu'):
    # PyTorch >= 2.6 defaults to weights_only=True, unlike the original framework.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
