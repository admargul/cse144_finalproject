import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    #set random seed for more reproducible results

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # set settings to trade speed for reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    #use gpu is available, otw cpu

    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_accuracy(outputs, labels):
    #compute num of correct preds in a batch

    #choose the class w/ highest logit score
    _, predicted = torch.max(outputs, 1)

    correct = (predicted == labels).sum().item()
    total = labels.size(0)

    return correct, total