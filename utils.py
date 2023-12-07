import torch
import random
import numpy as np


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def predict_by_max_logit(logits):
    return torch.argmax(logits, dim=-1)


def compute_accuracy_from_predictions(predictions, labels):
    """
    Compute classification accuracy.
    """
    return torch.mean(torch.eq(labels, predictions).float())