from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import os 
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)


def plot_loss_curves(train_losses, val_losses, cur_itrs):
    if not os.path.exists('results/loss_plots'):
        os.makedirs('results/loss_plots')

    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend(loc='upper right')
    plt.title('Training and Validation Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(f'results/loss_plots/loss_curves_iter_{cur_itrs}.png')
    print("Loss curves saved!!")

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
