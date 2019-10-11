"""
This script generates a dataset similar to the MultiMNIST dataset
described in [1]. However, we remove any translation.

[1] Eslami, SM Ali, et al. "Attend, infer, repeat: Fast scene
understanding with generative models." Advances in Neural Information
Processing Systems. 2016.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pandas as pd
import numpy as np
import numpy.random as npr
from PIL import Image
from random import shuffle
from scipy.misc import imresize
from utils import transform

import torch
import torchvision.datasets as dset
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from utils import transform

class digit(Dataset):
    """Images with 0 to 4 digits of non-overlapping MNIST numbers.

    @param root: string
                 path to dataset root
    @param train: boolean [default: True]
           whether to return training examples or testing examples
    @param transform: ?torchvision.Transforms
                      optional function to apply to training inputs
    @param target_transform: ?torchvision.Transforms
                             optional function to apply to training outputs
    """
    processed_folder = 'digit'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, aug=False):
        self.root = os.path.expanduser(root)

        self.transformer = transforms.ToTensor()
        train_data, test_data = load_mnist()

        if train:
            self.data = train_data
        else:
            self.data = test_data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        a_data = transform(self.data['imgs'][index].numpy())
        b_data = self.data['labels'][index]

        return a_data, b_data, self.data['labels'][index].numpy(), index

    def __len__(self):
        # return self.input_a.size(0)
        return len(self.data['labels'])



def load_mnist():
    train_loader = torch.utils.data.DataLoader(
        dset.MNIST(root='../data/mnist', train=True, download=True))

    test_loader = torch.utils.data.DataLoader(
        dset.MNIST(root='../data/mnist', train=False, download=True))

    train_data = {
        'imgs': train_loader.dataset.train_data,
        'labels': train_loader.dataset.train_labels
    }

    test_data = {
        'imgs': test_loader.dataset.test_data,
        'labels': test_loader.dataset.test_labels
    }

    return train_data, test_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
