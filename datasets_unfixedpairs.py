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
import torch
import torchvision.datasets as dset
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from utils import transform
import random


class DIGIT(Dataset):
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

    def __init__(self, root, train=True):
        self.root = os.path.expanduser(root)

        # self.transformA        = transforms.Compose([
        #                      transforms.Resize(32),
        #                      transforms.ToTensor()
        #                  ])
        # self.transformB        = transforms.Compose([
        #                      transforms.ToTensor()
        #                  ])
        self.train = train  # training set or test set

        self.input_a, self.input_b = make_dataset_fixed(self.train)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        identity_idx = np.mod(index, 10)
        a_imgs, b_imgs = self.input_a[identity_idx], self.input_b[identity_idx]

        if self.train:
            a_img = random.choice(a_imgs)
            b_img = random.choice(b_imgs)
        else:
            a_img = a_imgs[index]
            b_img = b_imgs[index]

        a_img = transform(a_img, resize=32)
        b_img = transform(b_img, resize=32)

        # svhn_img = np.transpose(svhn_img, (1, 2, 0))
        # svhn_img = Image.fromarray(svhn_img, mode='RGB')
        # svhn_img = self.transformB(svhn_img)

        return a_img, b_img, index

    def __len__(self):
        cnt = 0
        for i in range(self.input_a.__len__()):
            cnt += len(self.input_a[i]) * len(self.input_b[i])
        return cnt


def load_mnist():
    train_loader = torch.utils.data.DataLoader(
        dset.MNIST(root='../data/mnist', train=True, download=True))

    test_loader = torch.utils.data.DataLoader(
        dset.MNIST(root='../data/mnist', train=False, download=True))

    train_data = {
        'imgs': train_loader.dataset.train_data.numpy(),
        'labels': train_loader.dataset.train_labels.numpy()
    }

    test_data = {
        'imgs': test_loader.dataset.test_data.numpy(),
        'labels': test_loader.dataset.test_labels.numpy()
    }

    return train_data, test_data


def load_svhn():
    train_loader = torch.utils.data.DataLoader(
        dset.SVHN(root='../data/svhn', split='train', download=True))

    test_loader = torch.utils.data.DataLoader(
        dset.SVHN(root='../data/svhn', split='test', download=True))

    train_data = {
        'imgs': train_loader.dataset.data,
        'labels': train_loader.dataset.labels
    }

    test_data = {
        'imgs': test_loader.dataset.data,
        'labels': test_loader.dataset.labels
    }

    return train_data, test_data


def load_fashionMNIST():
    train_loader = torch.utils.data.DataLoader(
        dset.FashionMNIST(root='../data/fMNIST', train=True, download=True))

    test_loader = torch.utils.data.DataLoader(
        dset.FashionMNIST(root='../data/fMNIST', train=False, download=True))

    train_data = {
        'imgs': train_loader.dataset.train_data.numpy(),
        'labels': train_loader.dataset.train_labels.numpy()
    }

    test_data = {
        'imgs': test_loader.dataset.test_data.numpy(),
        'labels': test_loader.dataset.test_labels.numpy()
    }

    return train_data, test_data


def match_label(mnist, svhn):
    a_imgs = {}
    b_imgs = {}

    for i in range(10):
        a_imgs.update({i: []})
        b_imgs.update({i: []})
    for i in range(len(mnist['labels'])):
        a_imgs[mnist['labels'][i]].append(mnist['imgs'][i])
    for i in range(len(svhn['labels'])):
        b_imgs[svhn['labels'][i]].append(svhn['imgs'][i])

    # x_img, y_img = [], []
    # for i in range(10):
    #     min_len = min(len(mnist_imgs[i]), len(svhn_imgs[i]))
    #     print('label {} len {}'.format(i, min_len))
    #     x_img.extend(mnist_imgs[i][:min_len])
    #     y_img.extend(svhn_imgs[i][:min_len])

    for i in range(10):
        print('i = {} : len of inputA={}, len of inputB={}'.format(i, len(a_imgs[i]), len(b_imgs[i])))
        # a_imgs[i] = np.array(a_imgs[i])
        # b_imgs[i] = np.array(b_imgs[i])

    return a_imgs, b_imgs


def make_dataset_fixed(train):
    np.random.seed(681307)
    trainA, testA = load_mnist()
    trainB, testB = load_fashionMNIST()

    if train:
        input_a, input_b = match_label(trainA, trainB)
    else:
        input_a, input_b = match_label(testA, testB)

    return input_a, input_b


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--min-digits', type=int, default=0,
    #                     help='minimum number of digits to add to an image')
    # parser.add_argument('--max-digits', type=int, default=4,
    #                     help='maximum number of digits to add to an image')
    # parser.add_argument('--no-resize', action='store_true', default=False,
    #                     help='if True, fix the image to be MNIST size')
    # parser.add_argument('--no-translate', action='store_true', default=False,
    #
    # args = parser.parse_args()
    # args.resize = not args.no_resize
    # args.translate = not args.no_translate
    #
    # if args.no_repeat and not args.fixed:
    #     raise Exception('Must have --fixed if --no-repeat is supplied.')
    #
    # if args.scramble and not args.fixed:
    #     raise Exception('Must have --fixed if --scramble is supplied.')
    #
    # if args.reverse and not args.fixed:
    #     raise Exception('Must have --fixed if --reverse is supplied.')
    #
    # if args.reverse and args.scramble:
    #     print('Found --reversed and --scrambling. Overriding --reversed.')
    #     args.reverse = False

    # Generate the training set and dump it to disk. (Note, this will
    # always generate the same data, else error out.)
    make_dataset_fixed('./data', 'digit', 'training.pt', 'test.pt')