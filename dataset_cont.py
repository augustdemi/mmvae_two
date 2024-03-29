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

import torch
import torchvision.datasets as dset
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from utils import transform

from utils import transform


class Position(Dataset):
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

    def __init__(self, root, train=True, aug=False):
        self.root = os.path.expanduser(root)

        self.train = train  # training set or test set
        self.aug = aug
        if self.aug:
            self.input_a, self.input_b, self.class_idxA, self.class_idxB, self.label, self.per_class_n_pairs = make_dataset_augment(self.train)
        else:
            self.input_a, self.input_b, self.a_idx, self.b_idx, self.label = make_dataset_fixed(self.train)



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.aug:
            a_idx, b_idx = self.get_pair(index)
            a_img, b_img, label = self.input_a[a_idx], self.input_b[b_idx], self.label[index]
        else:
            a_img, b_img, label = self.input_a[self.a_idx[index]], self.input_b[self.b_idx[index]], self.label[index]


        a_img = transform(a_img) * 255
        return a_img, b_img, label, index

    def __len__(self):
        # return self.input_a.size(0)
        return len(self.label)

    def get_pair(self, index):
        label = self.label[index]
        sum = np.array(self.per_class_n_pairs).cumsum()

        if label != 0:
            within_class_index = index - sum[label-1]
        else:
            within_class_index = index
        a_idx = int(within_class_index / len(self.class_idxB[label]))
        b_idx =  within_class_index % len(self.class_idxB[label])
        a_idx = self.class_idxA[label][a_idx]
        b_idx = self.class_idxB[label][b_idx]
        return a_idx, b_idx


def load_3dface():
    # latent factor = (id, azimuth, elevation, lighting)
    #   id = {0,1,...,49} (50)
    #   azimuth = {-1.0,-0.9,...,0.9,1.0} (21)
    #   elevation = {-1.0,0.8,...,0.8,1.0} (11)
    #   lighting = {-1.0,0.8,...,0.8,1.0} (11)
    # (number of variations = 50*21*11*11 = 127050)
    latent_classes, latent_values = np.load('../data/3dfaces/gt_factor_labels.npy')
    root = '../data/3dfaces/basel_face_renders.pth'
    data = torch.load(root).float().div(255)  # (50x21x11x11x64x64)
    data = data.view(-1, 64, 64).unsqueeze(1)
    n = latent_values.shape[0]
    class_idx = {}
    for i in range(121):
        class_idx.update({i:[]})

    face_pos_pair = {}
    idx = -1
    for i in np.round(np.linspace(1, -1, 11),2):
        for j in np.round(np.linspace(-1, 1, 11),2):
            idx += 1
            face_pos_pair.update({(i,j):idx})

    for i in range(n):
        az, el = np.round(latent_values[i, 1],2), np.round(latent_values[i, 2],2)
        if (az, el) in face_pos_pair.keys():
            idx = face_pos_pair[(az, el)]
            class_idx[idx].append(i)
    return data, class_idx


def load_dsprite():
    dataset_zip = np.load('../data/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding="latin1")

    imgs = dataset_zip['imgs']
    metadata = dataset_zip['metadata'][()]
    latents_sizes = metadata['latents_sizes']
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                    np.array([1, ])))

    def latent_to_index(latents):
        return np.dot(latents, latents_bases).astype(int)

    all_other_latents = []
    for i in range(3):
        for j in range(6):
            for k in range(31):
                all_other_latents.append([0, i, j, k])

    pos11 = [0, 4, 8, 12, 15, 16, 17, 20, 24, 28, 31]

    class_idx = {}
    for i in range(121):
        class_idx.update({i:[]})
    pos_pair = {}

    idx = -1
    for i in pos11:
        for j in pos11:
            idx += 1
            pos_pair.update({(i,j):idx}) # idx = cardinality of  all possible pairs: 0~120


    for i in pos11:
        for j in pos11:
            for one_latent in all_other_latents:
                latent = one_latent.copy()
                latent.extend([i, j]) # add posX, posY to (color, type, scale, rotation)
                idx = pos_pair[(i,j)]
                class_idx[idx].append(latent_to_index(latent))
    return imgs, class_idx


def augment_label(modalA, modalB):
    n_labels = len(modalA['labels'])
    class_pair = {}
    for i in range(n_labels):
        class_pair.update({i: []})
    per_class_n_pairs = []
    for i in range(n_labels):
        n_a_imgs = len(modalA['labels'][i])
        n_b_imgs = len(modalB['labels'][i])
        per_class_n_pairs.append(n_a_imgs * n_b_imgs)

    all_index_pairs, all_labels = [], []
    for i in range(n_labels):
        all_labels.extend([i] * per_class_n_pairs[i])

    return per_class_n_pairs, all_labels


def make_dataset_augment(train):
    np.random.seed(681307)
    imgsA, class_idxA = load_dsprite()
    imgsB, class_idxB = load_3dface()
    modalA = {'imgs': imgsA, 'labels': class_idxA}
    modalB = {'imgs': imgsB, 'labels': class_idxB}
    per_class_n_pairs, all_labels = augment_label(modalA, modalB)
    return imgsA, imgsB, class_idxA, class_idxB, all_labels, per_class_n_pairs




def match_label(modalA, modalB):

    a_idx, b_idx, labels = [], [], []
    for i in range(len(modalA['class_idx'])):
        class_idxA = modalA['class_idx'][i]
        class_idxB = modalB['class_idx'][i]
        print('label {} len A, B: {}, {}'.format(i, len(class_idxA), len(class_idxB)))
        min_len = min(len(class_idxA), len(class_idxB))
        if len(class_idxA) > min_len:
            b_idx.extend(class_idxB)
            np.random.shuffle(class_idxA)
            a_idx.extend(class_idxA[:len(class_idxB)])
        elif len(class_idxA) < min_len:
            a_idx.extend(class_idxA)
            np.random.shuffle(class_idxB)
            b_idx.extend(class_idxB[:len(class_idxA)])
        else:
            a_idx.extend(class_idxA)
            b_idx.extend(class_idxB)
        labels.extend([i] * min_len)
    return a_idx, b_idx, labels


def make_dataset_fixed(train):
    np.random.seed(681307)
    imgsA, class_idxA = load_dsprite()
    imgsB, class_idxB = load_3dface()
    modalA = {'imgs': imgsA, 'class_idx': class_idxA}
    modalB = {'imgs': imgsB, 'class_idx': class_idxB}
    a_idx, b_idx, labels = match_label(modalA, modalB)
    return modalA['imgs'], modalB['imgs'], a_idx, b_idx, labels



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    make_dataset_fixed('./data', 'digit', 'training.pt', 'test.pt')