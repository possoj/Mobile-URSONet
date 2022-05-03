"""
Copyright (c) 2022 Julien Posso
"""

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from utils import encode_ori, rotate_image, Camera
from PIL import Image
import numpy as np
import pandas as pd
import torchvision
import json
import os
import random


def copy_speed_dataset_resize(old_path, new_path, new_size=(224, 224), split='train'):
    """copy and resize Speed images to a new directory. The new (empty) folders must be created before calling
    this function"""

    if split not in {'train', 'test', 'real_test'}:
        raise ValueError('Invalid split, has to be either \'train\', \'test\' or \'real_test\'')

    with open(os.path.join(old_path, split + '.json'), 'r') as f:
        target_list = json.load(f)

    sample_ids = [label['filename'] for label in target_list]

    image_root = os.path.join(old_path, 'images', split)

    image_resize_root = os.path.join(new_path, 'images', split)

    for i in range(len(sample_ids)):
        img_name = os.path.join(image_root, sample_ids[i])
        pil_img = Image.open(img_name).resize(new_size)
        new_name = os.path.join(image_resize_root, sample_ids[i])
        pil_img.save(new_name)


class Speed(Dataset):
    """ SPEED dataset that can be used with DataLoader for PyTorch training. """

    def __init__(self, config, split='train', transform=None):

        if split not in {'train', 'test', 'real', 'real_test'}:
            raise ValueError('Invalid split, has to be either \'train\', \'test\', \'real\' or \'real_test\'')

        self.rot_augment = False
        self.camera = Camera

        with open(os.path.join(config.DATASET_PATH, split + '.json'), 'r') as f:
            target_list = json.load(f)

        self.config = config

        self.sample_ids = [target['filename'] for target in target_list]

        # Ground truth targets only available on train and real sets
        self.gt_targets_available = (split == 'train' or split == 'real')

        targets = {}
        if self.gt_targets_available:
            targets = {target['filename']: {'q': target['q_vbs2tango'], 'r': target['r_Vo2To_vbs_true']}
                       for target in target_list}

        self.image_root = os.path.join(config.DATASET_PATH, 'images', split)

        self.transform = transform

        n_examples = self.__len__()

        self.targets = {'ori': torch.zeros(n_examples, 4),
                        'pos': torch.zeros(n_examples, 3)}

        if self.gt_targets_available:
            for idx in range(len(self.sample_ids)):
                self.targets['ori'][idx] = torch.tensor(targets[self.sample_ids[idx]]['q'])
                self.targets['pos'][idx] = torch.tensor(targets[self.sample_ids[idx]]['r'])

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        # Method called to load 1 image while iterating through a dataloader. Called N times to load N images.
        sample_id = self.sample_ids[idx]
        img_name = os.path.join(self.image_root, sample_id)

        # note: despite grayscale images, we are converting to 3 channels here,
        # since most pre-trained networks expect 3 channel input
        pil_image = Image.open(img_name).convert('RGB')

        # If train or real dataset (where true Pose is available in the dataset)
        if self.gt_targets_available:
            y = {}
            ori, pos = self.targets['ori'][idx], self.targets['pos'][idx]

            if self.rot_augment:
                # Image rotation
                dice = np.random.rand(1)
                if dice >= self.config.ROT_PROBABILITY:
                    pil_image, ori, pos = rotate_image(pil_image, ori, pos, self.camera.K,
                                                       self.config.ROT_MAX_MAGNITUDE)

            if self.config.ORI_TYPE == 'Regression':
                y['ori'] = self.targets['ori'][idx]
            else:
                y['ori'] = encode_ori(ori, self.config.H_MAP, self.config.REDUNDANT_FLAGS,
                                      self.config.ORI_SMOOTH_FACTOR, self.config.N_ORI_BINS_PER_DIM)
                y['ori_original'] = ori

            if self.config.POS_TYPE == 'Regression':
                y['pos'] = pos
            else:
                raise ValueError('Classification for position estimation is not yet implemented')

        else:
            y = sample_id

        if self.transform is not None:
            torch_image = self.transform(pil_image)
        else:
            torch_image = pil_image

        return torch_image, y


class AddGaussianNoise(object):
    def __init__(self, generator, mean=0., std=1.):
        self.std = std
        self.mean = mean
        self.generator = generator

    def __call__(self, tensor):
        return torch.abs(tensor + torch.randn(tensor.size()) * self.std + self.mean)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_speed_dataset(config):
    # Reproducibility. See https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    g = torch.Generator()
    g.manual_seed(config.SEED)

    # Processing to match pre-trained networks
    data_transforms = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = Speed(config, 'train', data_transforms)

    train_set, val_set = torch.utils.data.random_split(full_dataset,
                                                       [int(len(full_dataset) * .85), int(len(full_dataset) * .15)],
                                                       generator=g)

    # set transforms on training set (data augmentation)
    train_set.dataset.transform = torchvision.transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        # torchvision.transforms.ToPILImage(),
        # transforms.RandomErasing(p=0.5, scale=(0.1, 0.1)),
        # GaussianBlur seems to decrease position error from 4 to 0.9 meters on real test set
        torchvision.transforms.GaussianBlur(kernel_size=(config.KERNEL_SIZE, config.KERNEL_SIZE),
                                            sigma=(config.SIGMA_MIN, config.SIGMA_MAX)),
        # torchvision.transforms.RandomCrop((32, 32), padding=),
        # torchvision.transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        # AddGaussianNoise(mean=0., std=0.02, generator=g),
        torchvision.transforms.ColorJitter(brightness=config.BRIGHTNESS, contrast=config.CONTRAST,
                                           saturation=config.SATURATION, hue=config.HUE),
        # torchvision.transforms.RandomRotation(7),
        # torchvision.transforms.RandomAffine(10)
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Manual rotations (data augmentation)
    if config.ROT_IMAGE_AUG:
        train_set.dataset.rot_augment = True

    test_set = Speed(config, 'test', data_transforms)
    real_test_set = Speed(config, 'real_test', data_transforms)
    real_set = Speed(config, 'real', data_transforms)

    datasets = {'train': train_set, 'valid': val_set, 'test': test_set, 'real_test': real_test_set, 'real': real_set}

    dataloader = {x: DataLoader(datasets[x], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=16,
                                worker_init_fn=seed_worker, generator=g)
                  for x in ['train', 'valid', 'test', 'real_test', 'real']}

    return dataloader
