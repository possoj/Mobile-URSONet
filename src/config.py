"""
Copyright (c) 2022 Julien Posso
"""

import torch
from utils import build_histogram, pre_compute_ori_decode


class Config:
    def __init__(self, device):

        self.HPARAM_TUNING = False  # Hyperparameter tuning with optuna. If true training and eval should be false
        self.TRAINING = False
        self.EVALUATION = True
        self.EVAL_SUBMIT = True  # For submission on ESA website
        self.EVAL_DISTANCE = True

        # Seed
        self.SEED = 1001
        # Automatic Mixed Precision
        self.USE_AMP = False
        # Number of trials for hyperparameter tuning with Optuna
        self.N_TRIALS = 20

        # Training parameters
        self.N_EPOCHS = 1

        # Data
        self.BATCH_SIZE = 8
        self.DATASET = 'SPEED'
        self.DATASET_PATH = '../../datasets/speed'
        self.IMG_SIZE = (384, 240)  # or (1920, 1200), (768, 480), (480, 300), etc...

        # Model used
        self.MODEL_NAME = 'Pytorch-Mobile-URSONet'  # 'Pytorch-Mobile-URSONet' 'My-Mobile-URSONet'
        self.MODEL_PATH = "../models/paper1_8_bins"
        self.PRETRAINED = True  # Init parameters of the backbone pretrained on ImageNet (from Pytorch repo)

        # ORI soft classification parameters
        self.N_ORI_BINS_PER_DIM = 8
        self.ORI_SMOOTH_FACTOR = 3

        # Loss
        self.ORI_TYPE = 'Classification'  # Or 'Regression'
        self.ORI_NORM_DISTANCE = True  # Regression only
        self.POS_TYPE = 'Regression'
        self.POS_NORM_DISTANCE = True
        self.BETA = 1  # Loss = BETA * ORI_LOSS + POS_LOSS

        # Optimizer
        self.OPTIMIZER = 'SGD'  # Or 'Adam'
        self.LEARNING_RATE = 0.01
        self.MOMENTUM = 0.9
        self.WEIGHT_DECAY = 0

        # Scheduler
        self.SCHEDULER = 'MultiStepLR'  # 'OnPlateau'
        self.MILESTONES = [30, 45]  # When MultiStepLR, multiply learning rate by GAMMA at every milestone
        self.GAMMA = 0.1
        self.PATIENCE = 3  # When OnPlateau

        # POS soft classification
        # self.N_POS_BINS_PER_DIM = 32
        # self.POS_SMOOTH_FACTOR = 3

        # Image augmentation (OpenCV)
        self.ROT_IMAGE_AUG = True
        self.ROT_PROBABILITY = 0.5
        self.ROT_MAX_MAGNITUDE = 50

        # Pytorch data augmentation parameters
        # Gaussian Blur
        self.KERNEL_SIZE = 5
        self.SIGMA_MIN = 0.1
        self.SIGMA_MAX = 2
        # Color Jitter
        self.BRIGHTNESS = 0.2
        self.CONTRAST = 0.2
        self.SATURATION = 0.2
        self.HUE = 0.2

        # Model parameters
        self.DROPOUT_RATE = 0.2

        # DO NOT MODIFY THE FOLLOWING
        self.DEVICE = device  # CPU or CUDA

        # Pre-compute histogram to save time during execution
        self.H_MAP, self.REDUNDANT_FLAGS = build_histogram(self.N_ORI_BINS_PER_DIM,
                                                           torch.tensor([-180, -90, -180]),
                                                           torch.tensor([180, 90, 180]))

        self.B = pre_compute_ori_decode(self.H_MAP)
