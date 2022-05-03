"""
Copyright (c) 2022 Julien Posso
"""

import torch
from utils import build_histogram, pre_compute_ori_decode


class Config:
    def __init__(self, device):

        self.HPARAM_TUNING = False  # Hyperparameter tuning with optuna. If true training and eval should be false
        self.TRAINING = True
        self.EVALUATION = True  # eval on validation and real sets
        self.EVAL_SUBMIT = True  # For submission on ESA website
        self.EVAL_DISTANCE = True  # See the impact of the distance with the target spacecraft on the score

        # Seed (for reproducibility)
        self.SEED = 1001
        # Automatic Mixed Precision (FP16)
        self.USE_AMP = False
        # Number of trials for hyperparameter tuning with Optuna
        self.N_TRIALS = 20

        # Training parameters
        self.N_EPOCHS = 50

        # Data
        self.BATCH_SIZE = 32
        self.DATASET = 'SPEED'
        self.DATASET_PATH = '../../datasets/speed'
        self.IMG_SIZE = (384, 240)  # or (1920, 1200), (768, 480), (480, 300), etc...

        # Model used
        self.MODEL_NAME = 'Pytorch-Mobile-URSONet'  # 'Pytorch-Mobile-URSONet' or 'My-Mobile-URSONet'
        # If TRAINING is True save trained model to MODEL_PATH else load model from MODEL_PATH
        self.MODEL_PATH = "../models/12bins_model.pt"
        self.PRETRAINED = True  # Init parameters with backbone pretrained on ImageNet (from Pytorch repo)

        # Dropout in the orientation branch
        self.DROPOUT_RATE = 0.2

        # Orientation and position branches
        self.ORI_TYPE = 'Classification'  # 'Classification' or 'Regression'
        self.POS_TYPE = 'Regression'  # Only regression implemented

        # ORI soft classification parameters
        self.N_ORI_BINS_PER_DIM = 12
        self.ORI_SMOOTH_FACTOR = 3

        # Loss
        self.ORI_NORM_DISTANCE = True  # Used only if ORI_TYPE == Regression
        self.POS_NORM_DISTANCE = True
        self.BETA = 1  # Loss = BETA * ORI_LOSS + POS_LOSS

        # Optimizer
        self.OPTIMIZER = 'SGD'  # 'SGD', 'Adam'
        self.LEARNING_RATE = 0.01
        self.MOMENTUM = 0.9
        self.WEIGHT_DECAY = 0

        # Scheduler
        self.SCHEDULER = 'MultiStepLR'  # 'MultiStepLR', 'OnPlateau'
        self.MILESTONES = [30, 45]  # When MultiStepLR, multiply learning rate by GAMMA at every milestone
        self.GAMMA = 0.1
        self.PATIENCE = 3  # When SCHEDULER == OnPlateau

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

        # DO NOT MODIFY THE FOLLOWING
        self.DEVICE = device  # CPU or CUDA

        # Pre-compute histogram to save time during execution
        self.H_MAP, self.REDUNDANT_FLAGS = build_histogram(self.N_ORI_BINS_PER_DIM,
                                                           torch.tensor([-180, -90, -180]),
                                                           torch.tensor([180, 90, 180]))

        self.B = pre_compute_ori_decode(self.H_MAP)
