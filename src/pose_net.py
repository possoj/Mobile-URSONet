"""
Copyright (c) 2022 Julien Posso
"""

import torch
import copy
import sys
import numpy as np
from tqdm import tqdm
from utils import AverageMeter, get_score, decode_ori_batch
from data import prepare_speed_dataset
from losses import ORIREGLoss, POSREGLoss, ORICLASSLoss
from mobile_ursonet_pytorch import import_pytorch_mobile_ursonet
from my_mobile_ursonet import import_my_mobile_ursonet


class POSENet:

    def __init__(self, config):

        self.config = config
        self.model = self.import_model()
        self.dataloader = self.import_dataset()
        self.ori_criterion, self.pos_criterion = self.set_loss()
        self.optimizer = self.set_optimizer()
        self.scheduler = self.set_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.USE_AMP)
        self.hparam_step = 0

    def set_loss(self):
        if self.config.ORI_TYPE == 'Regression':
            ori_criterion = ORIREGLoss(norm_distance=self.config.ORI_NORM_DISTANCE)
        elif self.config.ORI_TYPE == 'Classification':
            ori_criterion = ORICLASSLoss()
        else:
            raise ValueError('orientation estimation type has to be either \'Regression\' or \'Classification\'')

        if self.config.POS_TYPE == 'Regression':
            pos_criterion = POSREGLoss(norm_distance=self.config.ORI_NORM_DISTANCE)
        else:
            raise ValueError('position estimation type must be \'Classification\' (Classification not implemented)')

        return ori_criterion, pos_criterion

    def set_optimizer(self):
        if self.config.OPTIMIZER == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.LEARNING_RATE,
                                        momentum=self.config.MOMENTUM, weight_decay=self.config.WEIGHT_DECAY)
        elif self.config.OPTIMIZER == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE,
                                         weight_decay=self.config.WEIGHT_DECAY)
        else:
            raise ValueError('Optimizer has to be either \'SGD\', or \'Adam\'')
        return optimizer

    def set_scheduler(self):
        if self.config.SCHEDULER == 'OnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.config.GAMMA,
                                                                   patience=self.config.PATIENCE, verbose=True)
        elif self.config.SCHEDULER == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.config.MILESTONES,
                                                             gamma=self.config.GAMMA, verbose=True)
        else:
            raise ValueError('Scheduler has to be either \'OnPlateau\' or \'MultiStepLR\'')
        return scheduler

    def import_dataset(self):
        """Import the dataset. May take some seconds as we pre-compute the histogram to save time later"""
        print('Import dataset...')
        if self.config.DATASET == 'SPEED':
            dataloader = prepare_speed_dataset(self.config)
        else:
            raise ValueError('Dataset must be \'SPEED\' (URSO dataset not implemented)')
        return dataloader

    def import_model(self):

        if self.config.MODEL_NAME == 'Pytorch-Mobile-URSONet':
            model = import_pytorch_mobile_ursonet(self.config.DROPOUT_RATE,
                                                  self.config.ORI_TYPE,
                                                  self.config.N_ORI_BINS_PER_DIM)
        elif self.config.MODEL_NAME == 'My-Mobile-URSONet':
            model = import_my_mobile_ursonet(self.config.DROPOUT_RATE,
                                             self.config.ORI_TYPE,
                                             self.config.N_ORI_BINS_PER_DIM)
        else:
            raise ValueError("Error in model name")

        model = model.to(self.config.DEVICE)

        return model

    def get_model(self):
        return self.model

    def get_n_params(self):
        """Return the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_data(self):
        return self.dataloader

    def predict(self, inputs, phase):
        with torch.set_grad_enabled(phase == 'train'):
            ori, pos = self.model(inputs)
            if self.config.ORI_TYPE == 'Regression':
                ori = torch.nn.functional.normalize(ori, p=2, dim=1)
            else:
                ori = torch.nn.functional.softmax(ori, dim=1)
        return ori, pos

    def compute_loss(self, ori, pos, targets):
        if self.config.ORI_TYPE == 'Regression':
            ori_loss = self.ori_criterion(ori, targets['ori'], targets['pos'])
        else:
            ori_loss = self.ori_criterion(ori, targets['ori'])

        pos_loss = self.pos_criterion(pos, targets['pos'])
        loss = self.config.BETA * ori_loss + pos_loss
        return loss

    def train(self):
        best_loss = 1e6
        best_model = copy.deepcopy(self.model.state_dict())
        best_epoch = 1

        # Record loss during training
        rec_loss = {'train': [], 'valid': [], 'real': []}
        # Record score during training
        rec_score = {'train': {'ori': [], 'pos': [], 'esa': []},
                     'valid': {'ori': [], 'pos': [], 'esa': []},
                     'real': {'ori': [], 'pos': [], 'esa': []}}

        # Epoch loop
        for epoch in range(self.config.N_EPOCHS):
            for phase in ['train', 'real', 'valid']:
                running_loss, running_score = AverageMeter(), AverageMeter()
                running_ori_error, running_pos_error = AverageMeter(), AverageMeter()
                running_ori_score, running_pos_score = AverageMeter(), AverageMeter()

                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluation mode

                # batch loop
                loop = tqdm(self.dataloader[phase], desc=f"Epoch {epoch+1} - {phase}",
                            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', ncols=150, file=sys.stdout)

                for inputs, targets in loop:

                    # Send inputs and targets to GPU memory if device is CUDA
                    inputs = inputs.to(self.config.DEVICE)
                    targets['ori'], targets['pos'] = targets['ori'].to(self.config.DEVICE), \
                                                     targets['pos'].to(self.config.DEVICE)

                    # Runs the forward pass under autocast
                    with torch.cuda.amp.autocast(enabled=self.config.USE_AMP):
                        # forward: predict output
                        ori, pos = self.predict(inputs, phase)

                        # Compute loss
                        loss = self.compute_loss(ori, pos, targets)

                    # Runs the backward pass
                    if phase == 'train':
                        # Init gradient
                        self.optimizer.zero_grad()
                        # Backpropagation
                        self.scaler.scale(loss).backward()
                        # Update parameters
                        self.scaler.step(self.optimizer)
                        # Updates the scale for next iteration.
                        self.scaler.update()

                    # Compute score
                    if self.config.ORI_TYPE == 'Classification':
                        ori, _ = decode_ori_batch(ori, self.config.B)

                    ori_error, ori_error_deg, pos_error, pos_error_norm, esa_score = \
                        get_score(targets, ori, pos, self.config.ORI_TYPE)

                    # Update evaluation metrics
                    running_loss.update(loss.item(), inputs.size(0))
                    running_score.update(esa_score, inputs.size(0))
                    running_ori_error.update(ori_error_deg, inputs.size(0))
                    running_pos_error.update(pos_error, inputs.size(0))
                    running_ori_score.update(ori_error, inputs.size(0))
                    running_pos_score.update(pos_error_norm, inputs.size(0))

                    # Update progress bar
                    loop.set_postfix({'loss': running_loss.get_avg(),
                                      'ori_error(deg)': running_ori_error.get_avg(),
                                      'pos_error(m)': running_pos_error.get_avg(),
                                      'esa_score': running_score.get_avg()
                                      # 'gpu_alloc (MB)': torch.cuda.memory_allocated(device=device) / 1e6,
                                      # 'gpu_reserv (MB)': torch.cuda.memory_reserved(device=device) / 1e6
                                      })

                # Store loss and score for printing
                rec_loss[phase].append(running_loss.get_avg())
                rec_score[phase]['ori'].append(running_ori_score.get_avg())
                rec_score[phase]['pos'].append(running_pos_score.get_avg())
                rec_score[phase]['esa'].append(running_score.get_avg())

                if phase == 'train':
                    # Update learning rate at the end of epoch loop (train)
                    if self.config.SCHEDULER == 'OnPlateau':
                        self.scheduler.step(running_loss.get_avg())
                    else:
                        self.scheduler.step()

                elif phase == 'valid':
                    # Selecting the best model on validation set (lower loss is better)
                    # model selection done on validation
                    if running_loss.get_avg() < best_loss:
                        print(f"New best model at epoch {epoch+1}")
                        best_model = copy.deepcopy(self.model.state_dict())
                        best_loss = running_loss.get_avg()
                        best_epoch = epoch + 1
                        print("")

        # Load best model
        self.model.load_state_dict(best_model)

        print('Best epoch:', best_epoch)

        return self.model, rec_loss, rec_score

    def evaluate(self, phase):

        self.model.eval()  # Set model to evaluation mode

        running_loss, running_score = AverageMeter(), AverageMeter()
        running_ori_error, running_pos_error = AverageMeter(), AverageMeter()
        running_ori_score, running_pos_score = AverageMeter(), AverageMeter()

        # iterate over labeled examples
        loop = tqdm(self.dataloader[phase], desc=f"Evaluation on {phase} set",
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', ncols=150, file=sys.stdout)

        for inputs, targets in loop:

            # Send inputs and targets to GPU memory if device is CUDA
            inputs = inputs.to(self.config.DEVICE)
            targets['ori'], targets['pos'] = targets['ori'].to(self.config.DEVICE), \
                                             targets['pos'].to(self.config.DEVICE)

            ori, pos = self.predict(inputs, phase)

            # Compute loss
            loss = self.compute_loss(ori, pos, targets)

            # Compute score
            if self.config.ORI_TYPE == 'Classification':
                ori, _ = decode_ori_batch(ori, self.config.B)

            ori_error, ori_error_deg, pos_error, pos_error_norm, esa_score = \
                get_score(targets, ori, pos, self.config.ORI_TYPE)

            # Update evaluation metrics
            running_loss.update(loss.item(), inputs.size(0))
            running_score.update(esa_score, inputs.size(0))
            running_ori_error.update(ori_error_deg, inputs.size(0))
            running_pos_error.update(pos_error, inputs.size(0))
            running_ori_score.update(ori_error, inputs.size(0))
            running_pos_score.update(pos_error_norm, inputs.size(0))

            # Update progress bar
            loop.set_postfix({'loss': running_loss.get_avg(),
                              'ori_error(deg)': running_ori_error.get_avg(),
                              'pos_error(m)': running_pos_error.get_avg(),
                              'esa_score': running_score.get_avg()
                              # 'gpu_alloc (MB)': torch.cuda.memory_allocated(device=device) / 1e6,
                              # 'gpu_reserv (MB)': torch.cuda.memory_reserved(device=device) / 1e6
                              })

    def evaluate_submit(self, sub):
        """Evaluation on test set for submission on ESA website"""
        for phase in ['test', 'real_test']:

            loop = tqdm(self.dataloader[phase], desc="Evaluation for submission",
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', file=sys.stdout)

            for inputs, filenames in loop:

                # Send inputs to GPU memory if device is CUDA
                inputs = inputs.to(self.config.DEVICE)

                # Runs the forward pass under autocast
                with torch.cuda.amp.autocast(enabled=self.config.USE_AMP):
                    # forward: predict output
                    ori, pos = self.predict(inputs, phase)

                if self.config.ORI_TYPE == 'Classification':
                    ori, _ = decode_ori_batch(ori, self.config.B)

                append = sub.append_real_test if phase == 'real_test' else sub.append_test
                for filename, q, r in zip(filenames, ori.cpu().numpy(), pos.cpu().numpy()):
                    append(filename, q, r)

    def eval_error_distance(self):
        """Evaluation on validation set. Distance with the target spacecraft is also returned for each prediction"""
        phase = 'valid'

        loop = tqdm(self.dataloader[phase], desc="Evaluation by distance", file=sys.stdout,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        ori_error = []
        pos_error = []
        distance = []
        for inputs, targets in loop:

            # Send inputs and targets tu GPU memory if device is CUDA                    
            inputs = inputs.to(self.config.DEVICE)
            targets['ori'], targets['pos'] = targets['ori'].to(self.config.DEVICE), \
                                             targets['pos'].to(self.config.DEVICE)

            with torch.set_grad_enabled(False):

                ori, pos = self.predict(inputs, phase)

            if self.config.ORI_TYPE == 'Classification':
                ori, _ = decode_ori_batch(ori, self.config.B)

            dist = torch.linalg.norm(targets['pos'], dim=1)

            pos_err = torch.linalg.norm(targets['pos'] - pos, dim=1)

            if self.config.ORI_TYPE == 'Regression':
                inter_sum = torch.abs(torch.sum(ori * targets['ori'], dim=1))
            else:
                inter_sum = torch.abs(torch.sum(ori * targets['ori_original'], dim=1))

            # Scaling down intermediate sum. See get_score function for more details
            if True in inter_sum[inter_sum > 1.01]:
                raise ValueError('Error computing orientation score intermediate sum')
            inter_sum[inter_sum > 1] = 1

            ori_err = 2 * torch.arccos(inter_sum) * 180 / np.pi

            distance.extend(dist.tolist())
            ori_error.extend(ori_err.tolist())
            pos_error.extend(pos_err.tolist())

        return ori_error, pos_error, distance

    def objective(self, trial):
        """This is an objective function for hyperparameter tuning with Optuna"""

        self.hparam_step += 1
        # Uncomment the following to add hyperparameters:
        # lr = trial.suggest_uniform("lr", 1e-5, 1e-1)
        # self.config.ROT_PROBABILITY = trial.suggest_float("ROT_PROBABILITY", 0, 1, step=0.1)
        # self.config.ROT_MAX_MAGNITUDE = trial.suggest_int("ROT_MAX_MAGNITUDE", 0, 170, step=10)
        # self.config.KERNEL_SIZE = trial.suggest_int("KERNEL_SIZE", 3, 7, step=2)
        # self.config.BRIGHTNESS = trial.suggest_float("BRIGHTNESS", 0, 0.5, step=0.1)
        # self.config.CONTRAST = trial.suggest_float("CONTRAST", 0, 0.5, step=0.1)
        # self.config.SATURATION = trial.suggest_float("SATURATION", 0, 0.5, step=0.1)
        # self.config.HUE = trial.suggest_float("HUE", 0, 0.5, step=0.1)
        # self.config.DROPOUT_RATE = trial.suggest_float("DROPOUT_RATE", 0, 0.5, step=0.1)
        self.config.WEIGHT_DECAY = trial.suggest_float("WEIGHT_DECAY", 0, 1e-2, step=1e-5)

        self.model = self.import_model()
        self.dataloader = self.import_dataset()
        self.ori_criterion, self.pos_criterion = self.set_loss()
        self.optimizer = self.set_optimizer()
        self.scheduler = self.set_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.USE_AMP)

        model, loss, score = self.train()

        # Score
        idx = score['valid']['esa'].index(min(score['valid']['esa']))
        # CSV: intermediate_value 0 column
        trial.report(score['real']['esa'][idx], step=0)
        # csv: value column
        score_to_optimize = score['valid']['esa'][idx]
        return score_to_optimize
