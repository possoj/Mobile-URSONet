"""
Copyright (c) 2022 Julien Posso
"""

import torch
import optuna
from config import Config
from pose_net import POSENet
from submission import SubmissionWriter
from print_results import print_training_loss, print_training_score, print_beta_tuning, print_error_distance
import os
import numpy as np
import random


def main():

    # SELECT DEVICE AUTOMATICALLY: if available, select the GPU with the most available memory, else select the CPU
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            # The following branch works with nvidia 470 drivers + cuda 11.4
            # Do not work with nvidia 510 + cuda 11.6: Free memory is replaced with reserved memory
            # get the GPU id with max memory available
            os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >../tmp')
            memory_available = [int(x.split()[2]) for x in open('../tmp', 'r').readlines()]
            os.remove('../tmp')  # remove the temporary file
            gpu_id = np.argmax(memory_available)
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Device used: {device} with {memory_available[gpu_id]} MB memory available")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Device used:", device)

    # Create config and Pose estimation class
    config = Config(device)
    pose_estimation = POSENet(config)

    # Set manual seeds for reproducibility. See https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    torch.manual_seed(config.SEED)
    random.seed(config.SEED)  # Python random module.
    np.random.seed(config.SEED)  # Numpy module.
    torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)
        torch.cuda.manual_seed_all(config.SEED)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    # Count number of model parameters
    pytorch_total_params = pose_estimation.get_n_params()
    print(f"Number of trainable parameters in the model :{pytorch_total_params:,}")

    # Submissions on ESA website
    sub = SubmissionWriter()

    if config.HPARAM_TUNING:
        print("hyperparameter tuning")
        sampler = optuna.samplers.TPESampler(seed=config.SEED)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(sampler=sampler, direction="minimize")
        print(f"Sampler is {study.sampler.__class__.__name__}")
        study.optimize(pose_estimation.objective, n_trials=config.N_TRIALS)
        data = study.trials_dataframe(("number", "value", "intermediate_values", "datetime_start", "datetime_complete",
                                       "duration", "params", "user_attrs", "system_attrs", "state"))
        data.to_csv('../optuna_tuning/hyperparameter_tuning_result.csv', encoding='utf-8')
        pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    if config.TRAINING:
        print("Training...")
        model, loss, score = pose_estimation.train()
        # Save model
        model.cpu()
        torch.save(model.state_dict(), config.MODEL_PATH)
        # Print training
        print_training_loss(loss, show=False, save=True)
        print_training_score(score, show=False, save=True)

    else:
        # If not training, try to load the model from config.MODEL_PATH instead of saving the model to config.MODEL_PATH
        model = pose_estimation.get_model()
        model.load_state_dict(torch.load(config.MODEL_PATH))

    # Move model to GPU if needed and prepare it for evaluation
    model.to(config.DEVICE)
    model.eval()

    if config.EVALUATION:
        print("Evaluation on valid and real set")
        pose_estimation.evaluate('valid')
        pose_estimation.evaluate('real')

    if config.EVAL_SUBMIT:
        print("Evaluation for submission...")
        pose_estimation.evaluate_submit(sub)
        sub.export(out_dir='../submissions/', suffix="pytorch")
        sub.reset()

    if config.EVAL_DISTANCE:
        print("Evaluate by  distance...")
        ori_err, pos_err, distance = pose_estimation.eval_error_distance()
        print_error_distance(ori_err, pos_err, distance, show=False, save=True)

    print("The end!")


if __name__ == '__main__':
    main()
