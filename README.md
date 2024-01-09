# Mobile-URSONet: an embeddable Neural Network for Spacecraft Pose Estimation 
Pytorch implementation of Mobile-URSONet: a mobile version of the popular [URSONet](https://github.com/pedropro/UrsoNet) 
that achieved the 3rd place on the [European Space Agency's Pose Estimation Challenge](https://kelvins.esa.int/satellite-pose-estimation-challenge/). 
We propose Mobile-URSONet: a spacecraft pose estimation convolutional neural network with 178 times fewer parameters 
while degrading accuracy by no more than four times compared to URSONet. 

For more details check our [Arxiv preprint](https://arxiv.org/abs/2205.02065). This work was presented at 
IEEE ISCAS 2022, a peer-reviewed conference and available on [IEEE Xplore](https://ieeexplore.ieee.org/document/9937721)

```
@INPROCEEDINGS{9937721,
  author={Posso, Julien and Bois, Guy and Savaria, Yvon},
  booktitle={2022 IEEE International Symposium on Circuits and Systems (ISCAS)}, 
  title={Mobile-URSONet: an Embeddable Neural Network for Onboard Spacecraft Pose Estimation}, 
  year={2022},
  volume={},
  number={},
  pages={794-798},
  doi={10.1109/ISCAS48785.2022.9937721}}
```

## Installation

### Data and directory structure
Mobile-URSONet is ready to use the [SPEED dataset](https://kelvins.esa.int/satellite-pose-estimation-challenge/data/) 
released by Stanford SLAB and ESA. Just download the dataset and follow the directory structure (or modify the 
config.py DATASET_PATH):

```
    ├── datasets            # Datasets folder
    │   ├── speed           # Speed dataset
    │   │   ├── images      # Speed images folder
    │   │   └──...          # JSON files, license, etc...
    │   └── others          # Potentially other Spacecraft Pose Estimation datasets
    │
    ├── mobile_ursonet      # Project root directory (Git)
    │   ├── display         # matplotlib figures saved duning training/evaluation
    │   ├── models          # weights of our models used in the article
    │   ├── optuna_tuning   # results of hyperparameter tuning with optuna
    │   ├── src             # python source files
    │   ├── submissions     # CSV files for submission on ESA website
    │   └── ...             # License, readme, etc...
```

Remark: resizing images to 960x600 pixels helps reduce the latency when images are loaded into RAM. 
See copy_speed_dataset_resize function in `data.py`

### Dependencies

This code was developed using Python 3.8.12. Python packages are listed in `requirements.txt`. To install the necessary 
python packages with conda, execute:
```
conda install --file requirements.txt -c pytorch -c conda-forge
```

## Training and evaluation
The training and evaluation of Mobile-URSONet is controlled by the config.py file. To execute the code, just modify the 
`config.py` file and run from the `src` directory: `python main.py`

Example: to reproduce the results of our 16 bins model, change the default value of `config.py to the following:

| Variable           |            Value            |
|--------------------|:---------------------------:|
| TRAINING           |            False            |
| MODEL_PATH         | "../models/16bins_model.pt" |
| N_ORI_BINS_PER_DIM |             16              |

See `config.py`for more details. 
