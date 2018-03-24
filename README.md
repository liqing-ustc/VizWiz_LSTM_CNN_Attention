## Usage

#### Prerequisites

Make sure you are on a machine with a NVIDIA GPU and Python 2 with about 70 GB disk space.

1. Install [Tensorflow](http://pytorch.org/) with CUDA and Python 2.7.

#### Data Setup

All data should be downloaded to a data/ directory in the root directory of this repository.

#### Training

Simply run `python main.py` to start training. The training and validation scores will be printed
every epoch, and the best model will be saved under the directory "saved_models". The default flags should give you the result provided in the table above.