# TrajMIM Training Script

## Overview
This script is designed for training the TrajMIM model on trajectory datasets. The TrajMIM model is intended for metric learning tasks involving trajectory data, capable of learning embeddings that can be used for trajectory similarity assessment, clustering, or classification.

## Dependencies
Python 3.x
PyTorch
Numpy
Visdom (optional for visualization)
Other dependencies as required by the grid2vec and TrajMIM modules, as well as utility scripts provided (utils.py, etc.).

## Installation
* Ensure you have Python 3.x installed on your system. It's recommended to use a virtual environment:
python3 -m venv trajmim-env
source trajmim-env/bin/activate

* Install the required Python packages:
pip install torch numpy visdom argparse

## Running the Script
To train the TrajMIM model, you need to prepare your trajectory dataset in the expected format and possibly adjust parameters according to your dataset characteristics.

* Prepare your trajectory dataset in the required format, ensuring you have training and validation sets ready.
* Adjust parameters in the parameters.py script or prepare to override them via command-line arguments when running the training script.
* Run the script with the desired parameters. For example:
python train.py --batch_size 32 --epoch_num 100 --learning_rate 0.001 --device cuda

## Command-Line Arguments

--model: Model to train. Currently supports TrajMIM.
--batch_size: Size of batches for training.
--epoch_num: Number of epochs for training.
--learning_rate: Learning rate for the optimizer.
(Other arguments as described in the script's argparse setup.)
