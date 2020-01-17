import argparse
import logging
import pickle
from configparser import ConfigParser

import nengo_dl
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from dataloader import get_dataloaders
from functions import DecoderRNN, ResCNNEncoder
from network import build_SNN
from parse_config import ConfigParser
from utils import dataloader_to_np_array, setup

logger = logging.getLogger(__name__)


args = argparse.ArgumentParser(description="Action Recognition")
args.add_argument(
    "-c",
    "--config",
    default="./configs/config.json",
    type=str,
    help="config file path (default: ./configs/config.json)",
)
args.add_argument(
    "-d",
    "--device",
    default=None,
    type=str,
    help="indices of GPUs to enable (default: all)",
)

config = ConfigParser.from_args(args)
device = setup(config)
# Step 1 - Get Pre-trained CNN
# This is only needed if any of the config settings actually require this
# Because this was trained on a gpu, this seems to only work with a gpu also
# EncoderCNN architecture - Don't change architecture, it wont work.


class HardwareError(Exception):
    """Exception raised when attempting to run this software on a machine without gpu enabled"""

    def __init__(self, message):
        self.message = message


if not config["use_cuda"] and not config["SNN_trainer"]["import_CNN_forward_data"]:
    raise HardwareError(
        "Loading of the CNN is supported only for GPU, disable use of the CNN"
    )


if config["use_cuda"]:
    CNN = ResCNNEncoder().to(device)
    CNN.load_state_dict(torch.load(config["pickle_locations"]["CNN_weights"]))
    CNN = CNN.to(device)
    CNN.eval()


# Step 2 - Attach LMU to CNN
if not config["SNN_trainer"]["import_CNN_forward_data"]:
    # Forward pass data through CNN
    # Nengo expects data in the form of a giant numpy array of data.
    train_loader, valid_loader = get_dataloaders(config)
    train_data, train_labels = dataloader_to_np_array(CNN, device, train_loader)
    test_data, test_labels = dataloader_to_np_array(CNN, device, valid_loader)
else:
    # Load data in that was outputted from CNN (This is fast)
    def load_pickle(filename):
        with open(filename, "rb") as f:
            var_you_want_to_load_into = pickle.load(f)
        return var_you_want_to_load_into

    train_data = load_pickle(config["pickle_locations"]["train_data"])
    test_data = load_pickle(config["pickle_locations"]["test_data"])
    train_labels = load_pickle(config["pickle_locations"]["train_labels"])
    test_labels = load_pickle(config["pickle_locations"]["test_labels"])

SNN = build_SNN(train_data.shape, config)

with nengo_dl.Simulator(
    SNN,
    minibatch_size=config["SNN"]["minibatch_size"],
    unroll_simulation=config["SNN"]["unroll_simulation"],
) as sim:
    sim.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.optimizers.Adam(),
        metrics=["accuracy"],
    )

    if config["SNN_trainer"]["get_initial_testing_accuracy"]:
        logger.info(
            "Initial test accuracy: %.2f%%"
            % (sim.evaluate(test_data, test_labels, verbose=1)["probe_accuracy"] * 100)
        )
    # Step 3 - Train CNN + LMU
    if config["SNN_trainer"]["do_SNN_training"]:
        history = sim.fit(
            train_data, train_labels, epochs=config["SNN_trainer"]["epochs"]
        )
        logger.info("training parameters")
        logger.info(history.params)
        logger.info("training results")
        logger.info(history.history)
        # save the parameters to file
        sim.save_params(config["pickle_locations"]["SNN_weights"])
    else:
        sim.load_params(config["pickle_locations"]["SNN_weights"])

    # Step 4 - Test
    logger.info(
        "test accuracy: %.2f%%"
        % (sim.evaluate(test_data, test_labels, verbose=1)["probe_accuracy"] * 100)
    )
