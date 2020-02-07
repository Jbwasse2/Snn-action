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
import logging

from dataloader import get_dataloaders
from functions import DecoderRNN, ResCNNEncoder
from network import build_SNN, build_SNN_simple
from parse_config import ConfigParser
from utils import dataloader_to_np_array, setup


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
#logger = config.get_logger(__name__, 0)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
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

# To save gpu memory, only load CNN if its needed
if config["use_cuda"] and not config["SNN_trainer"]["import_CNN_forward_data"]:
    CNN = ResCNNEncoder().to(device)
    CNN.load_state_dict(torch.load(config["pickle_locations"]["CNN_weights"]))
    CNN = CNN.to(device)
    CNN.eval()


# Step 2 - Attach LMU to CNN
if not config["SNN_trainer"]["import_CNN_forward_data"]:
    # Forward pass data through CNN
    # Nengo expects data in the form of a giant numpy array of data.
    train_loader = get_dataloaders(config, "train")
    test_loader = get_dataloaders(config, "test")
    train_data, train_labels = dataloader_to_np_array(CNN, device, train_loader)
    test_data, test_labels = dataloader_to_np_array(CNN, device, test_loader)

    def save_pickle(filename, var):
        with open(filename, "wb") as f:
            pickle.dump(var, f)

    save_pickle(config["pickle_locations"]["train_data"], train_data)
    save_pickle(config["pickle_locations"]["test_data"], test_data)
    save_pickle(config["pickle_locations"]["train_labels"], train_labels)
    save_pickle(config["pickle_locations"]["test_labels"], test_labels)
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

SNN = build_SNN_simple(train_data.shape, config)

with nengo_dl.Simulator(
    SNN,
    minibatch_size=config["SNN"]["minibatch_size"],
    unroll_simulation=1,
    device="/gpu:1",
) as sim:
    sim.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.optimizers.Adam(),
        metrics=["accuracy"],
    )

    if config["SNN_trainer"]["get_initial_testing_accuracy"]:
        logger.debug(
            "Initial test accuracy: %.2f%%"
            % (sim.evaluate(test_data, test_labels, verbose=1)["probe_accuracy"] * 100)
        )
        # Step 3 - Train CNN + LMU
    test_accs = []
    train_accs = []
    if config["SNN_trainer"]["do_SNN_training"]:
        for i in range(config["SNN_trainer"]["epochs"]):
            logger.info("epoch " + str(i) )
            history = sim.fit(train_data, train_labels, epochs=1)
            logger.info("training parameters")
            logger.info(history.params)
            logger.info("training results")
            logger.info(history.history)
            train_accs.append(history.history["probe_accuracy"])
            # save the parameters to file
            test_acc = (
                sim.evaluate(test_data, test_labels, verbose=1)["probe_accuracy"] * 100
            )
            logger.info("test accuracy: %.2f%%" % (test_acc))
            test_accs.append(test_acc)
        sim.save_params(config["pickle_locations"]["SNN_weights"])
    else:
        sim.load_params(config["pickle_locations"]["SNN_weights"])
    final_test = sim.evaluate(test_data, test_labels, verbose=1)["probe_accuracy"] * 100
    # Step 4 - Test
    logger.info("final test accuracy: %.2f%%" % (final_test))
    test_accs.append(final_test)
    logger.info("Training = "+ str( train_accs))
    logger.info("Testing = " + str(test_accs))
