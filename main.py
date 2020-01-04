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

from dataloader import train_loader, valid_loader
from functions import DecoderRNN, ResCNNEncoder
from network import build_SNN
from parse_config import ConfigParser
from utils import dataloader_to_np_array, setup

args = argparse.ArgumentParser(description="Action Recognition")
args.add_argument(
    "-c",
    "--config",
    default="./config.json",
    type=str,
    help="config file path (default: ./config.json)",
)

config = ConfigParser.from_args(args)
# Step 1 - Get Pre-trained CNN
# EncoderCNN architecture - Don't change architecture, it wont work.
CNN = ResCNNEncoder().to(device)
CNN.load_state_dict(torch.load(config["pickle_locations"]["CNN_weights"]))
CNN = CNN.to(device)
CNN.eval()

# Step 2 - Attach LMU to CNN
if config["SNN_trainer"]["do_CNN_forward_data"]:
    # Forward pass data through CNN
    # Nengo expects data in the form of a giant numpy array of data.
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

SNN = build_SNN(train_data.shape, args)

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

    print(
        "Initial test accuracy: %.2f%%"
        % (sim.evaluate(test_data, test_labels, verbose=1)["probe_accuracy"] * 100)
    )
    # Step 3 - Train CNN + LMU
    if config["SNN_trainer"]["do_SNN_training"]
        sim.fit(train_data, train_labels, epochs=config["SNN_trainer"]["epochs"])
        # save the parameters to file
        sim.save_params(config["pickle_locations"]["SNN_weights"])
    else:
        sim.load_params(config["pickle_locations"]["SNN_weights"])

    # Step 4 - Test
    print(
        "test accuracy: %.2f%%"
        % (sim.evaluate(test_data, test_labels, verbose=1)["probe_accuracy"] * 100)
    )
