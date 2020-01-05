import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm


def setup(config):
    use_cuda = config["use_cuda"] and torch.cuda.is_available()

    tf.random.set_seed(config["seed"])
    np.random.seed(config["seed"])
    rng = np.random.RandomState(config["seed"])
    torch.manual_seed(config["seed"])

    device = torch.device("cuda" if use_cuda else "cpu")

    return device


def dataloader_to_np_array(cnn_encoder, device, loader):
    feature_space = []
    labels = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = cnn_encoder(X)
            feature_space.extend(output.cpu().data.squeeze().numpy().tolist())
            labels.extend(y.cpu().data.squeeze().numpy().tolist())
    # Convert lists into np.arrays of shape (len(data), time, data)
    labels_np = np.array(labels)
    labels_np = np.tile(labels_np[:, None, None], (1, 28, 1))
    feature_space_np = np.array(feature_space)
    return feature_space_np, labels_np


# https://github.com/victoresque/pytorch-template/blob/master/utils/util.py
def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
