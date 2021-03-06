# Dataloader code taken from https://github.com/HHTseng/video-classification/blob/master/ResNetCRNN/UCF101_ResNetCRNN.py
import os
import pickle

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from functions import Dataset_CRNN, labels2cat
from parse_config import ConfigParser

# training parameters
res_size = 224  # ResNet image size
begin_frame, end_frame, skip_frame = 1, 29, 1


def get_dataloaders(config, path):
    data_path = config["data_path"] + "/" + path
    # Select which frame to begin & end in videos
    params = (
        {
            "batch_size": config["dataloader"]["batch_size"],
            "shuffle": config["dataloader"]["shuffle"],
            "num_workers": config["dataloader"]["workers"],
            "pin_memory": config["dataloader"]["pin_memory"],
        }
        if config["use_cuda"]
        else {}
    )

    # load UCF101 actions names
    with open(config["pickle_locations"]["action_names"], "rb") as f:
        action_names = pickle.load(f)

    # convert labels -> category
    le = LabelEncoder()
    le.fit(action_names)

    # convert category -> 1-hot
    action_category = le.transform(action_names).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(action_category)

    actions = []
    fnames = os.listdir(data_path)

    all_names = []
    for f in fnames:
        loc1 = f.find("v_")
        loc2 = f.find("_g")
        actions.append(f[(loc1 + 2) : loc2])

        all_names.append(f)

    # list all data files
    all_X_list = all_names  # all video file names
    all_y_list = labels2cat(le, actions)  # all video labels

    transform = transforms.Compose(
        [
            transforms.Resize([res_size, res_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

    dataset = Dataset_CRNN(
        data_path, all_X_list, all_y_list, selected_frames, transform=transform
    )

    dataloader = data.DataLoader(dataset, **params)
    return dataloader
