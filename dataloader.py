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


def get_dataloaders(config):
    begin_frame, end_frame, skip_frame = 1, config["SNN"]["end_frame"], 1
    data_path = config["data_path"]
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

    # show how many classes there are
    list(le.classes_)

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
        if (f[(loc1 + 2) : loc2]) == "":
            actions.append("golf")
        else:
            actions.append(f[(loc1 + 2) : loc2])
        all_names.append(f)

    # list all data files
    all_X_list = all_names  # all video file names
    all_y_list = labels2cat(le, actions)  # all video labels

    # train, test split

    train_list, test_list, train_label, test_label = train_test_split(
        all_X_list,
        all_y_list,
        test_size=config["dataloader"]["test_percent_size"],
        random_state=config["seed"],
    )

    transform = transforms.Compose(
        [
            transforms.Resize([res_size, res_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

    train_set, valid_set = (
        Dataset_CRNN(
            data_path, train_list, train_label, selected_frames, transform=transform
        ),
        Dataset_CRNN(
            data_path, test_list, test_label, selected_frames, transform=transform
        ),
    )

    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)
    return train_loader, valid_loader
