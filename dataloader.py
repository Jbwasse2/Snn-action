# Dataloader code taken from https://github.com/HHTseng/video-classification/blob/master/ResNetCRNN/UCF101_ResNetCRNN.py
import os
import pickle

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from functions import Dataset_CRNN, labels2cat
from parse_config import ConfigParser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

data_path = "/home/justin/data/UCF101/preprocessed/jpegs_256/data/jpegs_256/"
# training parameters
k = 101  # number of target category
batch_size = 40
res_size = 224  # ResNet image size

# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 29, 1
use_cuda = not args.no_cuda and torch.cuda.is_available()
params = (
    {"batch_size": batch_size, "shuffle": True, "num_workers": 0, "pin_memory": True}
    if use_cuda
    else {}
)

# load UCF101 actions names
action_name_path = "./var_data/UCF101actions.pkl"
with open(action_name_path, "rb") as f:
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
    actions.append(f[(loc1 + 2) : loc2])

    all_names.append(f)


# list all data files
all_X_list = all_names  # all video file names
all_y_list = labels2cat(le, actions)  # all video labels

# train, test split
train_list, test_list, train_label, test_label = train_test_split(
    all_X_list, all_y_list, test_size=0.25, random_state=42
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
