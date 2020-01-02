import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle


data_path = "/home/justin/data/UCF101/preprocessed/jpegs_256/"
# training parameters
k = 101             # number of target category
epochs = 120        # training epochs
batch_size = 40  
learning_rate = 1e-3
log_interval = 10   # interval for displaying training info
res_size = 224        # ResNet image size

# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 29, 1
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# load UCF101 actions names
with open(action_name_path, 'rb') as f:
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

# # example
# y = ['HorseRace', 'YoYo', 'WalkingWithDog']
# y_onehot = labels2onehot(enc, le, y)
# y2 = onehot2labels(le, y_onehot)

actions = []
fnames = os.listdir(data_path)

all_names = []
for f in fnames:
    loc1 = f.find('v_')
    loc2 = f.find('_g')
    actions.append(f[(loc1 + 2): loc2])

    all_names.append(f)


# list all data files
all_X_list = all_names                  # all video file names
all_y_list = labels2cat(le, actions)    # all video labels

# train, test split
train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.25, random_state=42)

transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

train_set, valid_set = Dataset_CRNN(data_path, train_list, train_label, selected_frames, transform=transform), \
                       Dataset_CRNN(data_path, test_list, test_label, selected_frames, transform=transform)

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)
