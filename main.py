import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
import nengo_dl
import pickle
import tensorflow as tf

from dataloader import valid_loader, train_loader 
from network import build_SNN
from utils import get_args, setup, dataloader_to_np_array
from logger import Logger
from functions import ResCNNEncoder,DecoderRNN
logger = Logger('/home/justin/data/logs/action/')
args = get_args()
device, kwargs = setup(args)
#Step 1 - Get Pre-trained CNN
# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.0       # dropout probability

# training parameters
k = 101             # number of target category
epochs = 120        # training epochs
batch_size = 40
learning_rate = 1e-3
log_interval = 10   # interval for displaying training info

CNN = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
CNN_PATH = "/home/justin/data/models/action/pretrained/ResNetCRNN/cnn_encoder_epoch63_singleGPU.pth"
CNN.load_state_dict(torch.load(CNN_PATH))
CNN = CNN.to(device)
CNN.eval()

#Step 2 - Attach LMU to CNN

# Nengo expects data in the form of a giant numpy array of data.
#train_data, train_labels = dataloader_to_np_array(CNN, device, train_loader)
#test_data, test_labels = dataloader_to_np_array(CNN, device, valid_loader)
#Load data in
def load_pickle(filename):
    with open(filename, 'rb') as f:
        var_you_want_to_load_into = pickle.load(f)
    return var_you_want_to_load_into

train_data = load_pickle("./train_data.pickle")
test_data = load_pickle("./test_data.pickle")
train_labels = load_pickle("./train_labels.pickle")
test_labels= load_pickle("./test_labels.pickle")
SNN = build_SNN(train_data.shape, args)

with nengo_dl.Simulator(
        SNN, minibatch_size=100, unroll_simulation=14) as sim:
    sim.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.optimizers.Adam(),
        metrics=["accuracy"],
    )

    print(
        "Initial test accuracy: %.2f%%"
        % (sim.evaluate(train_data, train_labels, verbose=1)["probe_accuracy"] * 100)
    )
#Step 3 - Train CNN + LMU
    sim.fit(train_data, train_labels, epochs=10)
    # save the parameters to file
    sim.save_params("./mnist_params")
    #Step 4 - Test
    print(
        "test accuracy: %.2f%%"
        % (sim.evaluate(test_data, test_labels, verbose=1)["probe_accuracy"] * 100)
    )
