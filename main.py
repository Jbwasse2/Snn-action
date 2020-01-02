import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
import nengo_dl
import tensorflow as tf

from dataloader import valid_loader, train_loader, get_image_size
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
CNN.eval()

#Step 2 - Attach LMU to CNN
SNN = build_SNN(get_image_size(), args)

# Nengo expects data in the form of a giant numpy array of data.
train_data, train_labels = dataloader_to_np_array(CNN, device, train_loader)
test_data, test_labels = dataloader_to_np_array(CNN, device, valid_loader)

#Step 3 - Train CNN + LMU

with nengo_dl.Simulator(
        SNN, minibatch_size=5, unroll_simulation=14) as sim:
    sim.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.optimizers.Adam(),
        metrics=["accuracy"],
    )
#    import pudb; pu.db
    print(
        "Initial test accuracy: %.2f%%"
        % (sim.evaluate(train_data, train_labels, verbose=1)["probe_accuracy"] * 100)
    )
#Step 4 - Test
