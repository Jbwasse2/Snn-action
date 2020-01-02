import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

from dataloader import valid_loader, train_loader
from network import build_SNN
from utils import get_args, setup
from logger import Logger
from functions import ResCNNEncoder,DecoderRNN
logger = Logger('/home/justin/data/logs/action/')
args = get_args()
device, kwargs = setup(args)
#Step 1 - Get Pre-trained CNN
#SNN = build_SNN(get_image_size(), args)
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

#Step 3 - Train CNN + LMU

#Step 4 - Test
