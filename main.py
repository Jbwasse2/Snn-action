import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

#from dataloader import get_classes, create_dataloaders, get_image_size
from network import build_CNN, build_SNN
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

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 101             # number of target category
epochs = 120        # training epochs
batch_size = 40
learning_rate = 1e-3
log_interval = 10   # interval for displaying training info
CNN = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
RNN = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)
CNN_PATH = "/home/justin/data/models/action/pretrained/ResNetCRNN/cnn_encoder_epoch63_singleGPU.pth"
RNN_PATH = "/home/justin/data/models/action/pretrained/ResNetCRNN/rnn_decoder_epoch63_singleGPU.pth"
CNN.load_state_dict(torch.load(CNN_PATH))
CNN.eval()
RNN.load_state_dict(torch.load(RNN_PATH))
RNN.eval()

#Step 2 Attach RNN to CNN and get results to test everything works
data_path = "/home/justin/data/UCF101/preprocessed/jpegs_256"
action_name_path = './UCF101actions.pkl'


#Step 2 - Remove FC layer of CNN

#Step 3 - Attach LMU to CNN

#Step 4 - Train CNN + LMU

#Step 5 - Test
