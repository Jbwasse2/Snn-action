import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from dataloader import get_classes, create_dataloaders, get_image_size
from network import build_CNN, build_SNN
from utils import get_args, setup
args = get_args()
device, kwargs = setup(args)
#Step 1 - Train CNN
SNN = build_SNN(get_image_size(), args)
CNN = build_CNN()

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    #Train_loader gets video, audio, label
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        data = data.squeeze()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


CNN = CNN.to(device)
train_dataloader, test_dataloader = create_dataloaders(args.batch_size)
optimizer = optim.Adadelta(CNN.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

for epoch in range(1, args.epochs + 1):
    train(args, CNN, device, train_dataloader, optimizer, epoch)
    scheduler.step()

if args.save_model:
    torch.save(model.state_dict(), "mnist_cnn.pt")

#Step 2 - Remove FC layer of CNN

#Step 3 - Attach LMU to CNN

#Step 4 - Train CNN + LMU

#Step 5 - Test
