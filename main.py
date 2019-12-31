import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

from dataloader import get_classes, create_dataloaders, get_image_size
from network import build_CNN, build_SNN
from utils import get_args, setup
from logger import Logger
logger = Logger('/home/justin/data/logs/action/')
args = get_args()
device, kwargs = setup(args)
#Step 1 - Train CNN
SNN = build_SNN(get_image_size(), args)
CNN = build_CNN()

#https://github.com/victoresque/pytorch-template/blob/master/model/metric.py
def get_accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    accuracies = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            data = data.squeeze()
            #I have a gut feeling that the data shape could be wrong sometimes
            if len(data.shape) != 4:
                continue
            output = model(data)
            test_loss += F.cross_entropy(output, target)
            accuracies.append(get_accuracy(output, target))
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if batch_idx % args.log_interval == 0:
                step = epoch * len(test_loader) + batch_idx
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), test_loss.item()))

    accuracy = sum(accuracies)/ len(accuracies) 
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print("accuracy2",accuracy)
    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #
    #https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py

    # 1. Log scalar values (scalar summary)
    info = { 'test_loss': test_loss.item(), 'test_accuracy': accuracy }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch+1)

    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    #Train_loader gets video, audio, label
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        data = data.squeeze()
        #I have a gut feeling that the data shape could be wrong sometimes
        if len(data.shape) != 4:
            continue
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        accuracy = get_accuracy(output, target)
        if batch_idx % args.log_interval == 0:
            step = epoch * len(train_loader) + batch_idx
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #
            #https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py

            # 1. Log scalar values (scalar summary)
            info = { 'train_loss': loss.item(), 'train_accuracy': accuracy }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, step+1)

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), step+1)
                #logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), batch_idx+1)

            # 3. Log training images (image summary)
        #    info = { 'images': images.view(-1, 28, 28)[:10].cpu().numpy() }

#            for tag, images in info.items():
#                logger.image_summary(tag, images, batch_idx+1)


CNN = CNN.to(device)
train_dataloader, test_dataloader = create_dataloaders(args)
optimizer = optim.Adadelta(CNN.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

for epoch in range(1, args.epochs + 1):
    train(args, CNN, device, train_dataloader, optimizer, epoch)
    test(args, CNN, device, test_dataloader, epoch)
    scheduler.step()

if args.save_model:
    torch.save(CNN.state_dict(), "action_cnn.pt")

#Step 2 - Remove FC layer of CNN

#Step 3 - Attach LMU to CNN

#Step 4 - Train CNN + LMU

#Step 5 - Test
