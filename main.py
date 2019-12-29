from dataloader import train_loader, train_classes, test_loader, test_classes, image_size
from network import build_CNN, build_model
#Step 1 - Train CNN
SNN = build_model(image_size)
CNN = build_CNN()

#Step 2 - Remove FC layer of CNN

#Step 3 - Attach LMU to CNN

#Step 4 - Train CNN + LMU

#Step 5 - Test
