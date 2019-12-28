import torchvision.datasets as datasets
import torch

import pudb; pu.db
#Get data
ucf101_trainset = datasets.UCF101(root='/home/justin/research/action_recognition/ucf_data/videos/', annotation_path='/home/justin/research/action_recognition/ucf_data/ucfTrainTestlist/', frames_per_clip=30, train=True, transform=None)
ucf101_testset = datasets.UCF101(root='/home/justin/research/action_recognition/ucf_data/videos/', annotation_path='/home/justin/research/action_recognition/ucf_data/ucfTrainTestlist/', frames_per_clip=30, train=False, transform=None)

#Very it works
def verify_dataset(dataset):
    if len(dataset.samples) == 0 or len(dataset.classes) <= 1 or len(dataset.indices) == 0:
        raise RuntimeError("Data set up incorrectly!")
verify_dataset(ucf101_testset)
verify_dataset(ucf101_trainset)

#Create data loaders
batch_size = 32


train_loader = torch.utils.data.DataLoader(
                 dataset=ucf101_trainset,
                 batch_size=batch_size,
                 shuffle=False)
train_classes = ucf101_trainset.classes

test_loader = torch.utils.data.DataLoader(
                dataset=ucf101_testset,
                batch_size=batch_size,
                shuffle=False)
test_classes = ucf101_testset.classes
import pudb; pu.db
