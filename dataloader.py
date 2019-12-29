import torchvision.datasets as datasets
import torch

#Get data
data_folder= "./ucf_data/"
#data_folder="/home/justin/data/UCF101/"
ucf101_trainset = datasets.UCF101(root=data_folder + 'videos', annotation_path=data_folder+'ucfTrainTestlist/', frames_per_clip=30, train=True, transform=None, num_workers=1)
ucf101_testset = datasets.UCF101(root=data_folder + 'videos', annotation_path=data_folder+'ucfTrainTestlist/', frames_per_clip=30, train=False, transform=None, num_workers=1)


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

image_size = ucf101_trainset[0][0][0][:,:,0].shape
