from torchvision import transforms
import torch
from utils import get_args
from dataset import ucf101

    

args = get_args()
#Get data
#data_folder= "./ucf_data/"
data_folder="/home/justin/data/UCF101/"
transforms =  transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
 ])
ucf101_trainset = ucf101(root=data_folder + 'videos', annotation_path=data_folder+'ucfTrainTestlist/', frames_per_clip=1, train=True, transform=transforms, num_workers=args.workers)
ucf101_testset = ucf101(root=data_folder + 'videos', annotation_path=data_folder+'ucfTrainTestlist/', frames_per_clip=1, train=False, transform=transforms, num_workers=args.workers)


#Very it works
def verify_dataset(dataset):
    if len(dataset.samples) == 0 or len(dataset.classes) <= 1 or len(dataset.indices) == 0:
        raise RuntimeError("Data set up incorrectly!")
verify_dataset(ucf101_testset)
verify_dataset(ucf101_trainset)

def get_classes():
    train_classes = ucf101_trainset.classes
    test_classes = ucf101_testset.classes
    return train_classes, test_classes

#Create data loaders
def create_dataloaders(args):
    train_loader = torch.utils.data.DataLoader(
                     dataset=ucf101_trainset,
                     batch_size=args.batch_size,
                     num_workers=args.workers,
                     shuffle=True)

    test_loader = torch.utils.data.DataLoader(
                    dataset=ucf101_testset,
                    batch_size=args.test_batch_size,
                    num_workers=args.workers,
                    shuffle=True)
    return train_loader, test_loader

def get_image_size():
    image_size = ucf101_trainset[0][0][0][0,:,:].shape
    return image_size

if __name__ == "__main__":
    import time
    from utils import get_args, setup
    args = get_args()
    device, kwargs = setup(args)
    CNN = None
    def train(args, model, device, train_loader, optimizer, epoch):
        #Train_loader gets video, audio, label
        start = time.process_time()
        for batch_idx, (data, target) in enumerate(train_loader):
            print(time.process_time() - start)
            start = time.process_time()
    train_dataloader, test_dataloader = create_dataloaders(args)
    epoch = 1
    optimizer = None
    scheduler = None 

    train(args, CNN, device, train_dataloader, optimizer, epoch)
