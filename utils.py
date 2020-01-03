import argparse
import tensorflow as tf
import numpy as np
import torch
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='Action Recognition')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--workers', type=int, default=1, metavar='W',
                        help='number of workers to get data')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    return args

def setup(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
    return device, kwargs

def dataloader_to_np_array(cnn_encoder, device, loader):
    feature_space = []
    labels = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = cnn_encoder(X)
#            output = output.reshape(output.shape[0],28*512)
            feature_space.extend(output.cpu().data.squeeze().numpy().tolist())
            labels.extend(y.cpu().data.squeeze().numpy().tolist())
    #Convert lists into np.arrays of shape (len(data), time, data)
    labels_np = np.array(labels)
    labels_np = np.tile(labels_np[:,None,None], (1,28,1))
    feature_space_np = np.array(feature_space)
    return feature_space_np, labels_np

