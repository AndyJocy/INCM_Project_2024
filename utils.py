import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dset

# Directory containing the data.
root = '/kaggle/input/cifar10-python'


class Occlude(object):
    def __init__(self, drop_rate=0.0, tile_size=7):
        self.drop_rate = drop_rate
        self.tile_size = tile_size

    def __call__(self, imgs, d=0):
        imgs_n = imgs.clone()
        # print(self.drop_rate, self.tile_size)
        if d == 0:
            device = 'cpu'
        else:
            device = imgs.get_device()
            if device == -1:
                device = 'cpu'
        mask = torch.ones((imgs_n.size(d), imgs_n.size(
            d+1), imgs_n.size(d+2)), device=device)  # only ones = no mask
        # print("MASK = ", mask.size())
        i = 0
        while i < imgs_n.size(d+1):
            j = 0
            while j < imgs_n.size(d+2):
                if np.random.rand() < self.drop_rate:
                    for k in range(mask.size(0)):
                        # set to zero the whole tile
                        mask[k, i:i + self.tile_size, j:j + self.tile_size] = 0
                j += self.tile_size
            i += self.tile_size

        imgs_n = imgs_n * mask  # apply the mask to each image
        return imgs_n


def get_data(params):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.
    """
    # Data proprecessing.
    transform = transforms.Compose([
        transforms.Resize(params['A']),
        transforms.ToTensor()])

    # Create the dataset.
    dataset = dset.CIFAR10(root=root, train=True,
                           download=True, transform=transform)

    # Create the dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=params['batch_size'],
                                             shuffle=True)

    return dataloader

def calculate_kl_loss(model_encoder):
    """
    Calculates the KL divergence loss.
    """

    m = torch.mean(model_encoder, dim=0)
    s = torch.std(model_encoder, dim=0)

    kl_loss = torch.mean((s ** 2 + m ** 2) / 2 - torch.log(s) - 1/2)

    return kl_loss

class encoder_layer(torch.nn.Module):
    def __init__(self, read_N, channel, dec_size, enc_size):
        super().__init__()
        self.read_N = read_N
        self.channel = channel
        self.dec_size = dec_size
        self.enc_size = enc_size
        
        self.encoderception = nn.LSTMCell(
            2*self.read_N*self.read_N*self.channel + self.dec_size, self.enc_size)
        
        self.discriminator = nn.Linear(self.enc_size, 1)