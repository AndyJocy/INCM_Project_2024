import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader, TensorDataset
from model import DRAWModel
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score


class Classifier(nn.Module):
    def __init__(self, model, device_param):
        super(Classifier, self).__init__()
        self.device = device_param

        self.model = model

        self.linear = nn.Linear(400, 10, device=self.device)

    def forward(self, x):
        h_enc_prev, h_dec_prev, h_enc_total = model.wake_forward(x)
        # print(h_enc_total.shape)
        # exit()
        return self.linear(h_enc_total)


# Load the classifier model.
state_dict_classifier = torch.load('classifier_pad_final_20epoch')
state_dict_model = torch.load('checkpoint/model_final')

params = state_dict_model['params']

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print("Device: ", device)

# Data proprecessing.
transform = transforms.Compose([
    transforms.Resize(params['A']),
    transforms.ToTensor()])

test_dataset = dset.CIFAR10(root='../../kaggle/input/cifar-10-python', train=False,
                            download=True, transform=transform)

# Create the dataloader.
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=64,
                                              shuffle=True)

# Load the models
model = DRAWModel(params).to(device)
model.load_state_dict(state_dict_model['model'])
print(model)

Lclassifier = Classifier(model, device)
Lclassifier.load_state_dict(state_dict_classifier['model'])
print(Lclassifier)

loss_function = nn.CrossEntropyLoss()

Lclassifier.eval()

test_loss = 0

testloader = len(test_dataloader)

test_true = []
test_pred = []

for batchnum, (data, labels) in tqdm(enumerate(test_dataloader, 0), total=testloader):
    # Get batch size.
    bs = data.size(0)
    # Flatten the image.
    data = data.view(bs, -1).to(device)

    labels = labels.to(device)

    output = Lclassifier(data)

    loss = loss_function(output, labels)

    test_true.extend(labels.cpu().numpy())
    test_pred.extend(output.argmax(dim=1).cpu().numpy())
    test_loss += loss.item()

print('Test Loss: ', test_loss / testloader)

accuracy = accuracy_score(test_true, test_pred)
print('Accuracy: ', accuracy)
