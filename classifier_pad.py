import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader, TensorDataset
from model import DRAWModel
from utils import Occlude, get_data
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


# Load the checkpoint file.
state_dict = torch.load('checkpoint/model_final')

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print("Device: ", device)

# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Data proprecessing.
transform = transforms.Compose([
    transforms.Resize(params['A']),
    transforms.ToTensor()])

# Create the dataset.
train_dataset = dset.CIFAR10(root='../kaggle/input/cifar-10-python', train=True,
                             download=True, transform=transform)
test_dataset = dset.CIFAR10(root='../kaggle/input/cifar-10-python', train=False,
                            download=True, transform=transform)

# Create the dataloader.
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=64,
                                               shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=64,
                                              shuffle=True)

# Load the model
model = DRAWModel(params).to(device)

# Load the trained parameters.
model.load_state_dict(state_dict['model'])
print(model)

# Get the latent space for the training data.
train_data = []
train_labels = []

for param in model.parameters():
    param.requires_grad = False

loss_function = nn.CrossEntropyLoss()

Lclassifier = Classifier(model, device)

optimizer = optim.Adam(Lclassifier.parameters(), lr=0.001)

len_dataloader = len(train_dataloader)
total_loss = []

epochs = 20
for epoch in range(epochs):
    Lclassifier.train()
    curr_loss = 0
    for batchnum, (data, labels) in tqdm(enumerate(train_dataloader, 0), total=len_dataloader):
        # Get batch size.
        bs = data.size(0)
        # Flatten the image.
        data = data.view(bs, -1).to(device)
        # print(data.shape)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = Lclassifier(data)

        loss = loss_function(output, labels)
        curr_loss += loss.item()

        loss.backward()
        optimizer.step()

        # if batchnum % 100 == 0:
        #     print('Epoch: ', epoch, 'Batch: ', batchnum, 'Loss: ', loss.item())

    print('Epoch: ', epoch, 'Loss: ', curr_loss / len_dataloader)
    total_loss.append(curr_loss / len_dataloader)

# Save the classifier model.
torch.save({
    'model': Lclassifier.state_dict(),
    'optimizer': optimizer.state_dict()
}, '../kaggle/working/checkpoint/classifier_pad_final')

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

plt.figure(figsize=(10, 5))
plt.plot(total_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
plt.savefig('classifier_loss.png')
