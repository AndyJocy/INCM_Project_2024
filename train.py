import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from copy import deepcopy
from tqdm import tqdm

from model import DRAWModel
from utils import calculate_kl_loss, get_data

checkpoint_dir = '/kaggle/working/checkpoint'
os.makedirs(checkpoint_dir, exist_ok=True)

# Function to generate new images and save the time-steps as an animation.


def generate_image(epoch):
    x = model.generate(64)
    fig = plt.figure(figsize=(16, 16))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in x]
    anim = animation.ArtistAnimation(
        fig, ims, interval=500, repeat_delay=1000, blit=True)
    anim.save('/kaggle/working/draw_epoch_{}.gif'.format(epoch),
              dpi=100, writer='imagemagick')
    plt.close('all')


# Dictionary storing network parameters.
params = {
    'T': 64,  # Number of glimpses.
    'batch_size': 128,  # Batch size.
    'A': 32,  # Image width
    'B': 32,  # Image height
    'z_size': 200,  # Dimension of latent space.
    'read_N': 5,  # N x N dimension of reading glimpse.
    'write_N': 5,  # N x N dimension of writing glimpse.
    'dec_size': 400,  # Hidden dimension for decoder.
    'enc_size': 400,  # Hidden dimension for encoder.
    'epoch_num': 50,  # Number of epochs to train for.
    'learning_rate': 1e-3,  # Learning rate.
    'beta1': 0.5,
    'clip': 5.0,
    # After how many epochs to save checkpoints and generate test output.
    'save_epoch': 10,
    'channel': None}  # Number of channels for image.(3 for RGB, etc.)

# Use GPU is available else use CPU.
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

params['device'] = device

train_loader = get_data(params)
params['channel'] = 3

"""
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/', train='train', download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()])),
    batch_size=params['batch_size'], shuffle=True)

params['channel'] = 1
"""

# Plot the training images.
sample_batch = next(iter(train_loader))
plt.figure(figsize=(16, 16))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[: 64], nrow=8, padding=1, normalize=True, pad_value=1).cpu(), (1, 2, 0)))
plt.savefig("/kaggle/working/Training_Data")

# Initialize the model.
model = DRAWModel(params).to(device)
print("Occlusion Probability: ", model.occlusion_prob)
# Adam Optimizer
encoder_optimizer = optim.Adam(
    model.encoder.parameters(), lr=params['learning_rate'], betas=(params['beta1'], 0.999))
decoder_optimizer = optim.Adam(
    model.decoder.parameters(), lr=params['learning_rate'], betas=(params['beta1'], 0.999))

# List to hold the losses for each iteration.
# Used for plotting loss curve.
wake_losses = []
NREM_losses = []
REM_losses = []
iters = 0
avg_loss = 0
avg_wake_loss = 0
NREM_avg_loss = 0
REM_avg_loss = 0
previous_Z = None

print("-"*25)
print("Starting Training Loop...\n")
print('Epochs: %d\nBatch Size: %d\nLength of Data Loader: %d' %
      (params['epoch_num'], params['batch_size'], len(train_loader)))
print("-"*25)

start_time = time.time()

for epoch in range(params['epoch_num']):
    epoch_start_time = time.time()

    for i, (data, _) in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
        # Get batch size.
        bs = data.size(0)
        # Flatten the image.
        data = data.view(bs, -1).to(device)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        # Calculate the loss.
        loss, model_wake_encoder, discriminator_loss_wake = model.wake_loss(data)
        loss_val = loss.cpu().data.numpy()
        avg_loss += loss_val

        # KL divergence loss between encoder and normal distribution.
        # Calculate the KL Divergence Loss
        kl_loss = calculate_kl_loss(model_wake_encoder)

        wake_loss = loss + kl_loss + discriminator_loss_wake
        avg_wake_loss_val = wake_loss.cpu().data.numpy()
        avg_wake_loss += avg_wake_loss_val

        # Wake Step
        # Calculate the gradients
        wake_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip'])
        # Update parameters.
        encoder_optimizer.step()
        decoder_optimizer.step()

        # reset the gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # decoder.eval() will set the gradients off for the decoder
        # also wrap it in a torch.no_grad() block

        # NREM Step
        # Turn off gradients for the decoder
        model.decoder.eval()

        # Calculate the NREM Loss
        NREM_loss = model.NREM_loss(model_wake_encoder)
        NREM_loss_val = NREM_loss.cpu().data.numpy()
        NREM_avg_loss += NREM_loss_val

        # Calculate the gradients
        NREM_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip'])
        # Update parameters
        encoder_optimizer.step()

        # REM Step
        # Turn on the gradients for the decoder
        model.decoder.train()
        model.encoder.train()

        # Reset the gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # REM Forward call
        if i == 0:
            previous_Z = deepcopy(model_wake_encoder)

        rem_final_encoder, previous_Z, discriminator_loss_REM = model.REM_forward(model_wake_encoder, previous_Z)

        # Calculate the REM Loss
        REM_loss_val = discriminator_loss_REM.cpu().data.numpy()
        REM_avg_loss += REM_loss_val
        
        discriminator_loss_REM.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip'])

        for param in model.decoder.parameters():
            if param.grad is not None:
                param.grad = -param.grad

        encoder_optimizer.step()
        decoder_optimizer.step()

        # The encoding for this step will be Z = 0.25 * current wake encoding + 0.25 * previous Z + 0.5 * random loss vector.
        # For rem we have to do do nrem but without occlusion.
        # When we get the final encoder state we will NOT do MSE and instead pass the encoding into a linear layer that outputs a singular float values (batch size x 1) and then we will do a sigmoid on that and then do BCELoss with the target being 1.
        # After we get loss and do backward, we will go through the gradients of the decoder and negate all of them before doing the step.

        # Loss in Wake is MSELoss between images, BCELoss between discriminator and 0, and KL Divergence Loss between the encoder and a normal distribution.
        # Loss in NREM is MSELoss between the encoder states.
        # Loss in REM is BCELoss between the discriminator and 1.

        # Check progress of training.
        if i != 0 and i % 100 == 0:
            print('[%d/%d][%d/%d]\tWake_Loss: %.4f\tNREM_Loss: %.4f\tREM_Loss: %.4f'
                  % (epoch+1, params['epoch_num'], i, len(train_loader), avg_wake_loss / 100, NREM_avg_loss / 100, REM_avg_loss / 100))

            avg_loss = 0
            avg_wake_loss = 0
            NREM_avg_loss = 0
            REM_avg_loss = 0

        # Save the losses for plotting.
        wake_losses.append(avg_wake_loss_val)
        NREM_losses.append(NREM_loss_val)
        REM_losses.append(REM_loss_val)
        iters += 1

    avg_loss = 0
    avg_wake_loss = 0
    NREM_avg_loss = 0
    REM_avg_loss = 0
    epoch_time = time.time() - epoch_start_time
    print("Time Taken for Epoch %d: %.2fs" % (epoch + 1, epoch_time))
    # Save checkpoint and generate test output.
    if (epoch+1) % params['save_epoch'] == 0:
        torch.save({
            'model': model.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict(),
            'params': params
        }, '/kaggle/working/checkpoint/model_epoch_{}'.format(epoch+1))

        with torch.no_grad():
            generate_image(epoch+1)

training_time = time.time() - start_time
print("-"*50)
print('Training finished!\nTotal Time for Training: %.2fm' %
      (training_time / 60))
print("-"*50)
# Save the final trained network paramaters.
torch.save({
    'model': model.state_dict(),
    'encoder_optimizer': encoder_optimizer.state_dict(),
    'decoder_optimizer': decoder_optimizer.state_dict(),
    'params': params
}, '/kaggle/working/checkpoint/model_final'.format(epoch))

# Generate test output.
with torch.no_grad():
    generate_image(params['epoch_num'])

# Plot the training losses.
plt.figure(figsize=(10, 5))
plt.title("Training Loss")
plt.plot(wake_losses)
plt.xlabel("iterations")
plt.ylabel("Wake_Loss")
plt.savefig("/kaggle/working/Wake_Loss_Curve")

plt.figure(figsize=(10, 5))
plt.title("Training Loss")
plt.plot(NREM_losses)
plt.xlabel("iterations")
plt.ylabel("NREM_Loss")
plt.savefig("/kaggle/working/NREM_Loss_Curve")

plt.figure(figsize=(10, 5))
plt.title("Training Loss")
plt.plot(REM_losses)
plt.xlabel("iterations")
plt.ylabel("REM_Loss")
plt.savefig("/kaggle/working/REM_Loss_Curve")