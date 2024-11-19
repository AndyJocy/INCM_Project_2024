import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import random

from utils import Occlude, encoder_layer
from copy import copy

class DRAWModel(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.T = params['T']
        self.A = params['A']
        self.B = params['B']
        self.z_size = params['z_size']
        self.read_N = params['read_N']
        self.write_N = params['write_N']
        self.enc_size = params['enc_size']
        self.dec_size = params['dec_size']
        self.device = params['device']
        self.channel = params['channel']

        # Stores the generated image for each time step.
        self.cs = [0] * self.T

        # To store appropriate values used for calculating the latent loss (KL-Divergence loss)
        self.logsigmas = [0] * self.T
        self.sigmas = [0] * self.T
        self.mus = [0] * self.T

        # Define Occlusion parameters
        self.occlusion_prob = random.uniform(0.0, 1.0)
        self.square_size = 6

        # Define NREM Loss function
        self.NREM_loss_function = nn.MSELoss()

        # Define Linear Layer for Discriminator
        
        self.encoder = encoder_layer(self.read_N, self.channel, self.dec_size, self.enc_size)

        # To get the mean and standard deviation for the distribution of z.
        self.fc_mu = nn.Linear(self.enc_size, self.z_size)
        self.fc_sigma = nn.Linear(self.enc_size, self.z_size)

        self.decoder = nn.LSTMCell(self.z_size, self.dec_size)

        self.fc_write = nn.Linear(
            self.dec_size, self.write_N*self.write_N*self.channel)

        # To get the attention parameters. 5 in total.
        self.fc_attention = nn.Linear(self.dec_size, 5)

    def wake_forward(self, x):
        self.batch_size = x.size(0)

        # requires_grad should be set True to allow backpropagation of the gradients for training.
        h_enc_prev = torch.zeros(
            self.batch_size, self.enc_size, requires_grad=True, device=self.device)
        h_dec_prev = torch.zeros(
            self.batch_size, self.dec_size, requires_grad=True, device=self.device)

        enc_state = torch.zeros(
            self.batch_size, self.enc_size, requires_grad=True, device=self.device)
        dec_state = torch.zeros(
            self.batch_size, self.dec_size, requires_grad=True, device=self.device)

        # Create a tensor to store the total latent space of the encoder
        h_enc_total = torch.zeros(
            self.batch_size, self.enc_size, requires_grad=True, device=self.device)
        
        for t in range(self.T):
            c_prev = torch.zeros(self.batch_size, self.B*self.A*self.channel,
                                 requires_grad=True, device=self.device) if t == 0 else self.cs[t-1]
            # Equation 3.
            x_hat = x - torch.sigmoid(c_prev)
            # Equation 4.
            # Get the N x N glimpse.
            r_t = self.read(x, x_hat, h_dec_prev)
            # Equation 5.
            h_enc, enc_state = self.encoder.encoderception(
                torch.cat((r_t, h_dec_prev), dim=1), (h_enc_prev, enc_state))
            # Equation 6.
            z, self.mus[t], self.logsigmas[t], self.sigmas[t] = self.sampleQ(
                h_enc)
            # Equation 7.
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            # Equation 8.
            self.cs[t] = c_prev + self.write(h_dec)

            h_enc_prev = h_enc
            with torch.no_grad():
                h_enc_total += h_enc.detach().clone()
            
            h_dec_prev = h_dec

        return h_enc_prev, h_dec_prev, h_enc_total

    def read(self, x, x_hat, h_dec_prev):
        # Using attention
        (Fx, Fy), gamma = self.attn_window(h_dec_prev, self.read_N)

        def filter_img(img, Fx, Fy, gamma):
            Fxt = Fx.transpose(self.channel, 2)
            if self.channel == 3:
                img = img.view(-1, 3, self.B, self.A)
            elif self.channel == 1:
                img = img.view(-1, self.B, self.A)

            # Equation 27.
            glimpse = torch.matmul(Fy, torch.matmul(img, Fxt))
            glimpse = glimpse.view(-1, self.read_N*self.read_N*self.channel)

            return glimpse * gamma.view(-1, 1).expand_as(glimpse)

        x = filter_img(x, Fx, Fy, gamma)
        x_hat = filter_img(x_hat, Fx, Fy, gamma)

        return torch.cat((x, x_hat), dim=1)
        # No attention
        # return torch.cat((x, x_hat), dim=1)

    def write(self, h_dec):
        # Using attention
        # Equation 28.
        w = self.fc_write(h_dec)
        if self.channel == 3:
            w = w.view(self.batch_size, 3, self.write_N, self.write_N)
        elif self.channel == 1:
            w = w.view(self.batch_size, self.write_N, self.write_N)

        (Fx, Fy), gamma = self.attn_window(h_dec, self.write_N)
        Fyt = Fy.transpose(self.channel, 2)

        # Equation 29.
        wr = torch.matmul(Fyt, torch.matmul(w, Fx))
        wr = wr.view(self.batch_size, self.B*self.A*self.channel)

        return wr / gamma.view(-1, 1).expand_as(wr)
        # No attention
        # return self.fc_write(h_dec)

    def sampleQ(self, h_enc):
        e = torch.randn(self.batch_size, self.z_size, device=self.device)

        # Equation 1.
        mu = self.fc_mu(h_enc)
        # Equation 2.
        log_sigma = self.fc_sigma(h_enc)
        sigma = torch.exp(log_sigma)

        z = mu + e * sigma

        return z, mu, log_sigma, sigma

    def attn_window(self, h_dec, N):
        # Equation 21.
        params = self.fc_attention(h_dec)
        gx_, gy_, log_sigma_2, log_delta_, log_gamma = params.split(1, 1)

        # Equation 22.
        gx = (self.A + 1) / 2 * (gx_ + 1)
        # Equation 23
        gy = (self.B + 1) / 2 * (gy_ + 1)
        # Equation 24.
        delta = (max(self.A, self.B) - 1) / (N - 1) * torch.exp(log_delta_)
        sigma_2 = torch.exp(log_sigma_2)
        gamma = torch.exp(log_gamma)

        return self.filterbank(gx, gy, sigma_2, delta, N), gamma

    def filterbank(self, gx, gy, sigma_2, delta, N, epsilon=1e-8):
        grid_i = torch.arange(
            start=0.0, end=N, device=self.device, requires_grad=True,).view(1, -1)

        # Equation 19.
        mu_x = gx + (grid_i - N / 2 - 0.5) * delta
        # Equation 20.
        mu_y = gy + (grid_i - N / 2 - 0.5) * delta

        a = torch.arange(0.0, self.A, device=self.device,
                         requires_grad=True).view(1, 1, -1)
        b = torch.arange(0.0, self.B, device=self.device,
                         requires_grad=True).view(1, 1, -1)

        mu_x = mu_x.view(-1, N, 1)
        mu_y = mu_y.view(-1, N, 1)
        sigma_2 = sigma_2.view(-1, 1, 1)

        # Equations 25 and 26.
        Fx = torch.exp(-torch.pow(a - mu_x, 2) / (2 * sigma_2))
        Fy = torch.exp(-torch.pow(b - mu_y, 2) / (2 * sigma_2))

        Fx = Fx / (Fx.sum(2, True).expand_as(Fx) + epsilon)
        Fy = Fy / (Fy.sum(2, True).expand_as(Fy) + epsilon)

        if self.channel == 3:
            Fx = Fx.view(Fx.size(0), 1, Fx.size(1), Fx.size(2))
            Fx = Fx.repeat(1, 3, 1, 1)

            Fy = Fy.view(Fy.size(0), 1, Fy.size(1), Fy.size(2))
            Fy = Fy.repeat(1, 3, 1, 1)

        return Fx, Fy

    def wake_loss(self, x):
        h_enc, h_dec, h_enc_total = self.wake_forward(x)

        # DRAW Original Loss function
        criterion = nn.BCELoss()
        x_recon = torch.sigmoid(self.cs[-1])
        # Reconstruction loss.
        # Only want to average across the mini-batch, hence, multiply by the image dimensions.
        Lx = criterion(x_recon, x) * self.A * self.B * self.channel
        # Latent loss.
        Lz = 0

        for t in range(self.T):
            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]

            kl_loss = 0.5*torch.sum(mu_2 + sigma_2 -
                                    2*logsigma, 1) - 0.5*self.T
            Lz += kl_loss

        Lz = torch.mean(Lz)
        net_loss = Lx + Lz

        # Discriminator with target 0
        target = torch.zeros(self.batch_size, 1, device=self.device)
        loss_discriminator = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            h_enc_total_clone = h_enc_total.clone().detach()

        discriminator_output = self.encoder.discriminator(h_enc_total_clone)
        discriminator_loss_wake = loss_discriminator(discriminator_output, target)

        return net_loss, h_enc_total, discriminator_loss_wake

    def generate(self, num_output):
        self.batch_size = num_output
        h_dec_prev = torch.zeros(num_output, self.dec_size, device=self.device)
        dec_state = torch.zeros(num_output, self.dec_size, device=self.device)

        for t in range(self.T):
            c_prev = torch.zeros(self.batch_size, self.B*self.A*self.channel,
                                 device=self.device) if t == 0 else self.cs[t-1]
            z = torch.randn(self.batch_size, self.z_size, device=self.device)
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec

        imgs = []

        for img in self.cs:
            # The image dimesnion is B x A (According to the DRAW paper).
            img = img.view(-1, self.channel, self.B, self.A)
            imgs.append(vutils.make_grid(torch.sigmoid(img).detach().cpu(), nrow=int(
                np.sqrt(int(num_output))), padding=1, normalize=True, pad_value=1))

        return imgs

    def NREM_loss(self, wake_enc_total):
        with torch.no_grad():
            # Reconstruct the image
            reconstructed_image = torch.sigmoid(self.cs[-1].detach().clone())

            # Reshape the image
            reconstructed_image_resized = reconstructed_image.view(
                -1, self.channel, self.B, self.A)

            # Create a tensor to store the occluded image
            occluded_image = torch.zeros(self.batch_size, reconstructed_image_resized.size(
                1), reconstructed_image_resized.size(2), reconstructed_image_resized.size(3))

            # Occlude the image
            occlude = Occlude(drop_rate=self.occlusion_prob,
                              tile_size=self.square_size)
            occluded_image = occlude(reconstructed_image_resized, d=1)

            # Reshape the occluded image back to the original dimensions
            occluded_image = occluded_image.view(-1,
                                                 self.channel * self.B * self.A)

        # occluded_enc, occluded_dec = self.wake_forward(occluded_image)

        # Train the encoder on the occluded image for the NREM forward pass
        # NREM changes here ?
        occluded_enc_prev = torch.zeros(
            occluded_image.size(0), self.enc_size, requires_grad=True, device=self.device)
        occluded_dec_prev = torch.zeros(
            occluded_image.size(0), self.dec_size, requires_grad=False, device=self.device)

        occluded_enc_state = torch.zeros(
            occluded_image.size(0), self.enc_size, requires_grad=True, device=self.device)
        occluded_dec_state = torch.zeros(
            occluded_image.size(0), self.dec_size, requires_grad=False, device=self.device)

        occluded_cs = copy(self.cs)
        occldued_mus = copy(self.mus)
        occluded_logsigmas = copy(self.logsigmas)
        occluded_sigmas = copy(self.sigmas)

        occluded_enc_total = torch.zeros(
            occluded_image.size(0), self.enc_size, requires_grad=True, device=self.device)

        for t in range(self.T):
            c_prev = torch.zeros(occluded_image.size(0), self.B*self.A*self.channel,
                                 requires_grad=True, device=self.device) if t == 0 else occluded_cs[t-1]
            # Equation 3.
            x_hat = occluded_image - torch.sigmoid(c_prev)
            # Equation 4.
            # Get the N x N glimpse.
            r_t = self.read(occluded_image, x_hat, occluded_dec_prev)
            # Equation 5.
            h_enc, occluded_enc_state = self.encoder.encoderception(
                torch.cat((r_t, occluded_dec_prev), dim=1), (occluded_enc_prev, occluded_enc_state))
            # Equation 6.
            z, occldued_mus[t], occluded_logsigmas[t], occluded_sigmas[t] = self.sampleQ(
                h_enc)
            # Equation 7.
            h_dec, occluded_dec_state = self.decoder(
                z, (occluded_dec_prev, occluded_dec_state))
            # Equation 8.
            occluded_cs[t] = c_prev + self.write(h_dec)

            occluded_enc_prev = h_enc
            with torch.no_grad():
                occluded_enc_total += h_enc.detach().clone()

            occluded_dec_prev = h_dec

        # Calculate the NREM Loss (NREM Loss is currently MSE Loss between the encoder states)
        NREM_loss = self.NREM_loss_function(
            occluded_enc_total, wake_enc_total)

        return NREM_loss

    def REM_forward(self, h_enc_total, previous_Z):

        # Create REM Encoder
        # model_wake_encoder is cumulative z
        if h_enc_total.size(0) != previous_Z.size(0):
            # Clip the previous Z to the batch size
            previous_Z = previous_Z[:h_enc_total.size(0), :]
            
        rem_enc = (0.25 * h_enc_total) + (0.25 * previous_Z) + \
            (0.5 * torch.randn(self.batch_size, self.enc_size, device=self.device))

        # Reconstruct the image
        c_prev = torch.zeros(self.batch_size, self.B*self.A*self.channel,
                                device=self.device)
        h_dec_prev = torch.zeros(
            self.batch_size, self.dec_size, device=self.device)
        dec_state = torch.zeros(
            self.batch_size, self.dec_size, device=self.device)
        
        rem_z, rem_mu, rem_logsigma, rem_sigma = self.sampleQ(rem_enc)
        
        h_dec, dec_state = self.decoder(rem_z, (h_dec_prev, dec_state))

        rem_cs = c_prev + self.write(h_dec)

        final_rem_cs = [0] * self.T

        with torch.no_grad():
            rem_image = torch.sigmoid(rem_cs.detach().clone())

        rem_enc_prev = torch.zeros(
            self.batch_size, self.enc_size, requires_grad=True, device=self.device)
        rem_dec_prev = torch.zeros(
            self.batch_size, self.dec_size, requires_grad=True, device=self.device)

        rem_enc_state = torch.zeros(
            self.batch_size, self.enc_size, requires_grad=True, device=self.device)
        rem_dec_state = torch.zeros(
            self.batch_size, self.dec_size, requires_grad=True, device=self.device)

        final_encoder = torch.zeros(
            self.batch_size, self.enc_size, requires_grad=True, device=self.device)
        
        for t in range(self.T):
            c_prev = rem_cs if t == 0 else final_rem_cs[t-1]
            # Equation 3.
            x_hat = rem_image - torch.sigmoid(c_prev)
            # Equation 4.
            # Get the N x N glimpse.
            r_t = self.read(rem_image, x_hat, rem_dec_prev)
            # Equation 5.
            h_enc, rem_enc_state = self.encoder.encoderception(
                torch.cat((r_t, rem_dec_prev), dim=1), (rem_enc_prev, rem_enc_state))
            # Equation 6.
            z, self.mus[t], self.logsigmas[t], self.sigmas[t] = self.sampleQ(
                h_enc)
            # Equation 7.
            h_dec, rem_dec_state = self.decoder(z, (rem_dec_prev, rem_dec_state))
            # Equation 8.
            final_rem_cs[t] = c_prev + self.write(h_dec)

            rem_enc_prev = h_enc
            with torch.no_grad():
                final_encoder += h_enc.detach().clone()
            
            rem_dec_prev = h_dec

        # Discriminator with target 1
        target = torch.ones(self.batch_size, 1, device=self.device)
        loss_discriminator = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            final_encoder_clone = final_encoder.clone().detach()

        discriminator_output = self.encoder.discriminator(final_encoder_clone)
        discriminator_loss_rem = loss_discriminator(discriminator_output, target)

        return final_encoder, rem_enc, discriminator_loss_rem