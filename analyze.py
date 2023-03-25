import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'

latent_size = 100
hidden_size = 256
image_size = 784
num_epochs = 300
batch_size = 100


# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()) # Binary Cross Entropy loss 를 사용할 것이기에 sigmoid 사용!

# Generator
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())


netG = G.to(device)
netG.load_state_dict(torch.load("G.ckpt"))
netD = D.to(device)
netD.load_state_dict(torch.load("D.ckpt"))

with torch.no_grad():
    z = torch.randn(100, 100).to(device)
    fake_images = G(z)
    predict = D(fake_images)
    fake_images = fake_images.cpu()
    predict = predict.cpu()
    p = predict.numpy()

    rows = 5
    cols = 20

    fig, axs = plt.subplots(rows, cols, figsize=(16, 5))
    fig.suptitle("0: predict as fake, 1: predict as real, all fake")

    for i in range(rows * cols):
        row = i // 20
        col = i % 20
        img = fake_images[i].reshape(28, 28)
        axs[row, col].imshow(img, cmap='gray')
        axs[row, col].set_title("P%.2f" %p[i])
        axs[row, col].axis('off')

    plt.show()

    # fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    # save_image(fake_images, "fake.png")
