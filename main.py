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


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])
# if RGB input
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

mnist_train = dsets.MNIST(root="data/",
                          train=True,
                          transform=transform,
                          download=True)

mnist_test = dsets.MNIST(root='data/',
                         train=False,
                         transform=transform,
                         download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)


# mnist_train 다루기
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(mnist_train), size=(1,)).item()
#     img, label = mnist_train[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.axis("off") # x축, y축 안보이게 설정
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()


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


D = D.to(device)
G = G.to(device)


def imshow(img):
    img = (img+1) / 2
    img = img.squeeze() # 차원 중 사이즈 1 을 제거
    np_img = img.numpy() # 이미지 픽셀을 넘파이 배열로 변환
    plt.imshow(np_img, cmap='gray')
    plt.show()


def imshow_grid(img):
    img = utils.make_grid(img.cpu().detach()) # 이미지 그리드 생성, 이미지 출력만을 위해 cpu에 담고 추적 방이
    img = (img+1)/2
    npimg = img.numpy() # 이미지 픽셀을 넘파이 배열로 변환
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 가짜 이미지 생성 및 관찰
# rand_img = torch.randn(1, 100, device=device)
# img_1 = G(rand_img).view(-1, 28, 28)
# imshow(img_1.squeeze().cpu().detach())

criterion = nn.BCELoss().to(device)
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def reset_grad(): # 가중치를 0으로 초기화
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


dx_epoch = []
dgx_epoch = []
# 데이터 개수 / 배치사이즈 = 60k / 100 = 600
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # 600 * 1 * 28 * 28 => 600 * 784
        images = images.reshape(batch_size, -1).to(device)

        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # 더 쉽게 설명하면:
        # data_loader 에서 온 이미지면   정답 y = 1 -> Loss = ln(x)
        # randn() 으로 만들어 온 이미지면 정답 y = 0 -> Loss = ln(1-x) x=0.5 대칭
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)

        g_loss = criterion(outputs, real_labels)

        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                          real_score.mean().item(), fake_score.mean().item()))
    dx_epoch.append(real_score.mean().item())
    dgx_epoch.append(fake_score.mean().item())
    # real image 저장
    # if (epoch + 1) == 1:
    #     images = images.reshape(images.size(0), 1, 28, 28)
    #     save_image(denorm(images), os.path.join("./", 'real_images.png'))

    # 생성된 이미지 저장
    # fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    # save_image(denorm(fake_images), os.path.join("./fake_images/", 'fake_images-{}.png'.format(epoch + 1)))

# 생성자, 판별자 각각 모델 저장
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')

# 참고문헌
# https://kingnamji.tistory.com/11

