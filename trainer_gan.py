import copy
from typing import Tuple

import numpy as np
from torch_geometric.nn.models import SchNet, DimeNet
from torch.utils.data import DataLoader
from datasets.AGU_dataset import AGUDataset
from datasets.GAN_dataset import GANDataset
import torch
from utils.meter import mae
import torch.nn as nn
from torch import Tensor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from models.heanet import HeaNet
from models.agunet import AguNet
from models.infogan import Generator, Discriminator, Infonet
import argparse
from utils.registry import registry, setup_imports
import time
from datasets.Mp_dataset import MpDataset, load_dataset, MpGeometricDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import SequentialSampler
from utils.utility import DataTransformer
import os
import pandas as pd

device = ('cuda' if torch.cuda.is_available() else 'cpu')




def train(dataloader, num_epochs, lamda):
    generator = Generator(6, 768).to(device)
    discriminator = Discriminator(768).to(device)
    infonet = Infonet(768, 6).to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.02, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.00000000000001, betas=(0.5, 0.999))
    optimizer_info = torch.optim.Adam(infonet.parameters(), lr=0.02, betas=(0.5, 0.999))

    adversarial_loss = torch.nn.BCELoss()
    continuous_loss = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            valid = torch.ones(imgs.size(0), 1).to(device)
            fake = torch.zeros(imgs.size(0), 1).to(device)
            imgs = imgs.to(device)
            labels = labels.to(device)

            # 训练判别器
            optimizer_d.zero_grad()
            # 真实图片损失
            real_loss = adversarial_loss(discriminator(imgs), valid)
            #生成图片
            noise = torch.rand(imgs.size(0), 58, dtype=torch.float32).to(device)
            labelstogen = torch.cat((noise, labels), 1)
            gen_imgs = generator(labels).detach()
            #假图片损失
            fake_loss = adversarial_loss(discriminator(gen_imgs), fake)
            #总损失
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            # 训练生成器和信息网络
            optimizer_g.zero_grad()
            optimizer_info.zero_grad()

            #生成图片的损失
            validity = discriminator(gen_imgs).detach()
            g_loss = adversarial_loss(validity, valid)

            #信息网络损失
            info_imgs = torch.cat((imgs, gen_imgs), 0)
            info_labels = torch.cat((labels, labels), 0)
            info_loss = continuous_loss(infonet(info_imgs), info_labels)

            #总损失

            gi_loss = g_loss + lamda * info_loss
            gi_loss.backward()
            optimizer_g.step()
            # info_loss.backward()
            optimizer_info.step()

            print(f"[Epoch {epoch}/{num_epochs}] [D loss: {d_loss.item()}] [G/Info loss: {gi_loss.item()}]")

    torch.save(generator.state_dict(), './saved_gan/generator.pt')
    torch.save(discriminator.state_dict(), './saved_gan/discriminator.pt')
    torch.save(infonet.state_dict(), './saved_gan/infonet.pt')

    return generator, discriminator, infonet

def test():
    generator = Generator(6, 768).to(device)
    genetator_name= './saved_gan/generator.pt'
    generator.load_state_dict(torch.load(genetator_name))
    generator.eval()
    noise = torch.rand(58, dtype=torch.float32).to(device)
    label = torch.tensor([0.4, 0., 0.15, 0.15, 0.15, 0.15], dtype=torch.float32).to(device)
    labelstogen = torch.cat((noise, label), 0)
    gen_imgs = generator(label)
    print(gen_imgs)




if __name__ == '__main__':
    dataset = GANDataset(txt_dir='change_fcc',
                         label_name='change_fcc/fcc.xlsx')
    data_loader = DataLoader(dataset=dataset, batch_size=32)

    # for data, labels in data_loader:
    #     data=data.to(device)
    #     labels= labels.to(device)
    #     print(labels)
    #     print(data)
    #     print(data.device, data.dtype, data.shape, labels.device, labels.dtype, labels.shape)

    #train(data_loader, 500, 1)
    test()


