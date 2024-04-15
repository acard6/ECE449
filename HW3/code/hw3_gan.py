import struct
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from hw3_utils import BASE_URL, download, GANDataset

class DNet(nn.Module):
    """This is discriminator network."""

    def __init__(self):
        super(DNet, self).__init__()
        
        # TODO: implement layers here
        #functions
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.ReLU = nn.ReLU()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1, bias=True)
        #self.ReLU2 = nn.ReLU()
        #self.MaxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0, bias=True)
        #self.ReLU3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(200 , 1, bias=True)

        self._weight_init()

    def _weight_init(self):
        # TODO: implement weight initialization here
        for layer in self.children():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        # TODO: complete forward function
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.MaxPool(x)
        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.MaxPool(x)
        x = self.conv3(x)
        x = self.ReLU(x)
        x = self.flatten(x)
        x = self.fc1(x)

        return x


class GNet(nn.Module):
    """This is generator network."""

    def __init__(self, zdim):
        """
        Parameters
        ----------
            zdim: dimension for latent variable.
        """
        super(GNet, self).__init__()

        # TODO: implement layers here
        self.fc1 = nn.Linear(zdim, 1568, True)
        self.leaky = nn.LeakyReLU(0.2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=True)
        #another leakyReLU function w/ slope=0.2 so repeat previous leakyReLU
        #repeat another upsample
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=True)
        #another leakyReLU
        self.conv3 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.sig = nn.Sigmoid()

        self._weight_init()

    def _weight_init(self):
        # TODO: implement weight initialization here
        for layer in self.children():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, z):
        """
        Parameters
        ----------
            z: latent variables used to generate images.
        """
        # TODO: complete forward function
        z = self.fc1(z)
        z = self.leaky(z)
        z = z.reshape(-1 ,32, 7, 7)
        z = self.upsample(z)
        z = self.conv1(z)
        z = self.leaky(z)
        z = self.upsample(z)
        z = self.conv2(z)
        z = self.leaky(z)
        z = self.conv3(z)
        z = self.sig(z)

        return z


class GAN:
    def __init__(self, zdim=64):
        """
        Parameters
        ----------
            zdim: dimension for latent variable.
        """
        torch.manual_seed(2)
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._zdim = zdim
        self.disc = DNet().to(self._dev)
        self.gen = GNet(self._zdim).to(self._dev)

    def _get_loss_d(self, batch_size, batch_data, z):
        """This function computes loss for discriminator.

        Parameters
        ----------
            batch_size: #data per batch.
            batch_data: data from dataset.
            z: random latent variable.
        """
        # TODO: implement discriminator's loss function

        criterion = nn.BCEWithLogitsLoss()

        real_label = torch.ones(batch_size, 1)
        real_label = torch.autograd.Variable(real_label)
        fake_label = torch.zeros(batch_size, 1)
        fake_label = torch.autograd.Variable(fake_label)
        
        real = self.disc(batch_data)
        real_loss = criterion(real, real_label)
        
        fake = self.gen(z)
        fake2pointO = self.disc(fake)        
        fake_loss = criterion(fake2pointO, fake_label) 

        loss = (real_loss + fake_loss)/2
    
        return loss

    def _get_loss_g(self, batch_size, z):
        """This function computes loss for generator.
        Compute -\sum_z\log{D(G(z))} instead of \sum_z\log{1-D(G(z))}
        
        Parameters
        ----------
            batch_size: #data per batch.
            z: random latent variable.
        """
        # TODO: implement generator's loss function
        criterion = nn.BCEWithLogitsLoss()

        fake = self.gen(z)
        output = self.disc(fake)

        real_label = torch.ones(batch_size, 1)
        loss = criterion(output, real_label)

        return loss


    def train(self, iter_d=1, iter_g=1, n_epochs=100, batch_size=256, lr=0.0002):

        # first download
        f_name = "train-images-idx3-ubyte.gz"
        download(BASE_URL + f_name, f_name)

        print("Processing dataset ...")
        train_data = GANDataset(
            f"./data/{f_name}",
            self._dev,
            transform=transforms.Compose([transforms.Normalize((0.0,), (255.0,))]),
        )
        print(f"... done. Total {len(train_data)} data entries.")

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        dopt = optim.Adam(self.disc.parameters(), lr=lr, weight_decay=0.0)
        dopt.zero_grad()
        gopt = optim.Adam(self.gen.parameters(), lr=lr, weight_decay=0.0)
        gopt.zero_grad()

        for epoch in tqdm(range(n_epochs)):
            for batch_idx, data in tqdm(
                enumerate(train_loader), total=len(train_loader)
            ):

                z = 2 * torch.rand(data.size()[0], self._zdim, device=self._dev) - 1

                if batch_idx == 0 and epoch == 0:
                    plt.imshow(data[0, 0, :, :].detach().cpu().numpy())
                    plt.savefig("goal.pdf")

                if batch_idx == 0 and epoch % 10 == 0:
                    with torch.no_grad():
                        tmpimg = self.gen(z)[0:64, :, :, :].detach().cpu()
                    save_image(
                        tmpimg, "test_{0}.png".format(epoch), nrow=8, normalize=True
                    )

                dopt.zero_grad()
                for k in range(iter_d):
                    loss_d = self._get_loss_d(batch_size, data, z)
                    loss_d.backward()
                    dopt.step()
                    dopt.zero_grad()

                gopt.zero_grad()
                for k in range(iter_g):
                    loss_g = self._get_loss_g(batch_size, z)
                    loss_g.backward()
                    gopt.step()
                    gopt.zero_grad()

            print(f"E: {epoch}; DLoss: {loss_d.item()}; GLoss: {loss_g.item()}")


if __name__ == "__main__":
    gan = GAN()
    gan.train()
