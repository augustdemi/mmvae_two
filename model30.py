import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


###############################################################################

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


# -----------------------------------------------------------------------------#

def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


###############################################################################


class Discriminator(nn.Module):
    '''
    returns (n x 2): Let D1 = 1st column, D2 = 2nd column, then the meaning is
      D(z) (\in [0,1]) = exp(D1) / ( exp(D1) + exp(D2) )

      so, it follows: log( D(z) / (1-D(z)) ) = D1 - D2
    '''

    ####
    def __init__(self, z_dim):

        super(Discriminator, self).__init__()

        self.z_dim = z_dim

        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000), nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000), nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000), nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000), nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000), nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )

        self.weight_init()

    ####
    def weight_init(self, mode='normal'):

        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    ####
    def forward(self, z):

        return self.net(z)



class Alpha(nn.Module):

    def __init__(self):
        super(Alpha, self).__init__()

        self.logalpha = nn.Parameter(0.01 * torch.randn(1))
        self.logalphaA = nn.Parameter(0.01 * torch.randn(1))
        self.logalphaB = nn.Parameter(0.01 * torch.randn(1))

    def forward(self):
        return self.logalpha, self.logalphaA, self.logalphaB


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""

    def forward(self, x):
        return x * F.sigmoid(x)


# -----------------------------------------------------------------------

class EncoderA(nn.Module):
    """Parametrizes q(z|x).

    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, zPrivate_dim=3, zShared_dim=5, channel=1):
        super(EncoderA, self).__init__()
        self.zP_dim = zPrivate_dim
        self.zS_dim = zShared_dim

        self.features = nn.Sequential(
            nn.Conv2d(channel, 32, 4, 2, 1),
            nn.ReLU(),
            # nn.Conv2d(32, 32, 4, 2, 1),
            # nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU())


        self.fc4 = nn.Linear(64*4*4, 256)
        self.fc5 = nn.Linear(256, 2*zPrivate_dim + zShared_dim)

        self.weight_init()

    ####
    def weight_init(self, mode='normal'):

        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for m in self._modules:
            initializer(self._modules[m])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc4(out))
        stats = self.fc5(out)

        muPrivate = stats[:, :self.zP_dim]
        logvarPrivate = stats[:, self.zP_dim:(2 * self.zP_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate))

        cate_prob = stats[:, (2 * self.zP_dim):]
        cate_prob = F.softmax(cate_prob, dim=1)
        return (muPrivate, stdPrivate, logvarPrivate, cate_prob)


class DecoderA(nn.Module):
    """Parametrizes p(x|z).

    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, zPrivate_dim=3, zShared_dim=5, channel=1):
        super(DecoderA, self).__init__()

        self.hallucinate = nn.Sequential(
            # nn.ConvTranspose2d(64, 64, 4, 2, 1),
            # nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channel, 4, 2, 1))


        self.fc1 = nn.Linear(zPrivate_dim + zShared_dim, 256)
        self.fc2 = nn.Linear(256, 64 * 4 * 4)
        self.weight_init()

    ####
    def weight_init(self, mode='normal'):

        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for m in self._modules:
            initializer(self._modules[m])

    def forward(self, zPrivate, zShared):
        z = torch.cat((zPrivate, zShared), 1)
        out = F.relu(self.fc1(z))
        out = F.relu(self.fc2(out))
        out = out.view(out.size(0), 64, 4, 4)
        x_recon = self.hallucinate(out)
        return x_recon


# -----------------------------------------------------------------


