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



class EncoderSingle3(nn.Module):
    '''
    single modal encoder architecture for the "3df" data
    '''

    ####
    def __init__(self, zPrivate_dim=3, zShared_dim=5):

        super(EncoderSingle3, self).__init__()

        self.zP_dim = zPrivate_dim
        self.zS_dim = zShared_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)
        self.fc5 = nn.Linear(64 * 4 * 4, 256)
        self.fc6 = nn.Linear(256, 2 * zPrivate_dim + 2 * zShared_dim)

        # initialize parameters
        self.weight_init()

    ####
    def weight_init(self, mode='normal'):

        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for m in self._modules:
            initializer(self._modules[m])

    ####
    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc5(out))
        stats = self.fc6(out)

        muPrivate = stats[:, :self.zP_dim]
        logvarPrivate = stats[:, self.zP_dim:(2 * self.zP_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate))

        muShared = stats[:, (2 * self.zP_dim):(2 * self.zP_dim + self.zS_dim)]
        logvarShared = stats[:, (2 * self.zP_dim + self.zS_dim):]
        stdShared = torch.sqrt(torch.exp(logvarShared))

        return (muPrivate, stdPrivate, logvarPrivate,
                muShared, stdShared, logvarShared)


# -----------------------------------------------------------------------------#

class DecoderSingle3(nn.Module):
    '''
    single modal decoder architecture for the "3df" data
    '''

    ####
    def __init__(self, zPrivate_dim=3, zShared_dim=5):

        super(DecoderSingle3, self).__init__()

        self.zP_dim = zPrivate_dim
        self.zS_dim = zShared_dim

        self.fc1 = nn.Linear(zPrivate_dim + zShared_dim, 256)
        self.fc2 = nn.Linear(256, 4 * 4 * 64)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.deconv6 = nn.ConvTranspose2d(32, 1, 4, 2, 1)

        # initialize parameters
        self.weight_init()

    ####
    def weight_init(self, mode='normal'):

        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for m in self._modules:
            initializer(self._modules[m])

    ####
    def forward(self, zPrivate, zShared):

        z = torch.cat((zPrivate, zShared), 1)

        out = F.relu(self.fc1(z))
        out = F.relu(self.fc2(out))
        out = out.view(out.size(0), 64, 4, 4)
        out = F.relu(self.deconv3(out))
        out = F.relu(self.deconv4(out))
        out = F.relu(self.deconv5(out))
        x_recon = self.deconv6(out)

        return x_recon

#--------------------------------------------------------

class ImageEncoder(nn.Module):
    """Parametrizes q(z|x).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, zPrivate_dim=3, zShared_dim=5):
        super(ImageEncoder, self).__init__()
        self.zP_dim = zPrivate_dim
        self.zS_dim = zShared_dim

        self.fc1   = nn.Linear(784, 512)
        self.fc2   = nn.Linear(512, 512)
        self.fc3  = nn.Linear(512, 2*zPrivate_dim + zShared_dim)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x.view(-1, 784)))
        h = self.swish(self.fc2(h))
        stats = self.fc3(h)
        muPrivate = stats[:, :self.zP_dim]
        logvarPrivate = stats[:, self.zP_dim:(2 * self.zP_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate))

        cate_prob = stats[:, (2 * self.zP_dim):]
        cate_prob = F.softmax(cate_prob, dim=1)
        return (muPrivate, stdPrivate, logvarPrivate, cate_prob)





class ImageDecoder(nn.Module):
    """Parametrizes p(x|z).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, zPrivate_dim=3, zShared_dim=5):
        super(ImageDecoder, self).__init__()

        self.zP_dim = zPrivate_dim
        self.zS_dim = zShared_dim
        self.fc1   = nn.Linear(zPrivate_dim + zShared_dim, 512)
        self.fc2   = nn.Linear(512, 512)
        self.fc3   = nn.Linear(512, 512)
        self.fc4   = nn.Linear(512, 784)
        self.swish = Swish()

    def forward(self, zPrivate, zShared):
        z = torch.cat((zPrivate, zShared), 1)
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        return self.fc4(h)  # NOTE: no sigmoid here. See train.py


class TextEncoder(nn.Module):
    """Parametrizes q(z|y).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, zPrivate_dim=3, zShared_dim=5):
        super(TextEncoder, self).__init__()

        self.zP_dim = zPrivate_dim
        self.zS_dim = zShared_dim
        self.fc1   = nn.Embedding(10, 512)
        self.fc2   = nn.Linear(512, 512)
        self.fc3  = nn.Linear(512, 2*zPrivate_dim + zShared_dim)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x))
        h = self.swish(self.fc2(h))
        stats = self.fc3(h)
        muPrivate = stats[:, :self.zP_dim]
        logvarPrivate = stats[:, self.zP_dim:(2 * self.zP_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate))

        cate_prob = stats[:, (2 * self.zP_dim):]
        cate_prob = F.softmax(cate_prob, dim=1)
        return (muPrivate, stdPrivate, logvarPrivate, cate_prob)


class TextDecoder(nn.Module):
    """Parametrizes p(y|z).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, zPrivate_dim=3, zShared_dim=5):
        super(TextDecoder, self).__init__()
        self.fc1   = nn.Linear(zPrivate_dim + zShared_dim, 512)
        self.fc2   = nn.Linear(512, 512)
        self.fc3   = nn.Linear(512, 512)
        self.fc4   = nn.Linear(512, 10)
        self.swish = Swish()

    def forward(self, zPrivate, zShared):
        z = torch.cat((zPrivate, zShared), 1)
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        return self.fc4(h)  # NOTE: no softmax here. See train.py

