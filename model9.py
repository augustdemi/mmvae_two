import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


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

    def __init__(self, zPrivate_dim=3, zShared_dim=5):
        super(EncoderA, self).__init__()

        self.zP_dim = zPrivate_dim
        self.zS_dim = zShared_dim

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2 * zPrivate_dim + 2 * zShared_dim)

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
        h = F.relu(self.fc1(x.view(-1, 784)))
        h = F.relu(self.fc2(h))
        stats = self.fc3(h)

        muPrivate = stats[:, :self.zP_dim]
        logvarPrivate = stats[:, self.zP_dim:(2 * self.zP_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate))

        muShared = stats[:, (2 * self.zP_dim):(2 * self.zP_dim + self.zS_dim)]
        logvarShared = stats[:, (2 * self.zP_dim + self.zS_dim):]
        stdShared = torch.sqrt(torch.exp(logvarShared))

        return (muPrivate, stdPrivate, logvarPrivate,
                muShared, stdShared, logvarShared)


class DecoderA(nn.Module):
    """Parametrizes p(x|z).

    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, zPrivate_dim=3, zShared_dim=5):
        super(DecoderA, self).__init__()
        self.fc1 = nn.Linear(zPrivate_dim + zShared_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 784)
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
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        x = self.fc4(h)
        x = x.view(-1, 1, 28, 28)
        return x  # NOTE: no sigmoid here. See train.py


# -----------------------------------------------------------------


class EncoderB(nn.Module):
    """Parametrizes q(z|x).

    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, zPrivate_dim=3, zShared_dim=5):
        super(EncoderB, self).__init__()
        self.zP_dim = zPrivate_dim
        self.zS_dim = zShared_dim

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.fc4 = nn.Linear(128*3*3, 256)
        self.fc5 = nn.Linear(256, 2*zPrivate_dim + 2*zShared_dim)

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
        x = F.relu(self.conv1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        out = F.dropout(x, p=0.5, training=self.training)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc4(out))
        stats = self.fc5(out)

        muPrivate = stats[:, :self.zP_dim]
        logvarPrivate = stats[:, self.zP_dim:(2 * self.zP_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate))

        muShared = stats[:, (2 * self.zP_dim):(2 * self.zP_dim + self.zS_dim)]
        logvarShared = stats[:, (2 * self.zP_dim + self.zS_dim):]
        stdShared = torch.sqrt(torch.exp(logvarShared))

        return (muPrivate, stdPrivate, logvarPrivate,
                muShared, stdShared, logvarShared)




class DecoderB(nn.Module):
    """Parametrizes p(x|z).

    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, zPrivate_dim=3, zShared_dim=5):
        super(DecoderB, self).__init__()

        self.fc1 = nn.Linear(zPrivate_dim + zShared_dim, 256)
        self.fc2 = nn.Linear(256, 128 * 3 * 3)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=5)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=5)
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=5)
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
        out = out.view(out.size(0), 128, 3, 3)
        out = F.upsample(out, scale_factor=2)
        out = F.relu(self.deconv3(out))
        out = F.upsample(out, scale_factor=2)
        out = F.relu(self.deconv4(out))
        x_recon = F.relu((self.deconv5(out)))
        return x_recon
