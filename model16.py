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

    def __init__(self, zPrivate_dim=3, zShared_dim=5):
        super(EncoderA, self).__init__()
        self.zP_dim = zPrivate_dim
        self.zS_dim = zShared_dim

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU())


        self.fc4 = nn.Linear(32*4*4, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 2*zPrivate_dim + 2*zShared_dim)

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


class DecoderA(nn.Module):
    """Parametrizes p(x|z).

    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, zPrivate_dim=3, zShared_dim=5):
        super(DecoderA, self).__init__()

        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1))


        self.fc1 = nn.Linear(zPrivate_dim + zShared_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 32 * 4 * 4)
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
        out = out.view(out.size(0), 32, 4, 4)
        x_recon = self.hallucinate(out)
        return x_recon


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

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU())


        self.fc4 = nn.Linear(32*4*4, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 2*zPrivate_dim + 2*zShared_dim)

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


class DecoderB(nn.Module):
    """Parametrizes p(x|z).

    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, zPrivate_dim=3, zShared_dim=5):
        super(DecoderB, self).__init__()

        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1))


        self.fc1 = nn.Linear(zPrivate_dim + zShared_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 32 * 4 * 4)
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
        out = out.view(out.size(0), 32, 4, 4)
        x_recon = self.hallucinate(out)
        return x_recon



class EncoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Encoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(EncoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar

class DecoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Decoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(DecoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x
