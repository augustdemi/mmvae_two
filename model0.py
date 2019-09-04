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

#-----------------------------------------------------------------------------#
            
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


#-----------------------------------------------------------------------

class EncoderA(nn.Module):
    """Parametrizes q(z|x).

    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, zPrivate_dim=3, zShared_dim=5):
        super(EncoderA, self).__init__()

        self.zP_dim = zPrivate_dim
        self.zS_dim = zShared_dim

        self.fc1   = nn.Linear(784, 512)
        self.fc2   = nn.Linear(512, 512)
        self.fc3  = nn.Linear(512, 2 * zPrivate_dim + 2 * zShared_dim)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x.view(-1, 784)))
        h = self.swish(self.fc2(h))
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
        x = self.fc4(h)
        x = x.view(-1, 1, 28, 28)
        return x  # NOTE: no sigmoid here. See train.py



#-----------------------------------------------------------------


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
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            Swish())
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            Swish(),
            nn.Linear(512, 2 * zPrivate_dim + 2 * zShared_dim))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        stats = self.classifier(x)
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
        self.upsampler = nn.Sequential(
            nn.Linear(zPrivate_dim + zShared_dim, 512),
            nn.ReLU,
            nn.Linear(512, 128 * 4 * 4),
            Swish())
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            Swish(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False))

    def forward(self, zPrivate, zShared):
        z = torch.cat((zPrivate, zShared), 1)
        # the input will be a vector of size |n_latents|
        z = self.upsampler(z)
        z = z.view(-1, 128, 4, 4)
        z = self.hallucinate(z)
        return z  # NOTE: no sigmoid here. See train.py

