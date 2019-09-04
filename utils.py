import os
import imageio
import argparse
import subprocess
from torchvision import transforms
import torch
from torch.nn import functional as F
import numpy as np

###############################################################################



def _traverse_discrete_line(dim, size):
    """
    Returns a (size, dim) latent sample, corresponding to a traversal of a
    discrete latent variable.

    Parameters
    ----------
    dim : int
        Number of categories of the selected discrete latent variable.

    traverse : bool
        If True, traverse the categorical variable otherwise keep it fixed
        or randomly sample.

    size : int
        Number of samples to generate.
    """
    samples = np.zeros((size, dim))

    for i in range(size):
        samples[i, i % dim] = 1.

    return torch.Tensor(samples)



def kl_multiple_discrete_loss(use_cuda, disc_latents):
    """
    Calculates the KL divergence between a set of categorical distributions
    and a set of uniform categorical distributions.

    Parameters
    ----------
    alphas : list
        List of the alpha parameters of a categorical (or gumbel-softmax)
        distribution. For example, if the categorical atent distribution of
        the model has dimensions [2, 5, 10] then alphas will contain 3
        torch.Tensor instances with the parameters for each of
        the distributions. Each of these will have shape (N, D).
    """
    # Calculate kl losses for each discrete latent
    kl_losses = [_kl_discrete_loss(use_cuda, alpha) for alpha in disc_latents]

    # Total loss is sum of kl loss for each discrete latent
    kl_loss = torch.sum(torch.cat(kl_losses))

    return kl_loss


def _kl_discrete_loss(use_cuda, alpha):
    """
    Calculates the KL divergence between a categorical distribution and a
    uniform categorical distribution.

    Parameters
    ----------
    alpha : torch.Tensor
        Parameters of the categorical or gumbel-softmax distribution.
        Shape (N, D)
    """
    Eps = 1e-12
    disc_dim = int(alpha.size()[-1])
    log_dim = torch.Tensor([np.log(disc_dim)])
    if use_cuda:
        log_dim = log_dim.cuda()
    # Calculate negative entropy of each row
    neg_entropy = torch.sum(alpha * torch.log(alpha + Eps), dim=1)
    # Take mean of negative entropy across batch
    mean_neg_entropy = torch.mean(neg_entropy, dim=0)
    # KL loss of alpha with uniform categorical variable
    kl_loss = log_dim + mean_neg_entropy
    return kl_loss




def apply_poe(use_cuda, muS_infA, logvarS_infA, muS_infB, logvarS_infB, logalpha, logalphaA, logalphaB):
    '''
    induce zS = encAB(xA,xB) via POE, that is,
        q(zI,zT,zS|xI,xT) := qI(zI|xI) * qT(zT|xT) * q(zS|xI,xT)
            where q(zS|xI,xT) \propto p(zS) * qI(zS|xI) * qT(zS|xT)
    '''

    ZERO = torch.zeros(logvarS_infA.shape)
    if use_cuda:
        ZERO = ZERO.cuda()

    aa=((ZERO - logalpha), -(logvarS_infA - logalphaA), -(logvarS_infB - logalphaB))
    bb= torch.stack(((ZERO - logalpha), -(logvarS_infA - logalphaA), -(logvarS_infB - logalphaB)), dim=2)
    logvarS = -torch.logsumexp(
        torch.stack(((ZERO - logalpha), -(logvarS_infA - logalphaA), -(logvarS_infB - logalphaB)), dim=2),
        dim=2
    )
    stdS = torch.sqrt(torch.exp(logvarS))
    muS = (muS_infA / torch.exp(logvarS_infA) +
           muS_infB / torch.exp(logvarS_infB)) * (stdS ** 2)

    return muS, stdS, logvarS

#-----------------------------------------------------------------------------#

def sample_gaussian(use_cuda, mu, std):

    # reparam'ed samples
    if use_cuda:
        Eps = torch.cuda.FloatTensor(mu.shape).normal_()
    else:
        Eps = torch.randn(mu.shape)
            
    return (mu + Eps*std)


def sample_gumbel_softmax(use_cuda, alpha, temperature=.67, train=True):
    """
    Samples from a gumbel-softmax distribution using the reparameterization
    trick.

    Parameters
    ----------
    alpha : torch.Tensor
        Parameters of the gumbel-softmax distribution. Shape (N, D)
    """

    # reparam'ed samples
    Eps = 1e-12
    if train:
        # Sample from gumbel distribution
        unif = torch.rand(alpha.size())
        if use_cuda:
            unif = unif.cuda()
        gumbel = -torch.log(-torch.log(unif + Eps) + Eps)
        # Reparameterize to create gumbel softmax sample
        log_alpha = torch.log(alpha + Eps)
        logit = (log_alpha + gumbel) / temperature
        return F.softmax(logit, dim=1)
    else:
        # In reconstruction mode, pick most likely sample
        _, max_alpha = torch.max(alpha, dim=1)
        one_hot_samples = torch.zeros(alpha.size())
        # On axis 1 of one_hot_samples, scatter the value 1 at indices
        # max_alpha. Note the view is because scatter_ only accepts 2D
        # tensors.
        one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
        if use_cuda:
            one_hot_samples = one_hot_samples.cuda()
        return one_hot_samples



###############################################################################

class DataGather(object):

    '''
    create (array)lists, one for each category, eg, 
      self.data['recon'] = [2.3, 1.5, 0.8, ...],
      self.data['kl'] = [0.3, 1.8, 2.2, ...], 
      self.data['acc'] = [0.3, 0.4, 0.5, ...], ...
    '''
    
    def __init__(self, *args):
        self.keys = args
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return {arg:[] for arg in self.keys}

    def insert(self, **kwargs):
        for key in kwargs.keys():
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()

###############################################################################

def str2bool(v):
    
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#-----------------------------------------------------------------------------#

def grid2gif(img_dir, key, out_gif, delay=100, duration=0.1):

    '''
    make (moving) GIF from images
    '''
    
    if True:  #os.name=='nt':
        
        fnames = [ \
          str(os.path.join(img_dir, f)) for f in os.listdir(img_dir) \
            if (key in f) and ('jpg' in f) ]
        
        fnames.sort()
        
        images = []
        for filename in fnames:
            images.append(imageio.imread(filename))
        
        imageio.mimsave(out_gif, images, duration=duration)
        
    else:  # os.name=='posix'
        
        img_str = str(os.path.join(img_dir, key+'*.jpg'))
        cmd = 'convert -delay %s -loop 0 %s %s' % (delay, img_str, out_gif)
        subprocess.call(cmd, shell=True)



def grid2gif2(img_dir, out_gif, delay=100, duration=0.1):
    '''
    make (moving) GIF from images
    '''

    if True:  # os.name=='nt':

        fnames = [ \
            str(os.path.join(img_dir, f)) for f in os.listdir(img_dir) \
            if ('jpg' in f)]

        fnames.sort()

        images = []
        for filename in fnames:
            images.append(imageio.imread(filename))

        imageio.mimsave(out_gif, images, duration=duration)

    else:  # os.name=='posix'

        img_str = str(os.path.join(img_dir, '*.jpg'))
        cmd = 'convert -delay %s -loop 0 %s %s' % (delay, img_str, out_gif)
        subprocess.call(cmd, shell=True)
#-----------------------------------------------------------------------------#

def mkdirs(path):

    if not os.path.exists(path):
        os.makedirs(path)


# def transform(image_size=None, image_size2=None, crop=None):
#     if image_size2:
#         transform = transforms.Compose([
#             transforms.Resize((image_size, image_size2)),
#             transforms.ToTensor()])
#     elif image_size:
#         transform = transforms.Compose([
#           transforms.Resize((image_size, image_size)),
#           transforms.RandomCrop(image_size),
#           transforms.ToTensor()])
#     elif crop:
#         transform = transforms.Compose([
#           transforms.RandomCrop(image_size),
#           transforms.ToTensor()])
#     else:
#         transform = transforms.Compose([
#           transforms.ToTensor()])
#     return transform


def transform(image, resize=None):
    from PIL import Image

    if len(image.shape) ==3:
        image = np.transpose(image, (1, 2, 0))
        image = Image.fromarray(image, mode='RGB')
    else:
        image = Image.fromarray(image, mode='L')
    if resize:
        image = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])(image)
    else:
        image = transforms.Compose([
            transforms.ToTensor()
        ])(image)
    return image



    # svhn_img = self.transformB(svhn_img)