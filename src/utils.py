import math
from typing import Tuple
from einops.einops import reduce, rearrange
import torch.nn as nn
import torch
from functools import partial
from einops.layers.torch import Rearrange
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from typing import Any, Callable
import numpy as np
from matplotlib import pyplot as plt
import cv2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class Siren(pl.LightningModule):
    def __init__(self):
        super(Siren, self).__init__()
    def forward(self,x):
        return torch.sin(x)

def plot_islands_agreement(levels, image):
    image_cpu = image.permute(1,2,0).detach().cpu().numpy()
    lin = nn.Linear(levels.shape[-1], 2).cuda()
    levels_2 = lin(levels)
    levels_2 = rearrange(levels_2,'(w h) l a -> w h l a', w = int(math.sqrt(levels.detach().cpu().numpy().shape[0])))
    levels_cpu_2 = levels_2.detach().cpu().numpy()

    mylevels = []
    for l in range(levels_cpu_2.shape[2]):
        mylevels.append(levels_cpu_2[:,:,l,:])

    fig, axs = plt.subplots(1, len(mylevels) + 1)
    plt.rcParams["figure.figsize"] = (25,3)
    axs[-1].imshow(image_cpu)
    axs[-1].set_box_aspect(1)
    axs[-1].grid(False)
    axs[-1].axes.xaxis.set_visible(False)
    axs[-1].axes.yaxis.set_visible(False)
    for i, matrice in enumerate(mylevels):
        x = np.arange(0.5, matrice.shape[0], 1)
        y = np.arange(0.5, matrice.shape[0], 1)
        xx, yy = np.meshgrid(x, y)
        r = np.power(np.add(np.power(matrice[:,:,0],2), np.power(matrice[:,:,1],2)),0.5)
        axs[i].imshow(r, cmap='inferno', interpolation='nearest')
        
        axs[i].set_box_aspect(1)
        axs[i].grid(False)
        axs[i].axes.xaxis.set_visible(False)
        axs[i].axes.yaxis.set_visible(False)
    
    plt.show()