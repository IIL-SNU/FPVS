import os

import math
import scipy.io
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import torch
import torch.optim as optim


import datetime
import wandb
from torchsummary import summary
import torch.nn as nn


from FPM_recon_dataset_RED import FPM_recon_dataset_RED
from FPM_dataset_RED import FPM_dataset_RED

import matplotlib.pyplot as plt

import time
import imageio
import visualizer
from PIL import Image
import io



    
def tensor_norm_double(X):
    X=X-torch.min(X)
    X=X/torch.max(X)
    return (X)

def alt_axis(X):
    return np.einsum('ijk->jki', X)    




class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, device, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device=device

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
        device=self.device
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
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def tv_iso2d(x):
    # nz-nx-ny
    if len(x.shape) == 4:
        dx = x[:-1,:-1,1:,:-1]-x[:-1,:-1,:-1,:-1]
        dy = x[:-1,:-1,:-1,1:]-x[:-1,:-1,:-1,:-1]
        return torch.sqrt(torch.abs(dx)**2 + torch.abs(dy)**2).mean()
    if len(x.shape) == 3:
        dx = x[:-1,1:,:-1]-x[:-1,:-1,:-1]
        dy = x[:-1,:-1,1:]-x[:-1,:-1,:-1]
        return torch.sqrt(torch.abs(dx)**2 + torch.abs(dy)**2).mean()
    if len(x.shape) == 2:
        dx = x[1:,:]-x[:-1,:]
        dy = x[:,1:]-x[:,:-1]
        return torch.sqrt(torch.abs(dx)**2 + torch.abs(dy)**2).mean()

def minmax_norm_number(x,maxVal,minVal):
    x=x-minVal
    x=x*(x>0)
    x=x/(maxVal-minVal)
    x=x*(x<1)+(x>=1)
    
    
    return x



def norm_minmax(x):
    x=x-torch.min(x)
    x=x/torch.max(x)
    return x


def led_visualize_individual(TPARAMS,led_idx):
    led_weights=TPARAMS['model_recon'].led_weights.clone().cpu().detach().numpy()
    # led_weights=np.zeros((1,193))
    # led_weights[0,led_idx]=1
    # led_weights=led_weights_tot(led_idx)
    fig=visualizer.visualize_square(np.expand_dims(led_weights[led_idx,:], axis=0), TPARAMS['led_list'])
    # fig=visualizer.visualize(np.expand_dims(led_weights, axis=0), TPARAMS['led_list'])
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png',dpi=200,transparent=True)
    plt.close()
    im = Image.open(img_buf)
    return im    
    


def led_visualize(TPARAMS):
    led_weights=TPARAMS['model_recon'].led_weights.clone().cpu().detach().numpy()
    fig=visualizer.visualize(led_weights, TPARAMS['led_list'])

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png',dpi=200,transparent=True)
    plt.close()
    im = Image.open(img_buf)
    return im    
    
    
    
def plot_train_recon(train_result,epoch,mode):
    
    max_phase,min_phase=torch.max(train_result['label_phase']),torch.min(train_result['label_phase'])
    grid_label_phase=torchvision.utils.make_grid((train_result['label_phase']),min=min_phase,max=max_phase)
    grid_output_phase=torchvision.utils.make_grid((train_result['output_phase']),min=min_phase,max=max_phase)
    grid_label_amp=torchvision.utils.make_grid((train_result['label_amp']))
    grid_output_amp=torchvision.utils.make_grid((train_result['output_amp']))
    grid_label_BF=torchvision.utils.make_grid((train_result['input_BF']))
    grid_label_DF=torchvision.utils.make_grid((train_result['input_DF']))
    # max_BF,min_BF=torch.max(train_result['input_BF']),torch.min(train_result['input_BF'])
    # max_DF,min_DF=torch.max(train_result['input_DF']),torch.min(train_result['input_DF'])
    
    # grid_output_BF=torchvision.utils.make_grid(torch.clamp(train_result['input_BF_estimated'],min=min_BF,max=max_BF))
    # grid_output_DF=torchvision.utils.make_grid(torch.clamp(train_result['input_DF_estimated'],min=min_DF,max=max_DF))
    

    if mode =='train':
        wandb.log({
                    "Train Label Phase": wandb.Image((grid_label_phase)),
                    "Train Output Phase": wandb.Image((grid_output_phase)),
                    "Train Label Amp" : wandb.Image((grid_label_amp)),
                    "Train Output Amp" : wandb.Image((grid_output_amp)),
                    "Train Label BF": wandb.Image((grid_label_BF)),
                    "Train Label DF": wandb.Image((grid_label_DF)),
                    "LED Power": wandb.Image((train_result['LED_Power'])),
            
                    # "Train Output BF": wandb.Image((grid_output_BF)),
                    # "Train Output DF": wandb.Image((grid_output_DF))
                    }, step=epoch)
    elif mode =='test':
        wandb.log({
                    "Test Label Phase": wandb.Image((grid_label_phase)),
                    "Test Output Phase": wandb.Image((grid_output_phase)),
                    "Test Label Amp" : wandb.Image((grid_label_amp)),
                    "Test Output Amp" : wandb.Image((grid_output_amp)),
                    "Test Label BF": wandb.Image((grid_label_BF)),
                    "Test Label DF": wandb.Image((grid_label_DF)),
                    # "Test Output BF": wandb.Image((grid_output_BF)),
                    # "Test Output DF": wandb.Image((grid_output_DF))
                    }, step=epoch)
        
        
def resize_image_tensor(image_tensor):
    # Get the original dimensions of the tensor
    _, original_height, original_width = image_tensor.shape

    # Calculate the new dimensions (half of the original size)
    new_width = original_width // 8
    new_height = original_height // 8

    # Resize the image tensor
    resized_tensor = F.resize(image_tensor, [new_height, new_width])

    return resized_tensor


def plot_train_entire(train_result,epoch,mode):
    
#     max_phase,min_phase=torch.max(train_result['label_phase']),torch.min(train_result['label_phase'])
#     grid_label_phase=torchvision.utils.make_grid(torch.clamp(train_result['label_phase'],min=min_phase,max=max_phase))
#     grid_output_phase=torchvision.utils.make_grid(torch.clamp(train_result['output_phase'],min=min_phase,max=max_phase))
#     grid_label_amp=torchvision.utils.make_grid((train_result['label_amp']))
#     grid_output_amp=torchvision.utils.make_grid((train_result['output_amp']))
#     grid_label_BF=torchvision.utils.make_grid((train_result['input_BF']))
#     grid_label_DF=torchvision.utils.make_grid((train_result['input_DF']))
#     # max_BF,min_BF=torch.max(train_result['input_BF']),torch.min(train_result['input_BF'])
#     # max_DF,min_DF=torch.max(train_result['input_DF']),torch.min(train_result['input_DF'])
    
#     # grid_output_BF=torchvision.utils.make_grid(torch.clamp(train_result['input_BF_estimated'],min=min_BF,max=max_BF))
#     # grid_output_DF=torchvision.utils.make_grid(torch.clamp(train_result['input_DF_estimated'],min=min_DF,max=max_DF))
#     grid_output_stain=torchvision.utils.make_grid(torch.clamp(train_result['output_stain'],min=0,max=1))
#     grid_label_stain=torchvision.utils.make_grid(torch.clamp(train_result['label_stain'],min=0,max=1))
    
    
    max_phase,min_phase=torch.max(train_result['label_phase']),torch.min(train_result['label_phase'])
    grid_label_phase=resize_image_tensor(torchvision.utils.make_grid(torch.clamp(train_result['label_phase'],min=min_phase,max=max_phase)))
    grid_output_phase=resize_image_tensor(torchvision.utils.make_grid(torch.clamp(train_result['output_phase'],min=min_phase,max=max_phase)))
    grid_label_amp=resize_image_tensor(torchvision.utils.make_grid((train_result['label_amp'])))
    grid_output_amp=resize_image_tensor(torchvision.utils.make_grid((train_result['output_amp'])))
    grid_label_BF=resize_image_tensor(torchvision.utils.make_grid((train_result['input_BF'])))
    grid_label_DF=resize_image_tensor(torchvision.utils.make_grid((train_result['input_DF'])))
    # max_BF,min_BF=torch.max(train_result['input_BF']),torch.min(train_result['input_BF'])
    # max_DF,min_DF=torch.max(train_result['input_DF']),torch.min(train_result['input_DF'])
    
    # grid_output_BF=torchvision.utils.make_grid(torch.clamp(train_result['input_BF_estimated'],min=min_BF,max=max_BF))
    # grid_output_DF=torchvision.utils.make_grid(torch.clamp(train_result['input_DF_estimated'],min=min_DF,max=max_DF))
    grid_output_stain=resize_image_tensor(torchvision.utils.make_grid(torch.clamp(train_result['output_stain'],min=0,max=1)))
    grid_label_stain=resize_image_tensor(torchvision.utils.make_grid(torch.clamp(train_result['label_stain'],min=0,max=1)))
    
    

    if mode =='train':
        wandb.log({
                    "Train Label Phase": wandb.Image((grid_label_phase)),
                    "Train Output Phase": wandb.Image((grid_output_phase)),
                    "Train Label Amp" : wandb.Image((grid_label_amp)),
                    "Train Output Amp" : wandb.Image((grid_output_amp)),
                    "Train Label BF": wandb.Image((grid_label_BF)),
                    "Train Label DF": wandb.Image((grid_label_DF)),
                    "LED Power": wandb.Image((train_result['LED_Power'])),
                    "Train Label Stain": wandb.Image((grid_label_stain)),
                    "Train Output Stain": wandb.Image((grid_output_stain)),
            
                    # "Train Output BF": wandb.Image((grid_output_BF)),
                    # "Train Output DF": wandb.Image((grid_output_DF))
                    }, step=epoch)
    elif mode =='test':
        wandb.log({
                    "Test Label Phase": wandb.Image((grid_label_phase)),
                    "Test Output Phase": wandb.Image((grid_output_phase)),
                    "Test Label Amp" : wandb.Image((grid_label_amp)),
                    "Test Output Amp" : wandb.Image((grid_output_amp)),
                    "Test Label BF": wandb.Image((grid_label_BF)),
                    "Test Label DF": wandb.Image((grid_label_DF)),
            
                    "Test Label Stain": wandb.Image((grid_label_stain)),
                    "Test Output Stain": wandb.Image((grid_output_stain)),
                    # "Test Output BF": wandb.Image((grid_output_BF)),
                    # "Test Output DF": wandb.Image((grid_output_DF))
                    }, step=epoch)        
        
    
    
def plot_train_forward(train_result,epoch,mode):
    
    grid_label_phase=torchvision.utils.make_grid((train_result['label_phase']))
    grid_label_amp=torchvision.utils.make_grid((train_result['label_amp']))

    grid_label_BF=torchvision.utils.make_grid((train_result['input_BF']))
    grid_label_DF=torchvision.utils.make_grid((train_result['input_DF']))
    max_BF,min_BF=torch.max(train_result['input_BF']),torch.min(train_result['input_BF'])
    max_DF,min_DF=torch.max(train_result['input_DF']),torch.min(train_result['input_DF'])
    
    grid_output_BF=torchvision.utils.make_grid(torch.clamp(train_result['input_BF_estimated'],min=min_BF,max=max_BF))
    grid_output_DF=torchvision.utils.make_grid(torch.clamp(train_result['input_DF_estimated'],min=min_DF,max=max_DF))

    if mode =='train':
        wandb.log({
                    "Train Label Phase": wandb.Image((grid_label_phase)),
                    "Train Label Amp" : wandb.Image((grid_label_amp)),
                    "Train Label BF": wandb.Image((grid_label_BF)),
                    "Train Label DF": wandb.Image((grid_label_DF)),
                    "Train Output BF": wandb.Image((grid_output_BF)),
                    "Train Output DF": wandb.Image((grid_output_DF))
                    }, step=epoch)    
    elif mode =='test':
        wandb.log({
                "Test Label Phase": wandb.Image((grid_label_phase)),
                "Test Label Amp" : wandb.Image((grid_label_amp)),
                "Test Label BF": wandb.Image((grid_label_BF)),
                "Test Label DF": wandb.Image((grid_label_DF)),
                "Test Output BF": wandb.Image((grid_output_BF)),
                "Test Output DF": wandb.Image((grid_output_DF))
                }, step=epoch)    

    
    
def plot_train_disc(train_result,epoch):
    grid_output_stain=torchvision.utils.make_grid(torch.clamp(train_result['output_stain'],min=0,max=1))
    grid_label_phase=torchvision.utils.make_grid((train_result['label_phase']))
    grid_label_stain=torchvision.utils.make_grid(torch.clamp(train_result['label_stain'],min=0,max=1))
    
    grid_output_stain_unstained=torchvision.utils.make_grid(torch.clamp(train_result['output_stain_unstained'],min=0,max=1))
    grid_label_phase_unstained=torchvision.utils.make_grid((train_result['label_phase_unstained']))
    grid_label_stain_unstained=torchvision.utils.make_grid(torch.clamp(train_result['label_stain_unstained'],min=0,max=1))
    
    grid_phase_hm=torchvision.utils.make_grid((train_result['output_phase_hm']))
    grid_label_stain_hm=torchvision.utils.make_grid(torch.clamp(train_result['output_stain_hm'],min=0,max=1))
    
    
    
    wandb.log({
                "Train Label Phase (Stained)": wandb.Image((grid_label_phase)),
                "Train Label Stain (Stained)": wandb.Image((grid_label_stain)),
                "Train Output Stain (Stained)": wandb.Image((grid_output_stain)),
                "Train Output Phase HM (Stained)": wandb.Image((grid_phase_hm)),
                "Train Output Stain HM (Stained)": wandb.Image((grid_label_stain_hm)),
        
                "Train Label Phase (Unstained)": wandb.Image((grid_label_phase_unstained)),
                "Train Label Stain (Unstained)" : wandb.Image((grid_label_stain_unstained)),
                "Train Output Stain (Unstained)": wandb.Image((grid_output_stain_unstained))
                
                }, step=epoch)    
    
    

