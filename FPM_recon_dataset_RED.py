from __future__ import print_function

import os
import pickle
import json
import glob
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import scipy.io
import natsort

class FPM_recon_dataset_RED(data.Dataset):
    """
    **Arguments**
    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Download the dataset if it's not available.
    """

    def __init__(
        self,
        root,
        mode='train',
        transform=None,
        target_transform=None,
    ):
        super(FPM_recon_dataset_RED, self).__init__()
#         self.root = os.path.expanduser(root[0])
#         if not os.path.exists(self.root):
#             os.mkdir(self.root)
            
#         self.root_stain = os.path.expanduser(root_stain[0])
#         if not os.path.exists(self.root_stain):
#             os.mkdir(self.root_stain)
            
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        
        for iSet in range(len(root)):
                
            files = glob.glob(root[iSet] + "*.mat", recursive= True)

            if iSet ==0:
                self.files=(natsort.natsorted(files))                   
            else:        
                self.files=np.concatenate((self.files,natsort.natsorted(files)))

                

        self.len = len(self.files)
        
    def __getitem__(self, index):
        image_url_raw = self.files[index]
        data_input = scipy.io.loadmat(image_url_raw)
        image=data_input['data']
        nSampR,_,nChannel=image.shape

        
        ## RAW IMAGE ##
        image=np.einsum('ijk->kij', image)     
        image=image.astype(np.float32)
        image=torch.from_numpy(image)
        
        ## Phase IMAGE ##  
        obj=data_input['obj']
        label_amp=obj[:,:,0]
        label_phase=obj[:,:,4]
        nObj,_=label_phase.shape

        amplitude = torch.from_numpy(label_amp.astype(np.float32))
        amplitude = amplitude.view(1,600,600)
        
        phase = torch.from_numpy(label_phase.astype(np.float32))
        phase = phase.view(1,600,600)
        
        ## Stain Image ##
        stain=obj[:,:,0:3]
        stain=torch.from_numpy(np.einsum('ijk->kij', stain)).type(torch.float32)
        
        ## normalization ##
        self.image=image/torch.max(image)  ## 16 bit normalize
        # self.image=image/(2**16)  ## 16 bit normalize
        self.amplitude = (amplitude)
        self.phase = (phase)
        self.stain=(stain)
        
        
        ## Illumination Angle ##       
        cropR=data_input['cropR'].astype(np.float16)

        self.cropR=torch.from_numpy(cropR).type(torch.int)
        self.label=torch.cat((self.stain,self.phase,self.amplitude),0)

            
        

        
        return (self.image, self.label, self.cropR)

    def __len__(self):
        return self.len



def alt_axis(X):
    return np.einsum('ijk->kij', X)        


def getAbs(x):
    return torch.sqrt(x[...,0]**2 + x[...,1]**2)

def getPhase(x):
    return torch.atan(x[...,1]/x[...,0])
 
    
def norm_minmax(x):
    x=x-torch.min(x)
    x=x/torch.max(x)
    return x

def norm_max(x):
    x=x/torch.max(x)   
    return x

    