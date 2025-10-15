import torch
import torch.nn as nn
import torch.nn.functional as F
# import rcan_common as common
# from FPM_model_parts import *
from torch.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np
import matplotlib.pyplot as plt


    
    
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


## Channel Attention (CA) Layer
# class CALayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(CALayer, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#                 nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#                 nn.Sigmoid()
#         )

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y

class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1, insn = True):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if insn : modules_body.append(nn.InstanceNorm2d(n_feat))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        # modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res



## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res



class RCAN_recon(nn.Module):
    def __init__(self,  in_shape, n_bf, n_df, conv=default_conv):  #common.default_conv
        super(RCAN_recon, self).__init__()
        channels, height, width = in_shape
        n_resgroups = 4  #opt.RCAN_n_groups ## 10 ,4
        n_resblocks = 8  #opt.RCAN_n_blocks
        
        n_multiplex = n_bf + n_df

        n_img = 193
        
        
        n_feats = 32 #args.n_feats         ## 64
        kernel_size = 3
        reduction = 16 #args.reduction     ## 16
            
        scale = 3
        
        
        [self.led_list,dfi] = led_visualize_init()
        dfi = np.expand_dims(dfi, axis=1)
        mask_led_bf = np.repeat(1-dfi, n_bf, axis=1)
        mask_led_df = np.repeat(dfi, n_df, axis=1)
        mask_led = np.concatenate((mask_led_bf,mask_led_df), axis=1)
        self.mask_led = torch.from_numpy(mask_led).permute(1,0)
        
        
        torch.manual_seed(1)
        self.led_weights_init=torch.rand(n_multiplex,n_img)*self.mask_led
        self.led_weights=nn.Parameter(self.led_weights_init.clone())
        
        
        # self.led_weights=(self.led_weights_init.clone())
        # self.led_weights.requires_grad_(True)
        

        # scale = args.scale[0]          ## 4
        act = nn.ReLU(True)            
        
        # RGB mean for DIV2K
        
        # define head module
#         modules_head = [common.CoordConvs_(193, 193, 1)]
#         modules_head.append(conv(193, n_feats, kernel_size))
        modules_head=[(conv(n_multiplex, n_feats, kernel_size))]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]


        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats,'shuffle', act=False),
            conv(n_feats, 2, kernel_size)]
        

        

        
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.fc = nn.Sequential(
            nn.Conv2d(8, 1, 3, bias=False),
            nn.Flatten(),
            # nn.Linear(2048, 1024), # @@ 2048????
            nn.Linear(39204, 2048), 
            nn.LeakyReLU(0.1,inplace=False),
            nn.Linear(2048, 1024),
            nn.Sigmoid(),
        )

        self.quantBit=4
    def quantize_weight(self):
        # led_weights_quantize=torch.zeros_like(self.led_weights)
        led_weights_quantize=(self.led_weights>1)*self.led_weights+(self.led_weights<=0)*self.led_weights
        bias=7
        beta=256
        # bias=4
        # beta=128
        # bias=16;
        # beta=512;
        # self.led_weights.clamp(min=0.0,max=1.0)
        for i_quantize in range(2**self.quantBit):
            
            mask_quantize=(self.led_weights>((i_quantize)/(2**self.quantBit)))*(self.led_weights<=((i_quantize+1)/(2**self.quantBit)))
            mask_quantize=mask_quantize.to(torch.float32)
            
            
            weight_quantize=((1/(1+torch.exp(-(beta*(self.led_weights-i_quantize/(2**self.quantBit))-bias))))+(i_quantize))/(2**self.quantBit)
            
            # weight_quantize=((1/(1+torch.exp(-(beta*(self.led_weights)-bias))))+(i_quantize))/(2**self.quantBit)
            led_weights_quantize=led_weights_quantize+weight_quantize*(mask_quantize)
            
        plt.figure(2)
        plt.imshow(led_weights_quantize.cpu().detach().numpy())
        plt.clim([0,1])
        plt.show()

            
        
        return led_weights_quantize

        
        
        
    def led_opt(self,x):

        led_weights_current = self.led_weights.clamp(min=0.0,max=1.0)    
        return torch.matmul(led_weights_current*self.mask_led,x.permute(3,0,1,2)).permute(1,2,3,0)    

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.led_opt(x)
        x = x / torch.amax(x,dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)

        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x
    
class RCAN_recon_inference(nn.Module):
    def __init__(self,  in_shape, n_bf, n_df, conv=default_conv):  #common.default_conv
        super(RCAN_recon_inference, self).__init__()
        channels, height, width = in_shape
        n_resgroups = 4  #opt.RCAN_n_groups ## 10 ,4
        n_resblocks = 8  #opt.RCAN_n_blocks
        
        n_multiplex = n_bf + n_df

        n_img = 193
        
        
        n_feats = 32 #args.n_feats         ## 64
        kernel_size = 3
        reduction = 16 #args.reduction     ## 16
            
        scale = 3
        
        
        [self.led_list,dfi] = led_visualize_init()
        dfi = np.expand_dims(dfi, axis=1)
        mask_led_bf = np.repeat(1-dfi, n_bf, axis=1)
        mask_led_df = np.repeat(dfi, n_df, axis=1)
        mask_led = np.concatenate((mask_led_bf,mask_led_df), axis=1)
        self.mask_led = torch.from_numpy(mask_led).permute(1,0)
        
        
        torch.manual_seed(1)
        self.led_weights_init=torch.rand(n_multiplex,n_img)*self.mask_led
        self.led_weights=nn.Parameter(self.led_weights_init.clone())
        
        
        # self.led_weights=(self.led_weights_init.clone())
        # self.led_weights.requires_grad_(True)
        

        # scale = args.scale[0]          ## 4
        act = nn.ReLU(True)            
        
        # RGB mean for DIV2K
        
        # define head module
#         modules_head = [common.CoordConvs_(193, 193, 1)]
#         modules_head.append(conv(193, n_feats, kernel_size))
        modules_head=[(conv(n_multiplex, n_feats, kernel_size))]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]


        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats,'shuffle', act=False),
            conv(n_feats, 2, kernel_size)]
        

        

        
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.fc = nn.Sequential(
            nn.Conv2d(8, 1, 3, bias=False),
            nn.Flatten(),
            # nn.Linear(2048, 1024), # @@ 2048????
            nn.Linear(39204, 2048), 
            nn.LeakyReLU(0.1,inplace=False),
            nn.Linear(2048, 1024),
            nn.Sigmoid(),
        )

        self.quantBit=4
    def quantize_weight(self):
        # led_weights_quantize=torch.zeros_like(self.led_weights)
        led_weights_quantize=(self.led_weights>1)*self.led_weights+(self.led_weights<=0)*self.led_weights
        bias=7
        beta=256
        # bias=4
        # beta=128
        # bias=16;
        # beta=512;
        # self.led_weights.clamp(min=0.0,max=1.0)
        for i_quantize in range(2**self.quantBit):
            
            mask_quantize=(self.led_weights>((i_quantize)/(2**self.quantBit)))*(self.led_weights<=((i_quantize+1)/(2**self.quantBit)))
            mask_quantize=mask_quantize.to(torch.float32)
            
            
            weight_quantize=((1/(1+torch.exp(-(beta*(self.led_weights-i_quantize/(2**self.quantBit))-bias))))+(i_quantize))/(2**self.quantBit)
            
            # weight_quantize=((1/(1+torch.exp(-(beta*(self.led_weights)-bias))))+(i_quantize))/(2**self.quantBit)
            led_weights_quantize=led_weights_quantize+weight_quantize*(mask_quantize)
            
        plt.figure(2)
        plt.imshow(led_weights_quantize.cpu().detach().numpy())
        plt.clim([0,1])
        plt.show()

            
        
        return led_weights_quantize

        
        
        
    def led_opt(self,x):

        led_weights_current = self.led_weights.clamp(min=0.0,max=1.0)    
        return torch.matmul(led_weights_current*self.mask_led,x.permute(3,0,1,2)).permute(1,2,3,0)    

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.led_opt(x)
        x_opt = x.clone()
        x = x / torch.amax(x,dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)

        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x, x_opt
    

    
    
    
class RCAN_recon_nonorm(nn.Module):
    def __init__(self,  in_shape, conv=default_conv):  #common.default_conv
        super(RCAN_recon_nonorm, self).__init__()
        channels, height, width = in_shape
        n_resgroups =4  #opt.RCAN_n_groups ## 10 ,4
        n_resblocks =8  #opt.RCAN_n_blocks
        
        n_bf=3
        n_df=13
        n_multiplex=n_bf+n_df
        n_img=193
        
        
        
        n_feats = n_multiplex*2 #args.n_feats         ## 64
        kernel_size = 3
        reduction = 16 #args.reduction     ## 16
            
        scale = 3
        
        
        
        [self.led_list,dfi]=led_visualize_init()
        dfi=np.expand_dims(dfi, axis=1)
        mask_led_bf=np.repeat(1-dfi, n_bf, axis=1)
        mask_led_df=np.repeat(dfi, n_df, axis=1)
        mask_led=np.concatenate((mask_led_bf,mask_led_df), axis=1)
        self.mask_led=torch.from_numpy(mask_led).permute(1,0)
        
        
        torch.manual_seed(1)
        self.led_weights_init=torch.rand(n_multiplex,n_img)*self.mask_led
        self.led_weights=nn.Parameter(self.led_weights_init.clone())
        
        
        # self.led_weights=(self.led_weights_init.clone())
        # self.led_weights.requires_grad_(True)
        

        # scale = args.scale[0]          ## 4
        act = nn.ReLU(True)            
        
        # RGB mean for DIV2K
        
        # define head module
#         modules_head = [common.CoordConvs_(193, 193, 1)]
#         modules_head.append(conv(193, n_feats, kernel_size))
        modules_head=[(conv(n_multiplex, n_feats, kernel_size))]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]


        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats,'shuffle' ,  act=False),
            conv(n_feats, 2, kernel_size)]

    
            
        
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.fc = nn.Sequential(
            nn.Conv2d(8, 1, 3, bias=False),
            nn.Flatten(),
            # nn.Linear(2048, 1024), # @@ 2048????
            nn.Linear(39204, 2048), 
            nn.LeakyReLU(0.1,inplace=False),
            nn.Linear(2048, 1024),
            nn.Sigmoid(),
        )

        self.quantBit=4
    def quantize_weight(self):
        # led_weights_quantize=torch.zeros_like(self.led_weights)
        led_weights_quantize=(self.led_weights>1)*self.led_weights+(self.led_weights<=0)*self.led_weights
        bias=7
        beta=256
        # bias=4
        # beta=128
        # bias=16;
        # beta=512;
        # self.led_weights.clamp(min=0.0,max=1.0)
        for i_quantize in range(2**self.quantBit):
            
            mask_quantize=(self.led_weights>((i_quantize)/(2**self.quantBit)))*(self.led_weights<=((i_quantize+1)/(2**self.quantBit)))
            mask_quantize=mask_quantize.to(torch.float32)
            
            
            weight_quantize=((1/(1+torch.exp(-(beta*(self.led_weights-i_quantize/(2**self.quantBit))-bias))))+(i_quantize))/(2**self.quantBit)
            
            # weight_quantize=((1/(1+torch.exp(-(beta*(self.led_weights)-bias))))+(i_quantize))/(2**self.quantBit)
            led_weights_quantize=led_weights_quantize+weight_quantize*(mask_quantize)
            
        plt.figure(2)
        plt.imshow(led_weights_quantize.cpu().detach().numpy())
        plt.clim([0,1])
        plt.show()
        #     print(torch.max(weight_quantize))
        #     print(torch.min(weight_quantize))
        # print(1)
        # print(torch.max(self.led_weights))
        # print(torch.min(self.led_weights))      
        # print(torch.max(led_weights_quantize))
        # print(torch.min(led_weights_quantize))
        # print(2)
            
        
        return led_weights_quantize

        
        
        
    def led_opt(self,x):
            # return torch.matmul(self.led_weights,x.permute(3,0,1,2)).permute(1,2,3,0)    
            # led_weights=torch.round(self.led_weights*(2**self.quantBit))/(2**self.quantBit)
        # led_weights_quantize=self.quantize_weight()
        # return torch.matmul(led_weights_quantize*self.mask_led,x.permute(3,0,1,2)).permute(1,2,3,0)    
            # self.quantize_weight()
        # self.led_weights.clamp(min=0.0,max=1.0)
        # print(self.led_weights.dtype)
        # print(self.led_weights.clamp(min=0.0,max=1.0).dtype)
        led_weights = self.led_weights.clamp(min=0.0,max=1.0)
        # print(torch.min(self.led_weights))
        # print(torch.max(self.led_weights))
              
        return torch.matmul(led_weights*self.mask_led,x.permute(3,0,1,2)).permute(1,2,3,0)    

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.led_opt(x)
        imgs_opt = x
        x = x / torch.amax(x,dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)

        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x, imgs_opt
    
    
    
    
    
    


class Forward_Crop(nn.Module):
    def __init__(self):
        super().__init__()
        
        

        
    def forward(self,amp,phase,cropR,low_pass_filt):

        nBatch,nChannel,_=cropR.shape

        obj=amp*torch.exp(1j*phase)
        objFT=fftshift(fft2(obj,dim=(-1,-2)),dim=(-1,-2))
        cropR=cropR
        
        # highResFT_crop=objFT[:,:,cropR[:,:,0]:cropR[:,:,1]+1,cropR[:,:,2]:cropR[:,:,3]+1]*self.low_pass_filt
        highResFT_crop=torch.zeros((nBatch,nChannel,200,200),dtype=torch.cfloat)
        for iBatch in range(nBatch):
            for iImg in range (nChannel):
                    highResFT_crop[iBatch,iImg,:,:]=objFT[iBatch,cropR[iBatch,iImg,0]:cropR[iBatch,iImg,1]+1,cropR[iBatch,iImg,2]:cropR[iBatch,iImg,3]+1]*low_pass_filt
        
        
        highRes_crop=ifft2(ifftshift(highResFT_crop,dim=(-1,-2)),dim=(-1,-2))
        
        # out=torch.cat((torch.abs(highRes_crop),torch.angle(highRes_crop),0))
        out=torch.abs(highRes_crop).type(torch.FloatTensor)
        
        
        return out


class Post_Pupil_Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.PN_mode = 'amp'
        self.base_channel = 193
        self.PN_structure = 'unet'
        if self.PN_mode == 'amp':
            
            if self.PN_structure == 'convs':
                self.convs = nn.Sequential(
                    
                    # common.CoordConvs_(opt.nUsed, 2*opt.nUsed, 1),
                    DoubleConv_residual(self.base_channel, 2*self.base_channel),
                    DoubleConv_residual(2*self.base_channel, 4*self.base_channel),
                    DoubleConv_residual(4*self.base_channel, 4*self.base_channel),
                    # DoubleConv_residual(4*self.base_channel, 4*self.base_channel),
                    # DoubleConv_residual(8*self.base_channel, 16*self.base_channel),
                    nn.Conv2d(4*self.base_channel, self.base_channel, kernel_size=1)   )
                
                
            elif self.PN_structure == 'unet':
                
#                 basebase= self.base_channel*2
                basebase= 64
                self.inc = DoubleConv_unet(self.base_channel, 2*basebase)
                self.down1 = Down_unet(2*basebase, 4*basebase)
                self.down2 = Down_unet(4*basebase, 8*basebase)
                self.down3 = Down_unet(8*basebase, 16*basebase)
                factor = 1
                self.down4 = Down(16*basebase, 32*basebase // factor)
                self.up1 = Up_unet(32*basebase, 16*basebase // factor)
                self.up2 = Up_unet(16*basebase, 8*basebase // factor)
                self.up3 = Up_unet(8*basebase, 4*basebase // factor)
                self.up4 = Up_unet(4*basebase, 2*basebase)
                self.outc = nn.Sequential(
                # nn.Conv2d(2*basebase, 2*basebase, stride=1 ,  kernel_size=3 , padding =1),
                # nn.ReLU(inplace = True),
                nn.Conv2d(2*basebase, 2*basebase, stride=1 ,  kernel_size=3 , padding =1),
                nn.ReLU(inplace = True),
                nn.Conv2d(2*basebase, self.base_channel, stride=1 ,  kernel_size=1 , padding =0)
                )
                    
                

    def forward(self, x):
        if self.PN_structure == 'unet':
            # print(x.shape,'x__shape')
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x51 = self.up1(x5, x4)
            x41 = self.up2(x51, x3)
            x31 = self.up3(x41, x2)
            x21 = self.up4(x31, x1)        
            x = self.outc(x21)
            
            
            
        elif self.PN_structure == 'AUTO' : 
            
            xshape = x.shape
            xf = nn.Flatten()(x)
            x = self.denses1(xf)
            x= x.reshape((xshape[0], xshape[1] , xshape[2] , xshape[3]) ) 
            # x = self.denses2(x)
            
            
        else:
            x = self.convs(x)
            
            
            
            
        return x

    
def generate_transfer_function():
    nPixel=200
    mag=3.3
    pSize=2.4e-6/mag
    wavelength=0.6244e-6
    NA=0.135
    coherent_transfer_function = np.zeros((nPixel,nPixel),dtype='cfloat')
    k0 = 2 * np.pi / wavelength
    cutoff_frequency_x = NA * k0
    cutoff_frequency_y = NA * k0
    
    for v in range(nPixel):
        for h in range(nPixel):
            # x,y coordinate on k-space
            kx = (v-nPixel/2) / nPixel / pSize * 2 * np.pi
            ky = (h-nPixel/2) / nPixel / pSize * 2 * np.pi
            # Elliptical aperture
            if ((kx)**2/cutoff_frequency_x**2+(ky)**2/cutoff_frequency_y**2<=1):
                coherent_transfer_function[h,v] = 1
    
    return torch.tensor(coherent_transfer_function).unsqueeze(0).unsqueeze(0)



def led_visualize_init():
        NA = 0.13           
        sampling_rad=8
        arraySize_led = 32
        ds_led = 4
        z_led = 80
        center_x=16
        center_y=16

        grid_array=np.arange(1,arraySize_led+1)
        rad_array_x,rad_array_y=np.meshgrid(grid_array,grid_array)
        rad_array_x=rad_array_x-center_x
        rad_array_y=rad_array_y-center_y

        litCoord=(rad_array_x**2+rad_array_y**2)<sampling_rad**2
        row,col=np.where(litCoord==1)
        xlocation=rad_array_x[row,col]*ds_led
        ylocation=rad_array_y[row,col]*ds_led
        nImg=xlocation.shape[0];

        kx_relative = -np.sin(np.arctan(xlocation/z_led));  
        ky_relative = -np.sin(np.arctan(ylocation/z_led)); 

        led_list=np.array([kx_relative,ky_relative])
        illum=np.sqrt(kx_relative**2+ky_relative**2)
        dfi=illum>NA
        return led_list, dfi  
    
    
def get_device_from_var(ref):
    return f'{ref.device.type}:{ref.device.index}'    






class RCAN_stain(nn.Module):
    def __init__(self,  in_shape, n_bf, n_df, conv=default_conv):  #common.default_conv
        super(RCAN_stain, self).__init__()
        channels, height, width = in_shape
        n_resgroups =4  #opt.RCAN_n_groups ## 10 ,4
        n_resblocks =8  #opt.RCAN_n_blocks
        
        n_multiplex=n_bf+n_df
        n_img=193
        
        
        
        # n_feats = n_multiplex*2 #args.n_feats         ## 64
        n_feats = 32 #args.n_feats         ## 64
        kernel_size = 3
        reduction = 16 #args.reduction     ## 16
            
        scale = 3
        
        
        
        [self.led_list,dfi]=led_visualize_init()
        dfi=np.expand_dims(dfi, axis=1)
        mask_led_bf=np.repeat(1-dfi, n_bf, axis=1)
        mask_led_df=np.repeat(dfi, n_df, axis=1)
        mask_led=np.concatenate((mask_led_bf,mask_led_df), axis=1)
        self.mask_led=torch.from_numpy(mask_led).permute(1,0)
        
        
        torch.manual_seed(1)
        self.led_weights_init=torch.rand(n_multiplex,n_img)*self.mask_led
        self.led_weights=nn.Parameter(self.led_weights_init.clone())
        
        
        # self.led_weights=(self.led_weights_init.clone())
        # self.led_weights.requires_grad_(True)
        

        # scale = args.scale[0]          ## 4
        act = nn.ReLU(True)            
        
        # RGB mean for DIV2K
        
        # define head module
#         modules_head = [common.CoordConvs_(193, 193, 1)]
#         modules_head.append(conv(193, n_feats, kernel_size))
        modules_head=[(conv(n_multiplex, n_feats, kernel_size))]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]


        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats,'shuffle' ,  act=False),
            conv(n_feats, 3, kernel_size)]

    
            
        
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.fc = nn.Sequential(
            nn.Conv2d(8, 1, 3, bias=False),
            nn.Flatten(),
            # nn.Linear(2048, 1024), # @@ 2048????
            nn.Linear(39204, 2048), 
            nn.LeakyReLU(0.1,inplace=False),
            nn.Linear(2048, 1024),
            nn.Sigmoid(),
        )

        self.quantBit=4
    def quantize_weight(self):
        # led_weights_quantize=torch.zeros_like(self.led_weights)
        led_weights_quantize=(self.led_weights>1)*self.led_weights+(self.led_weights<=0)*self.led_weights
        bias=7
        beta=256
        # bias=4
        # beta=128
        # bias=16;
        # beta=512;
        # self.led_weights.clamp(min=0.0,max=1.0)
        for i_quantize in range(2**self.quantBit):
            
            mask_quantize=(self.led_weights>((i_quantize)/(2**self.quantBit)))*(self.led_weights<=((i_quantize+1)/(2**self.quantBit)))
            mask_quantize=mask_quantize.to(torch.float32)
            
            
            weight_quantize=((1/(1+torch.exp(-(beta*(self.led_weights-i_quantize/(2**self.quantBit))-bias))))+(i_quantize))/(2**self.quantBit)
            
            # weight_quantize=((1/(1+torch.exp(-(beta*(self.led_weights)-bias))))+(i_quantize))/(2**self.quantBit)
            led_weights_quantize=led_weights_quantize+weight_quantize*(mask_quantize)
            
        plt.figure(2)
        plt.imshow(led_weights_quantize.cpu().detach().numpy())
        plt.clim([0,1])
        plt.show()

        return led_weights_quantize

        
        
        
    def led_opt(self,x):

        led_weights_current = self.led_weights.clamp(min=0.0,max=1.0)    
        return torch.matmul(led_weights_current*self.mask_led,x.permute(3,0,1,2)).permute(1,2,3,0)    

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.led_opt(x)
        x = x / torch.amax(x,dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)

        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x
    
    
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt




class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)



def input_mapping(x, B): 
  
    x_proj = (2.*np.pi*x).permute(0,2,3,1).contiguous() @ B.T
    #x_proj = torch.matmul( (2.*np.pi*x).permute(0,2,3,1).contiguous() , B.T)
    
    
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1).permute(0,3,1,2).contiguous()




class AddFFE(nn.Module):

    def __init__(self, mapping_size= 64 , scale=1):
        super().__init__()
        self.mapping_size=  mapping_size
        self.scale = scale

        self.rand_key = [0,0]
    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)
        
        
        B_gauss = torch.normal(0,1, size=(self.mapping_size, 2), device = 'cuda')
        B_dict = B_gauss * self.scale
        ss_gauss_1 =input_mapping(ret,B_dict)
    
        ss_gauss_1=  torch.cat([input_tensor, ss_gauss_1],dim=1)
        
        return ss_gauss_1


class FFE(nn.Module):

    def __init__(self, in_channels, out_channels, mapping_size=128, scale=1, **kwargs):
        super().__init__()
        self.addffe = AddFFE(mapping_size=mapping_size , scale = scale)
        in_size = in_channels+mapping_size*2
        self.conv = nn.Sequential(
                                  nn.Conv2d(in_size, in_size, kernel_size= 3, padding =1), 
                                  
                                  nn.ReLU(True),)
                                   #nn.Conv2d(in_size, in_size, kernel_size= 3 , padding = 1), 
                                   #nn.ReLU(True),
                                   #nn.Conv2d(in_size, out_channels, **kwargs), )

    def forward(self, x):

        
        ret = self.addffe(x)
        ret = self.conv(ret)
        return ret




class AddCoords(nn.Module):

    def __init__(self, with_r=True):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret




class AddCoords_cust(nn.Module):

    def __init__(self, fact, with_r=True ):
        super().__init__()
        self.fact = fact
        self.with_r = with_r
        
    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()
        
        
        xx_channel = torch.arange(x_dim*self.fact)[::self.fact].repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim*self.fact)[::self.fact].repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret











class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=True, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):

        ret = self.addcoords(x)

        ret = self.conv(ret)
        return ret


class CoordConv_cust(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=True, **kwargs):
        super().__init__()
        self.addcoords1 = AddCoords_cust(1, with_r=with_r)
        self.addcoords2 = AddCoords_cust(2, with_r=with_r)
        self.addcoords3= AddCoords_cust(3, with_r=with_r)
        self.addcoords4= AddCoords_cust(4, with_r=with_r)
        self.addcoords5= AddCoords_cust(8, with_r=with_r)
        
        in_size = in_channels+10
        
        if with_r:
            in_size += 5
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):

        ret = self.addcoords1(x)
        ret = self.addcoords2(ret)
        ret = self.addcoords3(ret)
        ret = self.addcoords4(ret)
        ret = self.addcoords5(ret)

        ret = self.conv(ret)
        return ret




class CoordConvs_(nn.Module):
    def __init__(self, in_features, out_features,  kernel_size  ,  stride = 1 , bias = True):
        super(CoordConvs_, self).__init__()

        self.conv_block = nn.Sequential(
                        CoordConv(in_features, out_features, kernel_size  =kernel_size  ,stride = stride, padding =kernel_size//2  , bias = bias),
                        nn.InstanceNorm2d(out_features),
                        # nn.ReLU(inplace=True),
                        # torch.sin(),
                        # 
                        )
    def forward(self, x):
        
        
        return  self.conv_block(x)
    


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res



class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, mode, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                
                if mode == 'shuffle':
                    m.append(nn.PixelShuffle(2))
                    if bn: m.append(nn.BatchNorm2d(n_feat))
                    if act: m.append(act())
                elif mode == 'transpose' :
                    m.append(   nn.ConvTranspose2d(4*n_feat, n_feat, kernel_size=3, padding=1 , stride = 2 , output_padding = 1 )    )
                    if bn: m.append(nn.BatchNorm2d(n_feat))
                    if act: m.append(act())
                
                elif mode == 'nearest' :
                    m.append(   nn.Upsample2d(scale_factor =  2, mode ='nearest'  )  )
                    m.append(   conv(4*n_feat,  n_feat, 3, bias)  )
                    if bn: m.append(nn.BatchNorm2d(n_feat))
                    if act: m.append(act())
                    
                    
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            
            if mode == 'shuffle':
                m.append(nn.PixelShuffle(3))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
                
            elif mode == 'transpose' : 
                m.append(   nn.ConvTranspose2d(9*n_feat, n_feat, kernel_size=3, padding=0 , stride = 3 , output_padding = 0 )    )
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
                
            elif mode == 'nearest' : 
                m.append(   nn.Upsample(scale_factor =  3, mode ='nearest'  )  )
                m.append(   conv(9*n_feat,  n_feat, 3, bias)  )
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
                    
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)    
    
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BN_EPS = 1e-4


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=(3, 3)):
        super(StackEncoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x):
        x = self.encode(x)
        x_small = F.max_pool2d(x, kernel_size=2, stride=2)
        return x, x_small


class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(StackDecoder, self).__init__()
        padding = (kernel_size - 1) // 2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x, down_tensor):
        _, channels, height, width = down_tensor.size()
        x = F.upsample(x, size=(height, width), mode='bilinear')
        x = torch.cat([x, down_tensor], 1)
        x = self.decode(x)
        return x
    
class StackDecoderFinal(nn.Module):
    def __init__(self, x_big_channels, y_channels, kernel_size=3):
        super(StackDecoderFinal, self).__init__()
        padding = (kernel_size - 1) // 2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x):
        _, channels, height, width = x.size()
        x = F.upsample(x, size=(height*3, width*3), mode='bilinear')
        x = self.decode(x)
        return x    
    
    

    # 32x32
class UNet(nn.Module):
    def __init__(self, in_shape):
        super(UNet, self).__init__()
        channels, height, width = in_shape

        self.down1 = StackEncoder(1, 24, kernel_size=3) ;# 256
        self.down2 = StackEncoder(24, 64, kernel_size=3)  # 128
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # 64
        self.down4 = StackEncoder(128, 256, kernel_size=3)  # 32
        self.down5 = StackEncoder(256, 512, kernel_size=3)  # 16
        

        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 128
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)  # 256
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)  # 512
        self.classify = nn.Conv2d(24, 3, kernel_size=1, bias=True)

        self.center = nn.Sequential(ConvBnRelu2d(512, 512, kernel_size=3, padding=1))


    def forward(self, x):
        out = x; 
   
        down1, out = self.down1(out); 
        down2, out = self.down2(out); 
        down3, out = self.down3(out); 
        down4, out = self.down4(out); 
        down5, out = self.down5(out); 

        out = self.center(out)
        out = self.up5(out, down5); 
        out = self.up4(out, down4); 
        out = self.up3(out, down3); 
        out = self.up2(out, down2); 
        out = self.up1(out, down1); 

        out = self.classify(out); 
#         print(out.shape)
#         print(2)
        return out


class UNet_contrastive(nn.Module):
    def __init__(self, in_shape):
        super(UNet_contrastive, self).__init__()
        channels, height, width = in_shape
        dim_in = 512
        feat_dim = 64
        
        self.down1 = StackEncoder(1, 32, kernel_size=3) ;# 256
        self.down2 = StackEncoder(32, 64, kernel_size=3)  # 128
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # 64
        self.down4 = StackEncoder(128, 256, kernel_size=3)  # 32
        self.down5 = StackEncoder(256, 512, kernel_size=3)  # 16
        
        
        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 128
        self.up2 = StackDecoder(64, 64, 32, kernel_size=3)  # 256
        self.up1 = StackDecoder(32, 32, 24, kernel_size=3)  # 512
        self.classify = nn.Conv2d(24, 3, kernel_size=1, bias=True)
        
            
        self.center = nn.Sequential(ConvBnRelu2d(512, 512, kernel_size=3, padding=1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self, x):
        out = x; 
   
        down1, out = self.down1(out); 
        feat1=self.avgpool(out)
        down2, out = self.down2(out); 
        feat2=self.avgpool(out)
        down3, out = self.down3(out); 
        feat3=self.avgpool(out)
        down4, out = self.down4(out); 
        feat4=self.avgpool(out)
        down5, out = self.down5(out); 
        feat5=self.avgpool(out)
        
        feats = torch.cat([feat1,feat2,feat3,feat4,feat5],dim=1)
        # feats=self.avgpool(out)
        # feats=F.normalize(self.mlp(feats), dim=1)
        
        out = self.center(out)
        
        
        out = self.up5(out, down5); 
        out = self.up4(out, down4); 
        out = self.up3(out, down3); 
        out = self.up2(out, down2); 
        out = self.up1(out, down1); 

        out = self.classify(out); 
#         print(out.shape)
#         print(2)
        return out,feats

class MLP_contrastive(nn.Module):
    def __init__(self, in_shape, feat_dim):
        super(MLP_contrastive, self).__init__()
        dim_in = 992 #672
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),
                nn.Sigmoid()
            )
        self.mlp2 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(dim_in, feat_dim),
            )
        self.mlp3 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(dim_in, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, feat_dim)
            )
    def forward(self, x):

        
        # feats=F.normalize(self.mlp(x), dim=1)
        feats=(self.mlp3(x))
        
        
        
        
        return feats    
    
    
    


class UNet_recon(nn.Module):
    def __init__(self, in_shape):
        super(UNet_recon, self).__init__()
        channels, height, width = in_shape
        
        n_bf=3
        n_df=13
        n_multiplex=n_bf+n_df
        n_img=193
        dim_in = 512
        feat_dim = 128
           
        
        
        [self.led_list,dfi]=led_visualize_init()
        dfi=np.expand_dims(dfi, axis=1)
        mask_led_bf=np.repeat(1-dfi, n_bf, axis=1)
        mask_led_df=np.repeat(dfi, n_df, axis=1)
        mask_led=np.concatenate((mask_led_bf,mask_led_df), axis=1)
        self.mask_led=torch.from_numpy(mask_led).permute(1,0)
        torch.manual_seed(1)
        self.led_weights_init=torch.rand(n_multiplex,n_img)*self.mask_led
        self.led_weights=nn.Parameter(self.led_weights_init.clone())
        
        self.down1 = StackEncoder(n_multiplex, 32, kernel_size=3) ;# 256
        self.down2 = StackEncoder(32, 64, kernel_size=3)  # 128
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # 64
        self.down4 = StackEncoder(128, 256, kernel_size=3)  # 32
        self.down5 = StackEncoder(256, 512, kernel_size=3)  # 16
        

        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 128
        self.up2 = StackDecoder(64, 64, 32, kernel_size=3)  # 256
        
        self.up1 = StackDecoder(32, 32, 24, kernel_size=3)  # 512
        self.up0 = StackDecoderFinal(16, 24, kernel_size=3)
        
        self.classify = nn.Conv2d(24, 3, kernel_size=1, bias=True)

        self.center = nn.Sequential(ConvBnRelu2d(512, 512, kernel_size=3, padding=1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        
        
        
    def led_opt(self,x):
            # return torch.matmul(self.led_weights,x.permute(3,0,1,2)).permute(1,2,3,0)    
            return torch.matmul(self.led_weights*self.mask_led,x.permute(3,0,1,2)).permute(1,2,3,0)    


        
    def forward(self, x):
        out = x; 
        out=self.led_opt(out)
        down1, out = self.down1(out) 
        down2, out = self.down2(out) 
        down3, out = self.down3(out) 
        down4, out = self.down4(out) 
        down5, out = self.down5(out) 
        out = self.center(out)
        feats=self.avgpool(out)
        # feats=F.normalize(self.mlp(feats), dim=1)
        feats=self.mlp(feats)
        
        out = self.up5(out, down5) 
        out = self.up4(out, down4) 
        out = self.up3(out, down3) 
        out = self.up2(out, down2) 
        out = self.up1(out, down1)
        out = self.up0(out)

        out = self.classify(out) 
#         print(out.shape)
#         print(2)
        return out, feats


def led_visualize_init():
        NA = 0.135           
        sampling_rad=8
        arraySize_led = 32
        ds_led = 4
        z_led = 80
        center_x=16
        center_y=16

        grid_array=np.arange(1,arraySize_led+1)
        rad_array_x,rad_array_y=np.meshgrid(grid_array,grid_array)
        rad_array_x=rad_array_x-center_x
        rad_array_y=rad_array_y-center_y

        litCoord=(rad_array_x**2+rad_array_y**2)<sampling_rad**2
        row,col=np.where(litCoord==1)
        xlocation=rad_array_x[row,col]*ds_led
        ylocation=rad_array_y[row,col]*ds_led
        nImg=xlocation.shape[0];

        kx_relative = -np.sin(np.arctan(xlocation/z_led));  
        ky_relative = -np.sin(np.arctan(ylocation/z_led)); 

        led_list=np.array([kx_relative,ky_relative])
        illum=np.sqrt(kx_relative**2+ky_relative**2)
        dfi=illum>NA
        return led_list, dfi  
    