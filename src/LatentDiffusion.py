import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from diffusers.models import AutoencoderKL
import matplotlib.pyplot as plt
import random

from .DenoisingDiffusionProcess import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Autoencoder(pl.LightningModule):
    def __init__(self, 
                 autoencoder: AutoencoderKL,
                 learning_rate = 1e-4):
        super().__init__()
        #self.save_hyperparameters()
        self.model = autoencoder
        self.learning_rate = learning_rate
    
    def forward(self, x):
        return self.model(x).sample
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def encode(self,input,mode=True):
        dist=self.model.encode(input).latent_dist
        if mode:
            return dist.mode()
        else:
            return dist.sample()
    
    def decode(self,input):
        return self.model.decode(input).sample
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon = self.model(x).sample

        loss = torch.nn.MSELoss()(x_recon, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_recon = self.model(x).sample

        loss = torch.nn.MSELoss()(x_recon, x)
        self.log("val_loss", loss, prog_bar=True)
    


class LatentDiffusion(pl.LightningModule):
    def __init__(self,
                 train_dataset,
                 autoencoder: Autoencoder,
                 valid_dataset=None,
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 batch_size=1,
                 schedule='linear',
                 lr=1e-4, 
                 warm_up_steps = 10000,
                 loss_fn = F.l1_loss):
        """
            This is a simplified version of Latent Diffusion        
        """        
        
        super().__init__()
        #self.save_hyperparameters()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.register_buffer('latent_scale_factor', torch.tensor(latent_scale_factor))
        self.batch_size=batch_size
        self.warm_up_steps = warm_up_steps
        self.loss_fn = loss_fn

        channels, h, w = train_dataset[0][0].shape   
        self.ae=autoencoder
        with torch.no_grad():
            self.latent_dim=self.ae.encode(torch.ones(1, channels, h, w).to(device)).shape[1]
        self.model=DenoisingDiffusionProcess(generated_channels=self.latent_dim,
                                             num_timesteps=num_timesteps,
                                             schedule=schedule)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.output_T(self.ae.decode(self.model(*args, **kwargs))/self.latent_scale_factor)
    
    def input_T(self, input):
        # By default, let the model accept samples in [0,1] range, and transform them automatically
        return (input.clip(0,1).mul_(2)).sub_(1)
    
    def output_T(self, input):
        # Inverse transform of model output from [-1,1] to [0,1] range
        return (input.add_(1)).div_(2)
    
    def training_step(self, batch, batch_idx):   
        
        latents=self.ae.encode(self.input_T(batch)).detach()*self.latent_scale_factor
        loss = self.model.p_loss(latents)
        
        self.log('train_loss',loss)

        return loss
            
    def validation_step(self, batch, batch_idx):     
        
        latents=self.ae.encode(self.input_T(batch)).detach()*self.latent_scale_factor
        loss = self.model.p_loss(latents)
        
        self.log('val_loss',loss)
        
        return loss
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)
    
    def val_dataloader(self):
        if self.valid_dataset is not None:
            return DataLoader(self.valid_dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=4)
        else:
            return None
    
    def configure_optimizers(self):
        return  torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=self.lr)
    
    def optimizer_step(self, epoch, 
                       batch_idx, 
                       optimizer,
                       optimizer_closure):
        
        # Linear Warm-up
        if self.trainer.global_step < self.warm_up_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warm_up_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        optimizer.step(closure=optimizer_closure)

        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
    
class LatentDiffusionConditional(LatentDiffusion):
    def __init__(self,
                 train_dataset,
                 autoencoder: Autoencoder,
                 valid_dataset=None,
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 schedule='linear',
                 batch_size=1,
                 lr=1e-4,
                 warm_up_steps = 10000,
                 loss_fn = F.l1_loss):
        pl.LightningModule.__init__(self)
        #self.save_hyperparameters()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.register_buffer('latent_scale_factor', torch.tensor(latent_scale_factor))
        self.batch_size=batch_size
        self.warm_up_steps = warm_up_steps
        self.loss_fn = loss_fn
        
        channels, h, w = train_dataset[0][0].shape 
        self.ae=autoencoder
        with torch.no_grad():
            self.latent_dim=self.ae.encode(torch.ones(1, channels, h, w).to(device)).shape[1]
        self.model=DenoisingDiffusionConditionalProcess(generated_channels=self.latent_dim,
                                                        condition_channels=self.latent_dim,
                                                        num_timesteps=num_timesteps,
                                                        schedule=schedule)
        

        
            
    @torch.no_grad()
    def forward(self,condition,*args,**kwargs):
        condition_latent=self.ae.encode(self.input_T(condition.to(self.device))).detach()*self.latent_scale_factor
        
        output_code=self.model(condition_latent,*args,**kwargs)/self.latent_scale_factor

        return self.output_T(self.ae.decode(output_code))
    
    def training_step(self, batch, batch_idx):   
        condition,output=batch
                
        with torch.no_grad():
            latents=self.ae.encode(self.input_T(output)).detach()*self.latent_scale_factor
            latents_condition=self.ae.encode(self.input_T(condition)).detach()*self.latent_scale_factor
        loss = self.model.p_loss(latents, latents_condition)
        
        self.log('train_loss',loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
            
    def validation_step(self, batch, batch_idx):     
        
        condition,output=batch
        
        with torch.no_grad():
            latents=self.ae.encode(self.input_T(output)).detach()*self.latent_scale_factor
            latents_condition=self.ae.encode(self.input_T(condition)).detach()*self.latent_scale_factor
        loss = self.model.p_loss(latents, latents_condition)
        
        self.log('val_loss',loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss