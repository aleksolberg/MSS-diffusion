from typing import Any, Callable, Optional, Union
import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .DenoisingDiffusionProcess import *

class PixelDiffusion(pl.LightningModule):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 num_timesteps=1000,
                 batch_size=1,
                 lr=1e-3,
                 warm_up_steps = 10000,
                 schedule='linear',
                 loss_fn = F.mse_loss):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.batch_size=batch_size
        self.warm_up_steps = warm_up_steps
        channels, h, w = train_dataset[0][0].shape   
        
        self.model=DenoisingDiffusionProcess(generated_channels=channels,
                                             num_timesteps=num_timesteps,
                                             schedule=schedule,
                                             loss_fn=loss_fn)

    @torch.no_grad()
    def forward(self,*args,**kwargs):
        return self.output_T(self.model(*args,**kwargs))
    
    def input_T(self, input):
        # By default, let the model accept samples in [0,1] range, and transform them automatically
        return (input.clip(0,1).mul_(2)).sub_(1)
    
    def output_T(self, input):
        # Inverse transform of model output from [-1,1] to [0,1] range
        return (input.add_(1)).div_(2)
    
    def training_step(self, batch, batch_idx):   
        images=batch
        loss = self.model.p_loss(self.input_T(images))
        
        self.log('train_loss',loss, prog_bar=True)
        
        return loss
            
    def validation_step(self, batch, batch_idx):     
        images=batch
        loss = self.model.p_loss(self.input_T(images))
        
        self.log('val_loss',loss, prog_bar=True)
        
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

        #self.log('lr', pg['lr'], prog_bar=True)

        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
    
class PixelDiffusionConditional(PixelDiffusion):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 batch_size=1,
                 lr=1e-3,
                 schedule = 'linear',
                 warm_up_steps = 10000,
                 num_timesteps = 1000,
                 loss_fn = F.mse_loss):
        pl.LightningModule.__init__(self)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.warm_up_steps = warm_up_steps
        self.batch_size=batch_size
        channels, h, w = train_dataset[0][0].shape   
        
        self.model=DenoisingDiffusionConditionalProcess(generated_channels = channels,
                                                        condition_channels = channels,
                                                        schedule=schedule,
                                                        num_timesteps=num_timesteps,
                                                        loss_fn=loss_fn)
    
    def forward(self, condition, *args, **kwargs):
        return self.output_T(self.model(self.input_T(condition)))

    def training_step(self, batch, batch_idx):   
        input,output=batch
        loss = self.model.p_loss(self.input_T(output),self.input_T(input))
        
        self.log('train_loss',loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
            
    def validation_step(self, batch, batch_idx):     
        input,output=batch
        loss = self.model.p_loss(self.input_T(output),self.input_T(input))
        
        self.log('val_loss',loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss