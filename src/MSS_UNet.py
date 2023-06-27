import torch
import pytorch_lightning as pl
from .DenoisingDiffusionProcess import *
from torch.utils.data import DataLoader

class plUnet(pl.LightningModule):
    def __init__(self,
                 train_dataset,
                 valid_dataset,
                 batch_size,
                 lr=1e-4,
                 warm_up_steps=10000,
                 loss_fn=torch.nn.functional.l1_loss):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset=train_dataset
        self.valid_dataset=valid_dataset
        self.batch_size=batch_size
        self.lr=lr
        self.warm_up_steps= warm_up_steps
        self.loss_fn = loss_fn

        channels, h, w = train_dataset[0][0].shape

        self.model=UnetConvNextBlock(dim=64,
                                     dim_mults = (1,2,4,8),
                                     channels=channels,
                                     out_dim=channels,
                                     with_time_emb=False)
        
    def forward(self, x):
        return self.output_T(self.model(self.input_T(x)))
    
    def input_T(self, input):
        # By default, let the model accept samples in [0,1] range, and transform them automatically
        return (input.clip(0,1).mul_(2)).sub_(1)
    
    def output_T(self, input):
        # Inverse transform of model output from [-1,1] to [0,1] range
        return (input.add_(1)).div_(2)
    
    def training_step(self, batch, batch_idx):
        mix, source = batch
        model_output=self.model(self.input_T(mix))

        loss = self.loss_fn(model_output, self.input_T(source))
        
        self.log('train_loss',loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mix, source = batch
        model_output=self.model(self.input_T(mix))

        loss = self.loss_fn(model_output, self.input_T(source))
        
        self.log('val_loss',loss, prog_bar=True, on_step=False, on_epoch=True)
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