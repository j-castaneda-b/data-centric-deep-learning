import os
import numpy as np
from pprint import pprint

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import datasets, transforms
from metaflow import FlowSpec, step, Parameter


## Implement data class

class DataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()

        self.train_dataset = datasets.MNIST('./data', download=True, train=False, transform= transforms.ToTensor())
        self.test_dataset = datasets.MNIST('./data', download=True, train=True, transform= transforms.ToTensor())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32)




class Predictor(pl.LightningModule):

    def __init__(self, lr):
        super().__init__()

        self.save_hyperparameters() # By default implemented in any Pytorch Lightning system. Saves whatever is passed to __init__
        self.model = self.get_model()
        self.test_loss = None #Track later
        self.lr = lr

    def get_model(self):
        return nn.Linear(28*28*1, 10) #MNIST images are 28x28x1 - 10 output channels


    def forward(self, image):
        batch_size = image.size(0)
        # shape: batch_size x 1 x 28 x 28 -> batch_size * 784
        image = image.view(batch_size, -1)
        logits = self.model(image) # Shape: batch_size x 10
        return logits

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr) #self.parameters() is a special function in Pytorch Lightning. Calls self.model.parameters() - parameters of model -

    def _common_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = F.cross_entropy(logits, labels)
        return loss


    def training_step(self, train_batch, batch_idx):
        return self._common_step(train_batch, batch_idx)

    def validation_step(self, dev_batch, batch_idx):# validation_step called for every minibatch but after accumulating batch losses in a list it puts them in "outputs"
        return self._common_step(dev_batch, batch_idx) #<- loss

    def test_step(self, test_batch, batch_idx):
        return self._common_step(test_batch, batch_idx)

    def validation_epoch_end(self, outputs): # Takes "outputs" list of validation_step for the epoch: outputs <- [losses...]
        avg_loss = torch.mean(torch.FloatTensor(outputs))
        self.log_dict({'dev_loss': avg_loss},
        on_step=False, on_epoch=True, prog_bar=True, logger=True) #special function in pytorch lightning. "This is important, I want to log it!"

    def test_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.FloatTensor(outputs))
        self.test_loss = avg_loss.item()

    def predict_step(self,image):
        return self.forward(image)



class Flow(FlowSpec):

    @step
    def start(self):
        """start"""
        self.lrs = [1e-3, 5e-3, 1e-2]
        self.next(self.train, foreach='lrs') #lrs is a class variable but must be passed as a string. metaflow will do the lookup for you 

    @step
    def train(self):
        """train"""
        dm = DataModule()
        system = Predictor(self.input) ## variable specified in foreach will be saved in self.input

        ckpt_callback = ModelCheckpoint(
            dirpath=f'./ckpts/lr_{self.input}',
            monitor = 'dev_loss', # dev_loss is actually being logged in the Predictor pytorch system. Names must be equal.
            mode = 'min',
            save_top_k=1,
            verbose=True
        )

        trainer = Trainer(max_epochs=5, callbacks=[ckpt_callback])
        trainer.fit(system, dm)

        self.trainer = trainer
        self.callback = ckpt_callback
        self.dm = dm
        self.system = system

        self.next(self.join)


    @step
    def join(self, inputs):
        """join"""
        self.dm = inputs[0].dm

        scores = [inp.callback.best_model_score for inp in inputs]
        best_index = np.argmin(scores)

        self.system = inputs[best_index].system
        self.trainer = inputs[best_index].trainer
        self.best_lr = inputs[0].lrs[best_index]

        self.next(self.evaluate)

    @step 
    def evaluate(self):
        """evaluate"""
        self.trainer.test(self.system, self.dm, ckpt_path="best")
        results = self.system.test_loss   ## Pulls out variable 'test_loss' from test_epoch_end function from the pytorch_lightning system
        pprint(results)

        self.next(self.end)



    @step
    def end(self):
        """end"""
        print("Done!")


if __name__ =="__main__":
    flow = Flow()