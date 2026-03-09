import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pandas as pd
import lightning as L
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchmetrics import R2Score
from os import sys
import random
import numpy as np

class MusicModel(L.LightningModule):
  def __init__(self):
    print("hi")
    # stuff
  
  def forward(self, input):
    output = torch.zeros(1)
    return output
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    return [optimizer], [scheduler]

  def training_step(self, batch_iterator, batch_idx):
    loss = torch.zeros(1)
    return loss
  
  def test_step(self, batch_iterator, batch_idx):
    return
  
  def validation_step(self):
    return
  
  def on_train_epoch_end(self):
    return
  
  def on_test_epoch_end(self):
    return
  
  def on_validation_epoch_end(self):
    return
  
  def predict_step(self, batch_iterator, batch_idx):
    return
  