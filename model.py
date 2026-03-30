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

class AttnHead(nn.Module):
  def __init__(self, model_dim = 256, attn_dim = 64):
    super().__init__()
    self.Wq = nn.Linear(model_dim, attn_dim, bias=False)
    self.Wk = nn.Linear(model_dim, attn_dim, bias=False)
    self.Wv = nn.Linear(model_dim, attn_dim, bias=False)
    self.attn_dim = attn_dim
  
  def forward(self, x, mask=None):
    Q = self.Wq(x)
    K = self.Wk(x)
    V = self.Wv(x)
    
    attn_scores = Q @ K.transpose(-1, -2) / torch.sqrt(self.attn_dim)
    if mask is not None:
      attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
    
    queries = F.softmax(attn_scores, dim=1)
    return queries @ V, queries

class MidiGenerator(L.LightningModule):
  def __init__(self, vocab_size = 1024, context_size = 512, model_dim = 256):
    self.embedding = nn.Embedding(vocab_size, model_dim)
    self.positional = nn.Embedding(context_size, model_dim)
    
    # stuff
  
  def forward(self, input):
    output = torch.zeros(1)
    return output
  
  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
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
