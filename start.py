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
