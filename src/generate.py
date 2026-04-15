from miditok import REMI
from pathlib import Path
from model import MidiGenerator
import torch

checkpoint = torch.load("checkpoints/midi-transformer-epoch=12-val_loss=5.77.ckpt")
weights = checkpoint["state_dict"]
model = MidiGenerator()
model.load_state_dict(weights)
