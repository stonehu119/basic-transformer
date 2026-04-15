import argparse

from miditok import REMI, MusicTokenizer, TokSequence
from pathlib import Path
from model import MidiGenerator
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_from_midi(
  model,
  midi_path,
  output_path = "out/test.midi",
  tokenizer = REMI(params=Path("tokenizer.json")),
  max_tokens = 1024,
):
  torch.set_default_device(device)
  model.eval()
  tokens = load_and_tokenize(midi_path, tokenizer)

def load_and_tokenize(midi_path, tokenizer: MusicTokenizer):
  tokenized_midi: TokSequence = tokenizer.encode(midi_path) # shape: (T)
  # print(f"EOS token: {tokenizer["EOS"]}\nBOS token: {tokenizer["BOS"]}")
  np.savetxt('token_stream.txt', tokenized_midi.ids)
  eos_token_id = tokenizer["EOS"]
  if tokenized_midi[-1] == eos_token_id:
    tokenized_midi = tokenized_midi[:-1]
  
  tokenized_midi = tokenized_midi.unsqueeze(0) # shape: (1, T)
  return tokenized_midi

# def save_tokens_to_file(token_stream, output_path, tokenizer: MusicTokenizer):

if __name__ == "__main__":
  torch.set_default_device(device)
  parser = argparse.ArgumentParser(
    description="Read from a MIDI file, generate, and output to a new MIDI file"
  )
  parser.add_argument(
    "--filepath",
    type=Path,
    default="midi-transformer-epoch=12-val_loss=5.77.ckpt",
    help="Path to .midi file",
  )
  args = parser.parse_args()

  checkpoint = torch.load("checkpoints/midi-transformer-epoch=12-val_loss=5.77.ckpt")
  weights = checkpoint["state_dict"]
  model = MidiGenerator()
  # model.load_state_dict(weights)
  
  generate_from_midi(model, midi_path = "data/maestro_midi/maestro-v3.0.0/2009/MIDI-Unprocessed_20_R1_2009_01-05_ORIG_MID--AUDIO_20_R1_2009_20_R1_2009_01_WAV.midi")
