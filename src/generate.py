import argparse

from miditok import REMI, MusicTokenizer, TokSequence
from pathlib import Path
from model import MidiGenerator, MidiLightningModule
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_from_midi(
  model: MidiGenerator,
  midi_path,
  output_path = "out/test.midi",
  tokenizer = REMI(params=Path("tokenizer.json")),
  max_tokens = 10000,
):
  torch.set_default_device(device)
  model.eval()
  tokens = load_and_tokenize(midi_path, tokenizer)
  generated_tokens = model.generate(input = tokens, max_len = max_tokens, eos_token_id=tokenizer["EOS_None"])
  save_tokens_to_file(generated_tokens, output_path, tokenizer)

def load_and_tokenize(midi_path, tokenizer: MusicTokenizer):
  tokenized_midi: list[TokSequence] = tokenizer.encode(midi_path)

  # save tokenized sequence to file as a sanity check
  with open("token_stream.txt", "w") as f:
    f.write("\n".join(map(str, tokenized_midi[0].ids)))

  out = torch.tensor(tokenized_midi)

  # prepend BOS token to input sequence
  bos_token = tokenizer["BOS_None"]
  prepend = torch.full((1, 1), bos_token)
  out = torch.cat([prepend, out], dim=1)

  print(out.shape)
  return out

def save_tokens_to_file(token_stream, output_path, tokenizer: MusicTokenizer):
  tokenizer.decode(token_stream, output_path)

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

  lightning_wrapper = MidiLightningModule.load_from_checkpoint("checkpoints/midi-transformer-epoch=93-val_loss=4.79.ckpt")
  model = lightning_wrapper.model

  generate_from_midi(model, midi_path = "data/maestro_midi/maestro-v3.0.0/2009/MIDI-Unprocessed_20_R1_2009_01-05_ORIG_MID--AUDIO_20_R1_2009_20_R1_2009_01_WAV.midi")
