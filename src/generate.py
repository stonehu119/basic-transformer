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
  max_tokens = 200,
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
  out = out[:, :480]

  print(out.shape)
  return out

def save_tokens_to_file(token_stream: torch.Tensor, output_path, tokenizer: MusicTokenizer):
  print(token_stream.shape)
  token_stream = token_stream[:, 1:-1]
  token_stream = token_stream.squeeze(0).cpu().tolist()
  token_sequence = TokSequence(ids=token_stream, are_ids_encoded=True)
  tokenizer.decode_token_ids(token_sequence)
  tokenizer.complete_sequence(token_sequence)
  print(token_sequence.tokens[:5])
  tokenizer.decode([token_sequence], programs=[(0, False)], output_path=output_path)

if __name__ == "__main__":
  torch.set_default_device(device)
  parser = argparse.ArgumentParser(
    description="Read from a MIDI file, generate, and output to a new MIDI file"
  )
  parser.add_argument(
    "--checkpoint-path",
    type=Path,
    default="saved_checkpoints/5_doubled_model_dim.ckpt",
    help="Path to .ckpt file",
  )
  parser.add_argument(
    "--midi-path",
    type=Path,
    default="test/arabesque.midi",
    help="Path to .midi file",
  )
  args = parser.parse_args()

  lightning_wrapper = MidiLightningModule.load_from_checkpoint(args.checkpoint_path, model_dim = 512, context_size=2048)
  model = lightning_wrapper.model

  generate_from_midi(model, midi_path = args.midi_path)
