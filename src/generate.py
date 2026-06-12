import argparse
from typing import cast

from miditok import REMI, MusicTokenizer, TokSequence
from pathlib import Path
from model import MidiGenerator, MidiLightningModule
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_from_midi(
  model: MidiGenerator,
  midi_path: Path | str,
  output_path: Path | str = "out/test.midi",
  tokenizer: MusicTokenizer = REMI(params=Path("tokenizer.json")),
  max_tokens: int = 1400,
):
  torch.set_default_device(device)
  model.eval()
  tokens = load_and_tokenize(midi_path, tokenizer)
  generated_tokens = model.generate(
    input = tokens,
    max_len = max_tokens,
    eos_token_id=cast(int, tokenizer["EOS_None"]),
    temperature=0.85
  )
  save_tokens_to_file(generated_tokens, output_path, tokenizer)

def load_and_tokenize(midi_path: Path | str, tokenizer: MusicTokenizer):
  tokenized_midi: list[TokSequence] = tokenizer.encode(midi_path) # type: ignore

  # save tokenized sequence to file as a sanity check
  with open("out/token_stream.txt", "w") as f:
    f.write("\n".join(map(str, tokenized_midi[0].ids)))

  out = torch.tensor(tokenized_midi)

  # prepend BOS token to input sequence
  bos_token = cast(int, tokenizer["BOS_None"])
  prepend = torch.full((1, 1), bos_token)
  out = torch.cat([prepend, out], dim=1)
  out = out[:, :1600]

  print(out.shape)
  return out

def save_tokens_to_file(token_stream: torch.Tensor, output_path: Path | str, tokenizer: MusicTokenizer):
  print(token_stream.shape)
  token_stream = token_stream[:, 1:-1]
  token_stream_list = cast(list[int | list[int]], token_stream.squeeze(0).cpu().tolist()) # type: ignore
  token_sequence = TokSequence(ids=token_stream_list, are_ids_encoded=True)
  tokenizer.decode_token_ids(token_sequence)
  tokenizer.complete_sequence(token_sequence)
  print(token_sequence.tokens[:5])
  tokenizer.decode([token_sequence], programs=[(0, False)], output_path=output_path) # type: ignore

if __name__ == "__main__":
  torch.set_default_device(device)
  parser = argparse.ArgumentParser(
    description="Read from a MIDI file, generate, and output to a new MIDI file"
  )
  parser.add_argument(
    "--checkpoint-path",
    type=Path,
    default="saved_checkpoints/9_3k-context-loss=3.083.ckpt",
    help="Path to .ckpt file",
  )
  parser.add_argument(
    "--midi-path",
    type=Path,
    default="test/rachmaninoff.midi",
    help="Path to .midi file",
  )
  args = parser.parse_args()

  # Architecture is restored from the checkpoint's saved hyperparameters.
  lightning_wrapper = MidiLightningModule.load_from_checkpoint(args.checkpoint_path) # pyright: ignore[reportUnknownMemberType]
  model = lightning_wrapper.model

  generate_from_midi(model, midi_path = args.midi_path)
