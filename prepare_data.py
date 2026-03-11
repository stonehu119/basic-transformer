"""
Prepare Maestro MIDI dataset for decoder-only transformer training.

Downloads the Maestro dataset (from Hugging Face), builds a REMI tokenizer with MidiTok,
splits long sequences into fixed-length chunks, and exposes a PyTorch Dataset/DataLoader
with input_ids and labels (shifted for autoregressive next-token prediction).
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DataCollator, DatasetMIDI, split_files_for_training
from torch.utils.data import DataLoader


# Default Hugging Face repo and filename for Maestro MIDI-only zip
MAESTRO_HF_REPO = "projectlosangeles/maestro-v3.0.0"
MAESTRO_MIDI_ZIP = "maestro-v3.0.0-midi.zip"


def get_maestro_midi_paths(data_dir: Path) -> list[Path]:
    """Return list of paths to .mid files under data_dir (e.g. after extracting Maestro zip)."""
    paths = sorted(data_dir.rglob("*.mid"))
    if not paths:
        raise FileNotFoundError(f"No .mid files found under {data_dir}")
    return paths


def download_maestro(cache_dir: Path) -> Path:
    """
    Download Maestro MIDI zip from Hugging Face and extract to cache_dir / 'maestro_midi'.
    Returns the directory containing the extracted .mid files.
    """
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = cache_dir / "maestro_midi"

    if extract_dir.exists() and any(extract_dir.rglob("*.mid")):
        return extract_dir

    zip_path = hf_hub_download(
        repo_id=MAESTRO_HF_REPO,
        filename=MAESTRO_MIDI_ZIP,
        repo_type="dataset",
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )
    zip_path = Path(zip_path)

    print(f"Extracting {zip_path} to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir


def _get_special_token_id(tokenizer: REMI, *candidates: str) -> int | None:
    """Return token id for the first candidate that exists in the tokenizer vocab."""
    for name in candidates:
        if name in tokenizer.vocab:
            return tokenizer.vocab[name]
    return None


def build_tokenizer(
    num_velocities: int = 16,
    use_chords: bool = False,
    use_programs: bool = False,
    train_bpe: bool = False,
    bpe_vocab_size: int = 30000,
    midi_paths: list[Path] | None = None,
    save_path: Path | None = None,
) -> REMI:
    """
    Build a REMI tokenizer with the given config. Optionally train BPE on midi_paths and save.
    """
    config = TokenizerConfig(
        num_velocities=num_velocities,
        use_chords=use_chords,
        use_programs=use_programs,
    )
    tokenizer = REMI(config)

    if train_bpe and midi_paths:
        print(f"Training BPE (vocab_size={bpe_vocab_size}) on {len(midi_paths)} files...")
        tokenizer.train(vocab_size=bpe_vocab_size, files_paths=midi_paths)
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save_params(save_path)
            print(f"Saved tokenizer to {save_path}")

    return tokenizer


def prepare_dataloaders(
    midi_paths: list[Path],
    tokenizer: REMI,
    max_seq_len: int = 1024,
    chunks_dir: Path | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, REMI]:
    """
    Split MIDIs into chunks, build DatasetMIDI and DataLoaders with train/val split.
    Returns (train_loader, val_loader, tokenizer).
    """
    import torch
    from torch.utils.data import random_split

    if chunks_dir is None:
        chunks_dir = Path(midi_paths[0]).parent / "_chunks"
    chunks_dir = Path(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths = list(chunks_dir.glob("**/*.mid"))
    if not chunk_paths:
        print(f"Splitting {len(midi_paths)} MIDIs into chunks (max_seq_len={max_seq_len})...")
        split_files_for_training(
            files_paths=midi_paths,
            tokenizer=tokenizer,
            save_dir=chunks_dir,
            max_seq_len=max_seq_len,
        )
        chunk_paths = sorted(chunks_dir.rglob("*.mid"))
    else:
        print(f"Using existing {len(chunk_paths)} chunk files in {chunks_dir}")

    if not chunk_paths:
        raise RuntimeError("No chunk files produced. Check MIDI paths and tokenizer.")

    bos_id = _get_special_token_id(tokenizer, "BOS_None", "BOS")
    eos_id = _get_special_token_id(tokenizer, "EOS_None", "EOS")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = _get_special_token_id(tokenizer, "PAD_None", "PAD")
    if pad_id is None:
        raise ValueError("Tokenizer has no PAD token id")

    dataset = DatasetMIDI(
        files_paths=chunk_paths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
    )

    gen = torch.Generator().manual_seed(seed)
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = n - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

    collator = DataCollator(
        pad_token_id=pad_id,
        copy_inputs_as_labels=True,
        shift_labels=True,
        labels_pad_idx=-100,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )

    return train_loader, val_loader, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Maestro, build REMI tokenizer, and prepare train/val dataloaders."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing .mid files (or where to download Maestro). If not set, downloads Maestro to ./data.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data"),
        help="Cache directory for Hugging Face download and extracted MIDIs.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1024,
        help="Max token sequence length (chunk size).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for DataLoader.",
    )
    parser.add_argument(
        "--num-velocities",
        type=int,
        default=16,
        help="Number of velocity bins for REMI.",
    )
    parser.add_argument(
        "--train-bpe",
        action="store_true",
        help="Train BPE on the dataset and save tokenizer.",
    )
    parser.add_argument(
        "--bpe-vocab-size",
        type=int,
        default=30000,
        help="Target vocab size when training BPE.",
    )
    parser.add_argument(
        "--save-tokenizer",
        type=Path,
        default=None,
        help="Path to save tokenizer params (e.g. tokenizer.json).",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=None,
        help="Directory to save/load MIDI chunks. Default: <data_dir>/_chunks.",
    )
    parser.add_argument(
        "--verify-batch",
        action="store_true",
        help="Print one batch shape and that labels are shifted input_ids.",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or Path("data")
    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
        midi_paths = get_maestro_midi_paths(data_dir)
        print(f"Using {len(midi_paths)} MIDI files from {data_dir}")
    else:
        data_dir = download_maestro(cache_dir)
        midi_paths = get_maestro_midi_paths(data_dir)
        print(f"Using {len(midi_paths)} MIDI files from {data_dir}")

    tokenizer = build_tokenizer(
        num_velocities=args.num_velocities,
        use_chords=False,
        use_programs=False,
        train_bpe=args.train_bpe,
        bpe_vocab_size=args.bpe_vocab_size,
        midi_paths=midi_paths if args.train_bpe else None,
        save_path=args.save_tokenizer,
    )
    vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {vocab_size}")

    train_loader, val_loader, _ = prepare_dataloaders(
        midi_paths=midi_paths,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        chunks_dir=args.chunks_dir or (data_dir / "_chunks"),
        batch_size=args.batch_size,
        train_ratio=0.9,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    if args.verify_batch:
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        print(f"Batch input_ids shape: {input_ids.shape}")
        print(f"Batch labels shape: {labels.shape}")
        # With shift_labels=True, labels[:, t] == input_ids[:, t+1]; padding in labels is -100
        labels_prev = labels[:, :-1]  # (B, T-1)
        targets = input_ids[:, 1:]   # (B, T-1)
        assert ((labels_prev == targets) | (labels_prev == -100)).all(), "labels should be shifted input_ids (or -100)"
        print("Labels are shifted input_ids (autoregressive). OK.")
        print(f"Vocab size for model: {vocab_size}")


if __name__ == "__main__":
    main()
