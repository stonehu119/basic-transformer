"""
Prepare MIDI datasets for decoder-only transformer training.

Downloads the Maestro dataset (from Hugging Face) and the GiantMIDI-Piano dataset
(from Kaggle), and builds a REMI tokenizer with MidiTok.
Has the option to train the tokenizer using BPE.
"""

# eventually we want to map tokens to what their actual midi data represents

import argparse
import os
import zipfile
from pathlib import Path

import kagglehub
from huggingface_hub import hf_hub_download # type: ignore
from miditok import REMI, TokenizerConfig


# Default Hugging Face repo and filename for Maestro MIDI-only zip
MAESTRO_HF_REPO = "projectlosangeles/maestro-v3.0.0"
MAESTRO_MIDI_ZIP = "maestro-v3.0.0-midi.zip"

# Kaggle dataset providing a processed (MIDI-only) version of GiantMIDI-Piano
GIANT_MIDI_KAGGLE_HANDLE = "pictureinthenoise/music-generation-with-giantmidi-piano"


def get_midi_paths(data_dir: Path) -> list[Path]:
    """Return list of paths to .mid/.midi files under data_dir."""
    paths = sorted(data_dir.rglob("*.mid")) + sorted(data_dir.rglob("*.midi"))
    if not paths:
        raise FileNotFoundError(f"No .mid/.midi files found under {data_dir}")
    return paths


def download_maestro(cache_dir: Path) -> Path:
    """
    Download Maestro MIDI zip from Hugging Face and extract to cache_dir / 'maestro_midi'.
    Returns the directory containing the extracted .midi files.
    """
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = cache_dir / "maestro_midi"

    if extract_dir.exists() and any(extract_dir.rglob("*.midi")):
        return extract_dir

    zip_path = hf_hub_download(
        repo_id=MAESTRO_HF_REPO,
        filename=MAESTRO_MIDI_ZIP,
        repo_type="dataset",
        local_dir=cache_dir,
    )
    zip_path = Path(zip_path)

    print(f"Extracting {zip_path} to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir


def download_giant_midi(cache_dir: Path) -> Path:
    """
    Download the GiantMIDI-Piano dataset (MIDI files only) from Kaggle via kagglehub.
    Returns the directory containing the downloaded .mid files.

    Requires Kaggle API credentials (~/.kaggle/kaggle.json or the KAGGLE_USERNAME /
    KAGGLE_KEY environment variables).
    """
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KAGGLEHUB_CACHE", str(cache_dir))

    print(f"Downloading {GIANT_MIDI_KAGGLE_HANDLE} from Kaggle...")
    return Path(kagglehub.dataset_download(GIANT_MIDI_KAGGLE_HANDLE))


def build_tokenizer(
    num_velocities: int = 32,
    use_chords: bool = True,
    use_programs: bool = False,
    train_bpe: bool = True,
    bpe_vocab_size: int = 5000,
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
        tokenizer.train(vocab_size=bpe_vocab_size, files_paths=midi_paths) # type: ignore
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save_params(save_path) # type: ignore
            print(f"Saved tokenizer to {save_path}")

    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Maestro, build REMI tokenizer, and prepare train/val dataloaders."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing .mid files. If not set, downloads Maestro and GiantMIDI-Piano to ./data.",
    )
    parser.add_argument(
        "--skip-giant-midi",
        action="store_true",
        help="Don't download/include the GiantMIDI-Piano dataset.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data"),
        help="Cache directory for Hugging Face download and extracted MIDIs.",
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
        default=5000,
        help="Target vocab size when training BPE.",
    )
    parser.add_argument(
        "--save-tokenizer",
        type=Path,
        default="tokenizer.json",
        help="Path to save tokenizer params (e.g. tokenizer.json).",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or Path("data")
    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
        midi_paths = get_midi_paths(data_dir)
        print(f"Using {len(midi_paths)} MIDI files from {data_dir}")
    else:
        maestro_dir = download_maestro(cache_dir)
        midi_paths = get_midi_paths(maestro_dir)
        print(f"Found {len(midi_paths)} MIDI files in Maestro ({maestro_dir})")

        if not args.skip_giant_midi:
            giant_midi_dir = download_giant_midi(cache_dir)
            giant_midi_paths = get_midi_paths(giant_midi_dir)
            print(f"Found {len(giant_midi_paths)} MIDI files in GiantMIDI-Piano ({giant_midi_dir})")
            midi_paths += giant_midi_paths

        print(f"Total: {len(midi_paths)} MIDI files")

    tokenizer = build_tokenizer(
        num_velocities=args.num_velocities,
        train_bpe=args.train_bpe,
        bpe_vocab_size=args.bpe_vocab_size,
        midi_paths=midi_paths if args.train_bpe else None,
        save_path=args.save_tokenizer,
    )
    vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {vocab_size}")


if __name__ == "__main__":
    main()
