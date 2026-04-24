
from miditok import REMI
from pathlib import Path
from random import shuffle

from miditok.pytorch_data import DataCollator, DatasetMIDI
from miditok.utils import split_files_for_training
import torch
from torch.utils.data import random_split, DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from model import MidiLightningModule

def _get_special_token_id(tokenizer: REMI, *candidates: str) -> int | None:
    """Return token id for the first candidate that exists in the tokenizer vocab."""
    for name in candidates:
        if name in tokenizer.vocab:
            return tokenizer.vocab[name]
    return None

def prepare_dataloaders(
    midi_paths: list[Path],
    tokenizer: REMI,
    max_seq_len: int = 512,
    chunks_dir: Path | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, REMI]:

    chunks_dir = Path("data/chunks")
    max_seq_len += 1

    shuffle(midi_paths)

    chunk_paths = list(chunks_dir.glob("**/*.midi"))
    if not chunk_paths:
        print(f"Splitting {len(midi_paths)} MIDIs into chunks (max_seq_len={max_seq_len})...")
        split_files_for_training(
            files_paths=midi_paths,
            tokenizer=tokenizer,
            save_dir=chunks_dir,
            max_seq_len=max_seq_len,
        )
        chunk_paths = sorted(chunks_dir.rglob("*.midi"))
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

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    tokenizer = REMI(params=Path("tokenizer.json"))
    # test_tokenize_process(tokenizer=tokenizer)
    train_loader, test_loader, _ = prepare_dataloaders(
        midi_paths = list(Path("data/maestro_midi").resolve().glob("**/*.midi")),
        tokenizer = tokenizer
    )
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    print(f"Batch input_ids shape: {input_ids.shape}")
    print(f"Batch labels shape: {labels.shape}")
    labels_prev = labels[:, :-1] # (B, T-1)
    targets = input_ids[:, 1:] # (B, T-1)
    assert ((labels_prev == targets) | (labels_prev == -100)).all(), "labels should be shifted input_ids (or -100)"
    print("Labels are shifted input_ids (autoregressive). OK.")
    print(f"Vocab size for model: {len(tokenizer)}")

    model = MidiLightningModule(model_dim=512)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="midi-transformer-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min"
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=True,
        mode="min"
    )

    trainer = L.Trainer(
        max_epochs=500,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        logger=TensorBoardLogger("lightning_logs/"),
        callbacks=[checkpoint_callback, early_stop_callback, TQDMProgressBar(refresh_rate=50)],
        enable_progress_bar=False,
        # fast_dev_run=True
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
