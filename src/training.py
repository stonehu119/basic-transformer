
import os
from datetime import timedelta
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
from cloud_checkpoint import UploadCheckpointCallback, maybe_download_resume


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))

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
    batch_size: int = 32,
    num_workers: int = 0,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, REMI]:

    max_seq_len += 1
    # Namespace the chunk cache by length: chunks split for a 1024 context must
    # not be silently reused when you ask for a 2048 context (and vice versa).
    chunks_dir = Path(f"data/chunks_{max_seq_len}")
    train_dir = chunks_dir / "train"
    val_dir = chunks_dir / "val"

    shuffle(midi_paths)
    n = len(midi_paths)
    n_train = int(n * train_ratio)
    train_midi_paths = midi_paths[:n_train]
    val_midi_paths = midi_paths[n_train:]

    chunk_paths = list(chunks_dir.glob("**/*.midi"))
    if not chunk_paths:
        print(f"Splitting {len(midi_paths)} MIDIs into chunks (max_seq_len={max_seq_len})...")
        split_files_for_training(
            files_paths=train_midi_paths,
            tokenizer=tokenizer,
            save_dir=train_dir,
            max_seq_len=max_seq_len,
        )
        split_files_for_training(
            files_paths=val_midi_paths,
            tokenizer=tokenizer,
            save_dir=val_dir,
            max_seq_len=max_seq_len,
        )
        chunk_paths = sorted(chunks_dir.rglob("*.midi"))
    else:
        print(f"Using existing {len(chunk_paths)} chunk files in {chunks_dir}")

    if not chunk_paths:
        raise RuntimeError("No chunk files produced. Check MIDI paths and tokenizer.")
    
    train_paths = sorted(train_dir.rglob("*.midi"))
    val_paths = sorted(val_dir.rglob("*.midi"))

    bos_id = _get_special_token_id(tokenizer, "BOS_None", "BOS")
    eos_id = _get_special_token_id(tokenizer, "EOS_None", "EOS")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = _get_special_token_id(tokenizer, "PAD_None", "PAD")
    if pad_id is None:
        raise ValueError("Tokenizer has no PAD token id")

    train_ds = DatasetMIDI(
        files_paths=train_paths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
    )
    val_ds = DatasetMIDI(
        files_paths=val_paths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
    )

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

    # ----- scaled-up model / run configuration (override any of these via env) -----
    CONTEXT_SIZE = _env_int("CONTEXT_SIZE", 2048)
    MODEL_DIM    = _env_int("MODEL_DIM", 1024)
    NUM_LAYERS   = _env_int("NUM_LAYERS", 8)
    NUM_HEADS    = _env_int("NUM_HEADS", 16)        # head_dim = 1024 / 16 = 64
    # Tuned for a 48GB card (A40 / A6000 / L40S) at 2048 context. If you see an
    # OOM, drop BATCH_SIZE to 8 and raise ACCUM_STEPS to 4; on an 80GB H100/A100
    # raise BATCH_SIZE (32-64) and set ACCUM_STEPS=1 to keep the card busy.
    BATCH_SIZE   = _env_int("BATCH_SIZE", 16)
    ACCUM_STEPS  = _env_int("ACCUM_STEPS", 2)       # effective batch = 16 * 2 = 32
    MAX_EPOCHS   = _env_int("MAX_EPOCHS", 100)
    ES_PATIENCE  = _env_int("ES_PATIENCE", 40)      # very tolerant of plateaus
    PRECISION    = os.environ.get("PRECISION", "bf16-mixed")  # use "16-mixed" on V100/T4

    # Persist to the RunPod Network Volume (/workspace) when present so
    # checkpoints survive pod restarts; fall back to a local dir otherwise.
    default_ckpt_dir = "/workspace/checkpoints" if os.path.isdir("/workspace") else "checkpoints"
    CKPT_DIR = os.environ.get("CKPT_DIR", default_ckpt_dir)

    tokenizer = REMI(params=Path("tokenizer.json"))
    # test_tokenize_process(tokenizer=tokenizer)
    train_loader, test_loader, _ = prepare_dataloaders(
        midi_paths = list(Path("data/maestro_midi").resolve().glob("**/*.midi")),
        tokenizer = tokenizer,
        num_workers=3,
        batch_size=BATCH_SIZE,
        max_seq_len=CONTEXT_SIZE,  # match the model's context window
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

    model = MidiLightningModule(
        model_dim=MODEL_DIM,
        context_size=CONTEXT_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
    )

    # Top-5 by val_loss, and always keep a rolling last.ckpt for crash resume.
    best_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=CKPT_DIR,
        filename="midi-{epoch:02d}-{step}-{val_loss:.3f}",
        save_top_k=5,
        mode="min",
        save_last=True,
        auto_insert_metric_name=False,
    )
    # Time-based snapshot every 20 min -- crucial on interruptible/spot GPUs
    # where the pod can vanish between validations.
    periodic_checkpoint = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename="periodic-{epoch:02d}-{step}",
        train_time_interval=timedelta(minutes=20),
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=ES_PATIENCE,   # epochs of no >min_delta improvement before stopping
        min_delta=1e-3,         # ignore tiny noise so plateaus must be real
        check_finite=True,
        verbose=True,
        mode="min"
    )
    upload_callback = UploadCheckpointCallback(dirpath=CKPT_DIR)

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        precision=PRECISION,
        accumulate_grad_batches=ACCUM_STEPS,
        logger=TensorBoardLogger("lightning_logs/"),
        callbacks=[
            best_checkpoint,
            periodic_checkpoint,
            early_stop_callback,
            upload_callback,
            TQDMProgressBar(refresh_rate=100),
        ],
        gradient_clip_val=1.0
        # fast_dev_run=True
    )

    # Resume automatically: prefer a local last.ckpt, otherwise try to pull one
    # back from remote storage (fresh pod after a restart).
    resume_path = maybe_download_resume(CKPT_DIR, "last.ckpt")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader, ckpt_path=resume_path)
