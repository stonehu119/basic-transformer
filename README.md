# Transformers!!!
"One shall stand, one shall fall" or smth like that idk i never played transformers

### Installation Notes:
This project is compatible with CUDA versions of PyTorch. Installing dependencies from the requirements.txt file
will install CPU-only versions of `torch`, `torchvision`, and `torchmetrics`, so if you would like
to use GPU acceleration, please manually install a CUDA compatible version of these libraries.

### Cloud training on RunPod

`src/training.py` defaults to the large config (`model_dim=1024`, `num_layers=8`,
`context_size=2048`, 16 heads) tuned for a **48GB GPU** (A40 / A6000 / L40S) using
`bf16-mixed`. Every knob is overridable via environment variables, e.g.:

```bash
# bigger card (80GB H100/A100): fill it and skip accumulation
BATCH_SIZE=48 ACCUM_STEPS=1 MAX_EPOCHS=60 python src/training.py
# smaller card (24GB 4090/3090):
BATCH_SIZE=8  ACCUM_STEPS=4 python src/training.py
# older GPU without bf16 (V100/T4):
PRECISION=16-mixed python src/training.py
```

Knobs: `CONTEXT_SIZE`, `MODEL_DIM`, `NUM_LAYERS`, `NUM_HEADS`, `BATCH_SIZE`,
`ACCUM_STEPS`, `MAX_EPOCHS`, `ES_PATIENCE`, `PRECISION`, `CKPT_DIR`.

**Checkpoints & persistence.** RunPod's container disk is wiped when a pod stops.
Checkpoints are written to `/workspace/checkpoints` automatically when a RunPod
**Network Volume** is mounted at `/workspace` (else a local `checkpoints/` dir).
On top of top-5-by-`val_loss`, a `last.ckpt` and a 20-minute time-based snapshot
are kept so an interrupted/spot pod can resume — `trainer.fit` auto-resumes from
`last.ckpt` (pulling it back from remote storage first if the pod is fresh).

**Off-box uploads** (so checkpoints survive a terminated pod) are handled by
`UploadCheckpointCallback`, selected with `CKPT_UPLOAD_BACKEND`:

```bash
# Hugging Face Hub (default; huggingface_hub is already a dependency)
export CKPT_UPLOAD_BACKEND=hf
export HF_TOKEN=hf_xxx                       # a write token
export HF_REPO_ID=your-username/midi-transformer-ckpts

# or any S3-compatible store (Cloudflare R2, Backblaze B2, RunPod S3, AWS)
export CKPT_UPLOAD_BACKEND=s3
export S3_BUCKET=my-bucket
export S3_ENDPOINT_URL=https://...           # omit for real AWS S3
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
```

If a backend isn't configured the callback no-ops (training still runs); set
`CKPT_UPLOAD_BACKEND=none` to disable uploads entirely.
