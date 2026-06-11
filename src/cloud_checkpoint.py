"""Push training checkpoints off the (ephemeral) RunPod container disk.

RunPod pods lose their container filesystem when stopped or terminated. Two
things survive: a Network Volume (mounted at ``/workspace``) and anything you
copy off-box. ``UploadCheckpointCallback`` handles the off-box half: whenever
Lightning writes a ``*.ckpt`` it mirrors the changed files to remote storage.

The backend is selected entirely through environment variables so you never
have to edit code on the pod:

    CKPT_UPLOAD_BACKEND   "hf" (default) | "s3" | "none"

Hugging Face Hub (``huggingface_hub`` is already a project dependency):
    HF_TOKEN              a write token (https://huggingface.co/settings/tokens)
    HF_REPO_ID            e.g. "your-username/midi-transformer-ckpts"
    HF_REPO_PRIVATE       "1" to create the repo private (optional)

S3 / any S3-compatible store (Cloudflare R2, Backblaze B2, RunPod S3, AWS):
    S3_BUCKET             bucket name
    S3_ENDPOINT_URL       custom endpoint (optional; omit for real AWS S3)
    S3_PREFIX             key prefix inside the bucket (optional)
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY

If the selected backend is missing its config, the callback degrades to a
no-op and prints why -- training still runs, you just don't get uploads.
"""

import os
import time
from pathlib import Path

import lightning as L


class UploadCheckpointCallback(L.Callback):
  def __init__(self, dirpath, upload_every_n_min: float = 20.0):
    super().__init__()
    self.dirpath = Path(dirpath)
    self.backend = os.environ.get("CKPT_UPLOAD_BACKEND", "hf").lower()
    self.throttle_seconds = upload_every_n_min * 60.0
    self._last_periodic_upload = 0.0
    self._uploaded_mtimes: dict[str, float] = {}
    self._hf_api = None
    self._hf_repo_id = None
    self._s3 = None
    self._s3_bucket = None
    self._s3_prefix = ""
    self._enabled = self._init_backend()

  # ----- backend setup -------------------------------------------------------
  def _init_backend(self) -> bool:
    if self.backend in ("none", "off", ""):
      print("[cloud-ckpt] uploads disabled (CKPT_UPLOAD_BACKEND=none).")
      return False
    if self.backend == "hf":
      return self._init_hf()
    if self.backend == "s3":
      return self._init_s3()
    print(f"[cloud-ckpt] unknown CKPT_UPLOAD_BACKEND={self.backend!r}; uploads disabled.")
    return False

  def _init_hf(self) -> bool:
    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID")
    if not token or not repo_id:
      print("[cloud-ckpt] HF backend selected but HF_TOKEN / HF_REPO_ID not set; uploads disabled.")
      return False
    try:
      from huggingface_hub import HfApi
    except ImportError:
      print("[cloud-ckpt] huggingface_hub not installed; uploads disabled.")
      return False
    self._hf_api = HfApi(token=token)
    self._hf_repo_id = repo_id
    private = os.environ.get("HF_REPO_PRIVATE", "0") == "1"
    self._hf_api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"[cloud-ckpt] uploading checkpoints to HF Hub repo '{repo_id}'.")
    return True

  def _init_s3(self) -> bool:
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
      print("[cloud-ckpt] S3 backend selected but S3_BUCKET not set; uploads disabled.")
      return False
    try:
      import boto3
    except ImportError:
      print("[cloud-ckpt] boto3 not installed (pip install boto3); uploads disabled.")
      return False
    endpoint = os.environ.get("S3_ENDPOINT_URL")
    self._s3 = boto3.client("s3", endpoint_url=endpoint) if endpoint else boto3.client("s3")
    self._s3_bucket = bucket
    self._s3_prefix = os.environ.get("S3_PREFIX", "").strip("/")
    print(f"[cloud-ckpt] uploading checkpoints to s3://{bucket}/{self._s3_prefix}".rstrip("/") + ".")
    return True

  # ----- upload plumbing -----------------------------------------------------
  def _upload_file(self, path: Path) -> None:
    if self.backend == "hf":
      self._hf_api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=self._hf_repo_id,
        repo_type="model",
      )
    elif self.backend == "s3":
      key = f"{self._s3_prefix}/{path.name}" if self._s3_prefix else path.name
      self._s3.upload_file(str(path), self._s3_bucket, key)

  def _sync(self, reason: str) -> None:
    """Upload every *.ckpt whose mtime changed since we last sent it."""
    if not self._enabled or not self.dirpath.exists():
      return
    for path in sorted(self.dirpath.glob("*.ckpt")):
      try:
        mtime = path.stat().st_mtime
      except OSError:
        continue  # file was rotated out from under us
      if self._uploaded_mtimes.get(path.name) == mtime:
        continue
      try:
        self._upload_file(path)
        self._uploaded_mtimes[path.name] = mtime
        print(f"[cloud-ckpt] uploaded {path.name} ({reason}).")
      except Exception as exc:  # never let an upload hiccup kill training
        print(f"[cloud-ckpt] WARNING failed to upload {path.name}: {exc}")

  # ----- Lightning hooks -----------------------------------------------------
  def on_validation_end(self, trainer, pl_module) -> None:
    if trainer.sanity_checking:
      return
    self._sync("post-validation")  # new best / last.ckpt just written

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
    now = time.time()
    if now - self._last_periodic_upload >= self.throttle_seconds:
      self._last_periodic_upload = now
      self._sync("periodic")

  def on_fit_end(self, trainer, pl_module) -> None:
    self._sync("fit-end")

  def on_exception(self, trainer, pl_module, exception) -> None:
    # Best-effort flush before the pod dies (OOM, spot reclaim, Ctrl-C).
    self._sync("on-exception")


def maybe_download_resume(dirpath, filename: str = "last.ckpt") -> str | None:
  """Pull ``last.ckpt`` back from the remote onto a fresh pod so training can
  resume across pod restarts. Returns a local path if one is available
  (already-present or downloaded), else ``None``.

  Honours the same env vars as :class:`UploadCheckpointCallback`.
  """
  dirpath = Path(dirpath)
  local = dirpath / filename
  if local.exists():
    return str(local)

  backend = os.environ.get("CKPT_UPLOAD_BACKEND", "hf").lower()
  dirpath.mkdir(parents=True, exist_ok=True)
  try:
    if backend == "hf":
      repo_id = os.environ.get("HF_REPO_ID")
      token = os.environ.get("HF_TOKEN")
      if not repo_id:
        return None
      from huggingface_hub import hf_hub_download
      downloaded = hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="model",
        token=token, local_dir=str(dirpath),
      )
      print(f"[cloud-ckpt] resumed from remote {filename}.")
      return downloaded
    if backend == "s3":
      bucket = os.environ.get("S3_BUCKET")
      if not bucket:
        return None
      import boto3
      endpoint = os.environ.get("S3_ENDPOINT_URL")
      s3 = boto3.client("s3", endpoint_url=endpoint) if endpoint else boto3.client("s3")
      prefix = os.environ.get("S3_PREFIX", "").strip("/")
      key = f"{prefix}/{filename}" if prefix else filename
      s3.download_file(bucket, key, str(local))
      print(f"[cloud-ckpt] resumed from remote {filename}.")
      return str(local)
  except Exception as exc:
    print(f"[cloud-ckpt] no remote checkpoint to resume from ({exc}).")
  return None
