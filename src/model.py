import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from os import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AttnHead(nn.Module):
  def __init__(self, model_dim = 256, attn_dim = 64):
    super().__init__()
    self.Wq = nn.Linear(model_dim, attn_dim, bias=False)
    self.Wk = nn.Linear(model_dim, attn_dim, bias=False)
    self.Wv = nn.Linear(model_dim, attn_dim, bias=False)
    self.dropout = nn.Dropout(p=0.1)
    self.attn_dim = attn_dim

  def forward(self, input: torch.Tensor, mask=None):
    Q: torch.Tensor = self.Wq(input)
    K: torch.Tensor = self.Wk(input)
    V: torch.Tensor = self.Wv(input) # all (B, T, attn_dim)

    attn_scores: torch.Tensor = Q @ K.transpose(-1, -2) / (self.attn_dim ** 0.5) # (B, T, T)
    if mask is not None: # mask is (B, T, T)
      attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

    weights = F.softmax(attn_scores, dim=-1)
    out = self.dropout(weights) @ V
    return out, weights

class MultiHeadAttn(nn.Module):
  def __init__(self, model_dim = 256, num_heads = 4):
    super().__init__()
    assert model_dim % num_heads == 0
    self.head_dim = model_dim // num_heads
    self.attn_heads = nn.ModuleList(
      [AttnHead(model_dim=model_dim, attn_dim=self.head_dim) for _ in range(num_heads)]
    )
    self.out_layer = nn.Linear(in_features=model_dim, out_features=model_dim)
    self.dropout = nn.Dropout(p=0.1)

  def forward(self, input, mask = None):
    head_outputs, head_weights = zip(*[attn_head(input, mask) for attn_head in self.attn_heads])
    out = self.out_layer(torch.cat(head_outputs, dim=-1))
    weights = torch.stack(head_weights, dim=-1)
    return self.dropout(out), weights

class TransformerBlock(nn.Module):
  def __init__(self, model_dim = 256):
    super().__init__()
    self.mha = MultiHeadAttn(model_dim, num_heads=4)
    self.ffn = nn.Sequential(
      nn.Linear(in_features=model_dim, out_features=model_dim * 4),
      nn.GELU(),
      nn.Dropout(p=0.1),
      nn.Linear(in_features=model_dim * 4, out_features=model_dim),
      nn.Dropout(p=0.1),
    )
    self.norm1 = nn.LayerNorm(normalized_shape=model_dim)
    self.norm2 = nn.LayerNorm(normalized_shape=model_dim)

  def forward(self, input, mask=None) -> torch.Tensor:
    # Pre LN block 1
    norm_input = self.norm1(input)
    attention, weights = self.mha(norm_input, mask)
    input = attention + input

    # Pre LN block 2
    norm_input = self.norm2(input)
    ffn = self.ffn(norm_input)
    input = ffn + input
    return input

class MidiGenerator(nn.Module):
  def __init__(self, vocab_size = 5000, context_size = 512, model_dim = 256, num_layers = 4):
    super().__init__()
    self.context_size = context_size
    self.embedding = nn.Embedding(vocab_size, model_dim)
    self.positional = nn.Embedding(context_size, model_dim)
    self.transformers = nn.ModuleList([TransformerBlock(model_dim=model_dim) for _ in range(num_layers)])
    self.output = nn.Linear(model_dim, vocab_size, bias=False)
    self.output.weight = self.embedding.weight
    self.causal_mask = 0

  def forward(self, input: torch.Tensor, attention_mask = None) -> torch.Tensor:
    embeddings = self.embedding(input) # (B, T, model_dim)
    position_offsets = torch.arange(self.context_size, device=device).unsqueeze(0)
    embeddings = embeddings + self.positional(position_offsets)

    mask = self.apply_mask(self.context_size, attention_mask, input.device)

    for block in self.transformers:
      embeddings = block(embeddings, mask)

    logits = self.output(embeddings) # (B, T, vocab_size)
    return logits
  
  @torch.no_grad()
  def generate(self, input, max_len, eos_token_id=None):
    B = input.size(0) # batch_size
    assert B == 1
    for _ in range(max_len):
      cropped = input[0, -self.context_size:]
      embeddings = self.embedding(cropped) # (1, T, model_dim)
      position_offsets = torch.arange(self.context_size, device=device).unsqueeze(0)
      embeddings = embeddings + self.positional(position_offsets)
      for block in self.transformers:
        embeddings = block(embeddings)
      
      last_embedding = embeddings[-1, -1, :] # (1, 1, model_dim)
      logits = self.output(last_embedding) # (1, 1, vocab_size)
      probs = F.softmax(logits, dim=-1)
      next_token = torch.multinomial(probs, num_samples=1).reshape((1, 1))
      input = torch.cat([input, next_token], dim=1)

      if eos_token_id is not None and next_token.item() == eos_token_id:
        break

    return input

  def apply_mask(self, seq_len, attention_mask: torch.Tensor, device) -> torch.Tensor:
    causal_mask: torch.Tensor = torch.tril(torch.ones(seq_len, seq_len, device = device))
    causal_mask = causal_mask.bool().unsqueeze(0) # (1, T, T)
    if attention_mask == None: return causal_mask

    B, T = attention_mask.shape
    padding_mask = attention_mask.unsqueeze(1).to(torch.bool)  # (B, 1, T)
    padding_mask = padding_mask.expand(B, T, T)

    full_mask = padding_mask & causal_mask # (B, T, T)

    return full_mask

class MidiLightningModule(L.LightningModule):
  def __init__(self, vocab_size = 5000, context_size = 512, model_dim = 256):
    super().__init__()
    self.vocab_size = vocab_size
    self.model = MidiGenerator(vocab_size=vocab_size, context_size=context_size, model_dim=model_dim)

  def forward(self, input, attention_mask = None):
    return self.model(input, attention_mask)

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    return [optimizer], [scheduler]

  def on_train_start(self):
    # set some training attribute here for self.model which causes it to apply a causal mask?
    return
  
  def _forward_loss(self, batch):
    # batch is a mapping from string to longtensor
    # has three keys: [input_ids, labels, attention_mask]
    input_token_batch = batch["input_ids"] # (B, T)
    label_token_batch = batch["labels"] # (B, T)
    attention_mask = batch["attention_mask"] # (B, T)

    logit_batch = self(input_token_batch, attention_mask) # (B, T, vocab_size)
    
    loss = F.cross_entropy( 
      logit_batch.reshape(-1, self.vocab_size),
      label_token_batch.reshape(-1) # apparently F.cross_entropy one hot encodes these already
    )
    return loss

  def training_step(self, batch, batch_idx):
    loss = self._forward_loss(batch)

    self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
    return loss

  def test_step(self, batch_iterator, batch_idx):
    return
  
  def validation_step(self, batch, batch_idx):
    loss = self._forward_loss(batch)

    self.log("val_loss", loss, prog_bar=True, on_epoch=True)
    return loss
  
  def on_train_epoch_end(self):
    return
  
  def on_test_epoch_end(self):
    return
  
  def on_validation_epoch_end(self):
    return
  
  def predict_step(self, batch_iterator, batch_idx):
    return
