import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from os import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Note: this module is unused now but when starting out I wrote this to
#       help me understand the attention mechanism better. However,
#       I've condensed the separate heads into a vectorized operation
#       inside MultiHeadAttn, which calls F.scaled_dot_product_attention
#       which from what I understand uses FlashAttention to speed up its
#       calculation as well as using optimized CUDA kernels.
class AttnHead(nn.Module):
  def __init__(self, model_dim = 256, attn_dim = 64):
    super().__init__()
    self.Wq = nn.Linear(model_dim, attn_dim, bias=False)
    self.Wk = nn.Linear(model_dim, attn_dim, bias=False)
    self.Wv = nn.Linear(model_dim, attn_dim, bias=False)
    self.attn_dim = attn_dim

  def forward(self, input: torch.Tensor, mask=None):
    Q: torch.Tensor = self.Wq(input)
    K: torch.Tensor = self.Wk(input)
    V: torch.Tensor = self.Wv(input) # all (B, T, attn_dim)

    attn_scores: torch.Tensor = Q @ K.transpose(-1, -2) / (self.attn_dim ** 0.5) # (B, T, T)
    if mask is not None: # mask is (B, T, T)
      attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

    weights = F.softmax(attn_scores, dim=-1)
    out = weights @ V
    return out, weights

class MultiHeadAttn(nn.Module):
  def __init__(self, model_dim = 256, num_heads = 4):
    super().__init__()
    assert model_dim % num_heads == 0
    self.num_heads = num_heads
    self.attn_dim = model_dim // num_heads # also known as head_dim

    # need to project input (B, T, model_dim) into (B, T, 3*model_dim), 3x because one for q, k, v
    self.qkv = nn.Linear(in_features=model_dim, out_features=3*model_dim)

    self.out_layer = nn.Linear(in_features=model_dim, out_features=model_dim)

  def forward(self, input: torch.Tensor, mask: torch.Tensor = None):
    # input starts as (B, T, model_dim)
    print("") # newline for readability
    B, T, model_dim = input.shape
    assert B == 1
    print(f"Input with dimensions (T, model_dim):\n{input.reshape(T, model_dim)}")
    projected: torch.Tensor = self.qkv(input) # (B, T, 3*model_dim)
    expanded = projected.reshape(B, T, 3, self.num_heads, self.attn_dim)
    permuted = expanded.permute(2, 0, 3, 1, 4) # (3, B, num_heads, T, attn_dim)
    Q, K, V = permuted.unbind(0) # 3 tensors of (B, num_heads, T, attn_dim)
    print(f"\nSingle head of attention, all (T, attn_dim):\nQ:\n{Q[0,0,:,:]}\nK:\n{K[0,0,:,:]}\nV:\n{Q[0,0,:,:]}")
    print(f"\nMask (T, T):\n{mask[0, :, :]}")

    # mask starts as (B, T, T). We need to project it into a shape that is broadcastable to (B, num_heads, T, T)
    mask = None if mask is None else mask.unsqueeze(1)

    attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask) # 1 tensor (B, num_heads, T, attn_dim)
    print(f"\nAttention output of 1 head (T, attn_dim):\n{attn_out}")
    de_permuted = attn_out.permute(0, 2, 1, 3) # (B, T, num_heads, attn_dim)
    out = de_permuted.reshape(B, T, model_dim)
    return self.out_layer(out)

class TransformerBlock(nn.Module):
  def __init__(self, model_dim = 256):
    super().__init__()
    self.mha = MultiHeadAttn(model_dim, num_heads=4)
    self.dropout = nn.Dropout(p=0.1)
    self.ffn = nn.Sequential(
      nn.Linear(in_features=model_dim, out_features=model_dim * 4),
      nn.GELU(),
      nn.Linear(in_features=model_dim * 4, out_features=model_dim),
      nn.Dropout(p=0.1),
    )
    self.norm1 = nn.LayerNorm(normalized_shape=model_dim)
    self.norm2 = nn.LayerNorm(normalized_shape=model_dim)

  def forward(self, input, mask=None) -> torch.Tensor:
    # Pre LN block 1
    norm_input = self.norm1(input)
    attention = self.mha(norm_input, mask)
    dropped = self.dropout(attention)

    input = dropped + input

    # Pre LN block 2
    norm_input2 = self.norm2(input)
    ffn = self.ffn(norm_input2)
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
    self.__init_weights()

  def forward(self, input: torch.Tensor, attention_mask = None) -> torch.Tensor:
    embeddings = self.embedding(input) # (B, T, model_dim)
    T = input.size(1)
    position_offsets = torch.arange(T, device=device).unsqueeze(0)
    embeddings = embeddings + self.positional(position_offsets)

    mask = self.apply_mask(T, attention_mask, input.device)

    for block in self.transformers:
      embeddings = block(embeddings, mask)

    logits = self.output(embeddings) # (B, T, vocab_size)
    return logits
  
  @torch.no_grad()
  def generate(self, input: torch.Tensor, max_len, eos_token_id=None):
    B = input.size(0)
    assert B == 1
    for _ in range(max_len):
      cropped = input[:, -self.context_size:] # (1, T, model_dim)
      T = cropped.size(1)
      embeddings = self.embedding(cropped) # (1, T, model_dim)
      position_offsets = torch.arange(T, device=device).unsqueeze(0)
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
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device = device))
    causal_mask = causal_mask.bool().unsqueeze(0) # (1, T, T)
    if attention_mask is None: return causal_mask

    B, T = attention_mask.shape
    padding_mask = attention_mask.unsqueeze(1).to(torch.bool)  # (B, 1, T)
    padding_mask = padding_mask.expand(B, T, T)

    full_mask = padding_mask & causal_mask # (B, T, T)

    return full_mask
  
  def __init_weights(self):
    """Initializes weights in the model to smaller values to stabilize training"""
    # scale down regular modules first:
    for module in self.modules():

      if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
          nn.init.zeros_(module.bias)

      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
    
    # residual layers build up variance because of addition. Scale these down:
    num_layers = len(self.transformers)
    residual_proj_std = 0.02 / (2 * num_layers) ** 0.5
    block: TransformerBlock
    for block in self.transformers:
      nn.init.normal_(block.mha.out_layer.weight, std=residual_proj_std)
      nn.init.normal_(block.ffn[2].weight, std=residual_proj_std)

class MidiLightningModule(L.LightningModule):
  def __init__(self, vocab_size = 5000, context_size = 512, model_dim = 256):
    super().__init__()
    self.vocab_size = vocab_size
    self.model = MidiGenerator(vocab_size=vocab_size, context_size=context_size, model_dim=model_dim)
  
  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
    warmup_steps = 100
    def lr_lambda(step: int):
      if step < warmup_steps:
        return step / warmup_steps
      return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30000)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    return {
      "optimizer": optimizer,
      "lr_scheduler": {
        "scheduler": scheduler,
        "interval": "step",
      }
    }

  def _get_param_groups(self):
    # No weight decay on biases and LayerNorm params
    decay, no_decay = [], []
    for name, param in self.named_parameters():
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ]


  def forward(self, input, attention_mask = None):
    return self.model(input, attention_mask)
  
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

  def training_step(self, batch):
    loss = self._forward_loss(batch)

    self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
    return loss

  def validation_step(self, batch):
    loss = self._forward_loss(batch)

    self.log("val_loss", loss, prog_bar=True, on_epoch=True)
    return loss
