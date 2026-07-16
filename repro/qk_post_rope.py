"""Version-tolerant post-RoPE Q/K transformation for LLaMA attention.

The Hugging Face LLaMA implementation keeps RoPE inside each attention
module's forward method.  QuaRot needs to act immediately afterwards, before
keys enter the cache.  This wrapper scopes a temporary replacement of the
module-local helper only for one attention invocation, so it does not copy or
fork Transformers' attention implementation.
"""

import importlib
import threading
from typing import Callable, Tuple

import torch
from torch import nn

from repro.qdq import qdq_attention_tensor


_ROPE_PATCH_LOCK = threading.RLock()


class PostRoPEQKAttention(nn.Module):
    """Delegate to a LLaMA attention module after Q/K post-RoPE processing."""

    def __init__(
        self,
        attention: nn.Module,
        head_rotation: torch.Tensor,
        *,
        rotate_qk: bool,
        key_bits: int,
        key_group_size: int,
        key_symmetric: bool,
        key_clip_ratio: float,
    ) -> None:
        super().__init__()
        module = importlib.import_module(attention.__class__.__module__)
        if not callable(getattr(module, "apply_rotary_pos_emb", None)):
            raise RuntimeError("the selected LLaMA attention backend has no patchable apply_rotary_pos_emb helper")
        self.attention = attention
        self._modeling_module = module
        self.register_buffer("head_rotation", head_rotation)
        self.rotate_qk = rotate_qk
        self.key_bits = key_bits
        self.key_group_size = key_group_size
        self.key_symmetric = key_symmetric
        self.key_clip_ratio = key_clip_ratio

    def _post_rope(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 4 or key.ndim != 4:
            raise RuntimeError("expected post-RoPE Q/K tensors with four dimensions")
        if query.shape[-1] != self.head_rotation.shape[0] or key.shape[-1] != self.head_rotation.shape[0]:
            raise RuntimeError("post-RoPE Q/K head dimension is incompatible with the configured Hadamard")
        if self.rotate_qk:
            query = query @ self.head_rotation
            key = key @ self.head_rotation
        if self.key_bits < 16:
            key = qdq_attention_tensor(
                key,
                self.key_bits,
                group_size=self.key_group_size,
                symmetric=self.key_symmetric,
                clip_ratio=self.key_clip_ratio,
            )
        return query, key

    def forward(self, *args: object, **kwargs: object) -> object:
        """Run one attention call with a thread-safe, invocation-local patch."""
        with _ROPE_PATCH_LOCK:
            original: Callable[..., Tuple[torch.Tensor, torch.Tensor]] = self._modeling_module.apply_rotary_pos_emb

            def wrapped(*rope_args: object, **rope_kwargs: object) -> Tuple[torch.Tensor, torch.Tensor]:
                query, key = original(*rope_args, **rope_kwargs)
                return self._post_rope(query, key)

            self._modeling_module.apply_rotary_pos_emb = wrapped
            try:
                return self.attention(*args, **kwargs)
            finally:
                self._modeling_module.apply_rotary_pos_emb = original


def install_post_rope_qk(
    model: nn.Module,
    head_rotation: torch.Tensor,
    *,
    rotate_qk: bool,
    key_bits: int,
    key_group_size: int,
    key_symmetric: bool,
    key_clip_ratio: float,
) -> None:
    """Wrap each LLaMA attention block before its first transformed forward."""
    for layer in model.model.layers:
        layer.self_attn = PostRoPEQKAttention(
            layer.self_attn,
            head_rotation,
            rotate_qk=rotate_qk,
            key_bits=key_bits,
            key_group_size=key_group_size,
            key_symmetric=key_symmetric,
            key_clip_ratio=key_clip_ratio,
        )
