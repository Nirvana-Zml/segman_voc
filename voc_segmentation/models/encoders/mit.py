"""Mix Transformer encoders used by SegFormer and GCB-Seg."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("drop-path probability must be in [0, 1)")
        self.probability = probability

    def forward(self, inputs: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return inputs
        keep_probability = 1.0 - self.probability
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        random_tensor = keep_probability + torch.rand(
            shape, dtype=inputs.dtype, device=inputs.device
        )
        return inputs * random_tensor.floor() / keep_probability


class OverlapPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        *,
        patch_size: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs: Tensor) -> tuple[Tensor, int, int]:
        features = self.proj(inputs)
        height, width = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        return self.norm(tokens), height, width


class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        spatial_reduction_ratio: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("attention dimension must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.sr_ratio = spatial_reduction_ratio
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim,
                dim,
                kernel_size=self.sr_ratio,
                stride=self.sr_ratio,
            )
            self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: Tensor, height: int, width: int) -> Tensor:
        batch, token_count, channels = tokens.shape
        queries = self.q(tokens).reshape(
            batch, token_count, self.num_heads, self.head_dim
        )
        queries = queries.permute(0, 2, 1, 3)

        reduced = tokens
        if self.sr_ratio > 1:
            reduced = tokens.transpose(1, 2).reshape(batch, channels, height, width)
            reduced = self.sr(reduced).reshape(batch, channels, -1).transpose(1, 2)
            reduced = self.norm(reduced)
        key_values = self.kv(reduced).reshape(
            batch, -1, 2, self.num_heads, self.head_dim
        )
        key_values = key_values.permute(2, 0, 3, 1, 4)
        keys, values = key_values[0], key_values[1]

        attention = (queries @ keys.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        output = (attention @ values).transpose(1, 2).reshape(
            batch, token_count, channels
        )
        return self.proj_drop(self.proj(output))


class DWConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=True,
        )

    def forward(self, tokens: Tensor, height: int, width: int) -> Tensor:
        batch, _, channels = tokens.shape
        features = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        features = self.dwconv(features)
        return features.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = dim * expansion
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, height: int, width: int) -> Tensor:
        tokens = self.fc1(tokens)
        tokens = self.dwconv(tokens, height, width)
        tokens = self.drop(self.act(tokens))
        return self.drop(self.fc2(tokens))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        spatial_reduction_ratio: int,
        drop_path_probability: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(
            dim,
            num_heads=num_heads,
            spatial_reduction_ratio=spatial_reduction_ratio,
        )
        self.drop_path = DropPath(drop_path_probability)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim)

    def forward(self, tokens: Tensor, height: int, width: int) -> Tensor:
        tokens = tokens + self.drop_path(self.attn(self.norm1(tokens), height, width))
        return tokens + self.drop_path(self.mlp(self.norm2(tokens), height, width))


class MixTransformer(nn.Module):
    """MiT-B1 encoder with official module naming for weight conversion."""

    feature_channels = (64, 128, 320, 512)
    feature_reductions = (4, 8, 16, 32)

    def __init__(
        self,
        *,
        embed_dims: Sequence[int] = feature_channels,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (1, 2, 5, 8),
        spatial_reduction_ratios: Sequence[int] = (8, 4, 2, 1),
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if not (
            len(embed_dims)
            == len(depths)
            == len(num_heads)
            == len(spatial_reduction_ratios)
            == 4
        ):
            raise ValueError("MiT requires four stage definitions")
        if tuple(embed_dims) != self.feature_channels:
            raise ValueError("this project currently supports the MiT-B1 dimensions only")

        self.patch_embed1 = OverlapPatchEmbed(
            3, embed_dims[0], patch_size=7, stride=4
        )
        self.patch_embed2 = OverlapPatchEmbed(
            embed_dims[0], embed_dims[1], patch_size=3, stride=2
        )
        self.patch_embed3 = OverlapPatchEmbed(
            embed_dims[1], embed_dims[2], patch_size=3, stride=2
        )
        self.patch_embed4 = OverlapPatchEmbed(
            embed_dims[2], embed_dims[3], patch_size=3, stride=2
        )

        total_blocks = sum(depths)
        drop_paths = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        offset = 0
        for index, (dim, depth, heads, sr_ratio) in enumerate(
            zip(embed_dims, depths, num_heads, spatial_reduction_ratios), start=1
        ):
            blocks = nn.ModuleList(
                TransformerBlock(
                    dim,
                    num_heads=heads,
                    spatial_reduction_ratio=sr_ratio,
                    drop_path_probability=drop_paths[offset + block_index],
                )
                for block_index in range(depth)
            )
            setattr(self, f"block{index}", blocks)
            setattr(self, f"norm{index}", nn.LayerNorm(dim))
            offset += depth
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, mean=0.0, std=(2.0 / fan_out) ** 0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward_stage(self, inputs: Tensor, stage: int) -> Tensor:
        patch_embed = getattr(self, f"patch_embed{stage}")
        blocks = getattr(self, f"block{stage}")
        norm = getattr(self, f"norm{stage}")
        tokens, height, width = patch_embed(inputs)
        for block in blocks:
            tokens = block(tokens, height, width)
        tokens = norm(tokens)
        batch, _, channels = tokens.shape
        return tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        features: list[Tensor] = []
        inputs = image
        for stage in range(1, 5):
            inputs = self._forward_stage(inputs, stage)
            features.append(inputs)
        return tuple(features)  # type: ignore[return-value]

    def load_pretrained(self, path: Path | str) -> None:
        payload = torch.load(Path(path), map_location="cpu")
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state_dict, dict):
            raise TypeError("MiT checkpoint must contain a state dictionary")
        cleaned: dict[str, Tensor] = {}
        for name, value in state_dict.items():
            clean_name = str(name)
            for prefix in ("module.", "backbone."):
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix) :]
            if clean_name.startswith("head."):
                continue
            cleaned[clean_name] = value
        self.load_state_dict(cleaned, strict=True)


def build_mit_b1(weights_path: Path | str | None = None) -> MixTransformer:
    encoder = MixTransformer()
    if weights_path is not None:
        encoder.load_pretrained(weights_path)
    return encoder
