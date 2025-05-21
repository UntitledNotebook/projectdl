import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Optional, Tuple
from einops import rearrange

class SinusoidalPosEmb(nn.Module):
    """Generates sinusoidal positional embeddings for diffusion timesteps.

    Takes a batch of timestep indices and produces sinusoidal embeddings used to
    condition the UNet on the diffusion process's time step.

    Attributes:
        dim (int): The dimensionality of the output embeddings. Must be even.
        dtype (Any): The data type of the embeddings (default: jnp.float32).

    Args:
        time (jnp.ndarray): Timestep indices, shape (batch_size,).

    Returns:
        jnp.ndarray: Sinusoidal embeddings, shape (batch_size, dim).
    """
    dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, time):
        assert len(time.shape) == 1
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=self.dtype) * -emb)
        emb = time.astype(self.dtype)[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb

class WeightStandardizedConv3D(nn.Module):
    """Applies a 3D convolution with weight standardization.

    Standardizes the convolution kernel weights by subtracting the mean and
    dividing by the standard deviation, improving training stability.

    Attributes:
        features (int): Number of output channels.
        kernel_size (Tuple[int, int, int]): Size of the 3D convolution kernel
            (default: (3, 3, 3)).
        strides (Tuple[int, int, int]): Stride of the convolution
            (default: (1, 1, 1)).
        padding (Any): Padding for the convolution (default: ((1, 1), (1, 1), (1, 1))).
        dtype (Any): Data type for computations (default: jnp.float32).
        param_dtype (Any): Data type for parameters (default: jnp.float32).

    Args:
        x (jnp.ndarray): Input tensor, shape (batch_size, depth, height, width, channels).

    Returns:
        jnp.ndarray: Output tensor, shape (batch_size, depth', height', width', features).
    """
    features: int
    kernel_size: Tuple[int, int, int] = (3, 3, 3)
    strides: Tuple[int, int, int] = (1, 1, 1)
    padding: Any = ((1, 1), (1, 1), (1, 1))
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        conv = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            parent=None
        )
        kernel_init = lambda rng, x: conv.init(rng, x)['params']['kernel']
        bias_init = lambda rng, x: conv.init(rng, x)['params']['bias']
        kernel = self.param('kernel', kernel_init, x)
        eps = 1e-5 if self.dtype == jnp.float32 else 1e-3
        redux = tuple(range(kernel.ndim - 1))
        mean = jnp.mean(kernel, axis=redux, dtype=self.dtype, keepdims=True)
        var = jnp.var(kernel, axis=redux, dtype=self.dtype, keepdims=True)
        standardized_kernel = (kernel - mean) / jnp.sqrt(var + eps)
        bias = self.param('bias', bias_init, x)
        return conv.apply({'params': {'kernel': standardized_kernel, 'bias': bias}}, x)

class Downsample3D(nn.Module):
    """Downsamples a 3D tensor using a strided convolution.

    Reduces the spatial dimensions (depth, height, width) by a factor of 2 using
    a 4x4x4 convolution with stride 2.

    Attributes:
        dim (Optional[int]): Number of output channels. If None, matches input channels.
        dtype (Any): Data type for computations (default: jnp.float32).

    Args:
        x (jnp.ndarray): Input tensor, shape (batch_size, depth, height, width, channels).

    Returns:
        jnp.ndarray: Downsampled tensor, shape (batch_size, depth//2, height//2, width//2, dim).
    """
    dim: Optional[int] = None
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, D, H, W, C = x.shape
        dim = self.dim if self.dim is not None else C
        x = nn.Conv(
            dim, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding=((1, 1), (1, 1), (1, 1)), dtype=self.dtype
        )(x)
        assert x.shape == (B, D // 2, H // 2, W // 2, dim)
        return x

class Upsample3D(nn.Module):
    """Upsamples a 3D tensor using nearest-neighbor interpolation and convolution.

    Increases spatial dimensions (depth, height, width) by a factor of 2 using
    nearest-neighbor interpolation, followed by a 3x3x3 convolution.

    Attributes:
        dim (Optional[int]): Number of output channels. If None, matches input channels.
        dtype (Any): Data type for computations (default: jnp.float32).

    Args:
        x (jnp.ndarray): Input tensor, shape (batch_size, depth, height, width, channels).

    Returns:
        jnp.ndarray: Upsampled tensor, shape (batch_size, depth*2, height*2, width*2, dim).
    """
    dim: Optional[int] = None
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, D, H, W, C = x.shape
        dim = self.dim if self.dim is not None else C
        x = jax.image.resize(x, (B, D * 2, H * 2, W * 2, C), 'nearest')
        x = nn.Conv(
            dim, kernel_size=(3, 3, 3), padding=((1, 1), (1, 1), (1, 1)), dtype=self.dtype
        )(x)
        assert x.shape == (B, D * 2, H * 2, W * 2, dim)
        return x

class ResnetBlock3D(nn.Module):
    """A 3D residual block with time-conditioned normalization.

    Applies two weight-standardized 3D convolutions with group normalization,
    conditioned on time embeddings via scale and shift parameters. Includes a
    residual connection.

    Attributes:
        dim (int): Number of output channels.
        groups (Optional[int]): Number of groups for group normalization (default: 8).
        dtype (Any): Data type for computations (default: jnp.float32).

    Args:
        x (jnp.ndarray): Input tensor, shape (batch_size, depth, height, width, channels).
        time_emb (jnp.ndarray): Time embeddings, shape (batch_size, time_dim).

    Returns:
        jnp.ndarray: Output tensor, shape (batch_size, depth, height, width, dim).
    """
    dim: int
    groups: Optional[int] = 8
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, time_emb):
        B, D, H, W, C = x.shape
        assert time_emb.shape[0] == B and len(time_emb.shape) == 2
        h = WeightStandardizedConv3D(
            features=self.dim, kernel_size=(3, 3, 3), padding=((1, 1), (1, 1), (1, 1)), name='conv_0'
        )(x)
        h = nn.GroupNorm(num_groups=self.groups, dtype=self.dtype, name='norm_0')(h)
        time_emb = nn.Dense(features=2 * self.dim, dtype=self.dtype, name='time_mlp.dense_0')(nn.swish(time_emb))
        time_emb = time_emb[:, None, None, None, :]  # [B, 1, 1, 1, 2*dim]
        scale, shift = jnp.split(time_emb, 2, axis=-1)
        h = h * (1 + scale) + shift
        h = nn.swish(h)
        h = WeightStandardizedConv3D(
            features=self.dim, kernel_size=(3, 3, 3), padding=((1, 1), (1, 1), (1, 1)), name='conv_1'
        )(h)
        h = nn.swish(nn.GroupNorm(num_groups=self.groups, dtype=self.dtype, name='norm_1')(h))
        if C != self.dim:
            x = nn.Conv(
                features=self.dim, kernel_size=(1, 1, 1), dtype=self.dtype, name='res_conv_0'
            )(x)
        assert x.shape == h.shape
        return x + h

class Attention3D(nn.Module):
    """Standard 3D multi-head attention mechanism.

    Computes attention over the 3D spatial dimensions (depth, height, width) using
    multi-head self-attention with normalization.

    Attributes:
        heads (int): Number of attention heads (default: 4).
        dim_head (int): Dimensionality of each head (default: 32).
        scale (int): Scaling factor for attention scores (default: 10).
        dtype (Any): Data type for computations (default: jnp.float32).

    Args:
        x (jnp.ndarray): Input tensor, shape (batch_size, depth, height, width, channels).

    Returns:
        jnp.ndarray: Output tensor, shape (batch_size, depth, height, width, channels).
    """
    heads: int = 4
    dim_head: int = 32
    scale: int = 10
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, D, H, W, C = x.shape
        dim = self.dim_head * self.heads
        qkv = nn.Conv(
            features=dim * 3, kernel_size=(1, 1, 1), use_bias=False, dtype=self.dtype, name='to_qkv.conv_0'
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b d h w (h d) -> b (d h w) h d', h=self.heads), (q, k, v))
        assert q.shape == k.shape == v.shape == (B, D * H * W, self.heads, self.dim_head)
        q = q / jnp.clip(jnp.linalg.norm(q, ord=2, axis=-1, keepdims=True), a_min=1e-12)
        k = k / jnp.clip(jnp.linalg.norm(k, ord=2, axis=-1, keepdims=True), a_min=1e-12)
        sim = jnp.einsum('b i h d, b j h d -> b h i j', q, k) * self.scale
        attn = nn.softmax(sim, axis=-1)
        out = jnp.einsum('b h i j, b j h d -> b h i d', attn, v)
        out = rearrange(out, 'b h (d h w) d -> b d h w (h d)', d=D, h=H, w=W)
        out = nn.Conv(features=C, kernel_size=(1, 1, 1), dtype=self.dtype, name='to_out.conv_0')(out)
        return out

class LinearAttention3D(nn.Module):
    """Linear 3D multi-head attention mechanism.

    Implements a memory-efficient attention mechanism by applying softmax to
    queries and keys separately, reducing memory usage for large 3D inputs.

    Attributes:
        heads (int): Number of attention heads (default: 4).
        dim_head (int): Dimensionality of each head (default: 32).
        dtype (Any): Data type for computations (default: jnp.float32).

    Args:
        x (jnp.ndarray): Input tensor, shape (batch_size, depth, height, width, channels).

    Returns:
        jnp.ndarray: Output tensor, shape (batch_size, depth, height, width, channels).
    """
    heads: int = 4
    dim_head: int = 32
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, D, H, W, C = x.shape
        dim = self.dim_head * self.heads
        qkv = nn.Conv(
            features=dim * 3, kernel_size=(1, 1, 1), use_bias=False, dtype=self.dtype, name='to_qkv.conv_0'
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b d h w (h d) -> b (d h w) h d', h=self.heads), (q, k, v))
        assert q.shape == k.shape == v.shape == (B, D * H * W, self.heads, self.dim_head)
        q = nn.softmax(q, axis=-1)
        k = nn.softmax(k, axis=-3)
        q = q / jnp.sqrt(self.dim_head)
        v = v / (D * H * W)
        context = jnp.einsum('b n h d, b n h e -> b h d e', k, v)
        out = jnp.einsum('b h d e, b n h d -> b h e n', context, q)
        out = rearrange(out, 'b h e (d h w) -> b d h w (h e)', d=D, h=H, w=W)
        out = nn.Conv(features=C, kernel_size=(1, 1, 1), dtype=self.dtype, name='to_out.conv_0')(out)
        out = nn.LayerNorm(epsilon=1e-5, use_bias=False, dtype=self.dtype, name='to_out.norm_0')(out)
        return out

class AttnBlock3D(nn.Module):
    """A 3D attention block with residual connection.

    Applies either standard or linear attention after layer normalization,
    followed by a residual connection to the input.

    Attributes:
        heads (int): Number of attention heads (default: 4).
        dim_head (int): Dimensionality of each head (default: 32).
        use_linear_attention (bool): If True, uses LinearAttention3D; else, Attention3D
            (default: True).
        dtype (Any): Data type for computations (default: jnp.float32).

    Args:
        x (jnp.ndarray): Input tensor, shape (batch_size, depth, height, width, channels).

    Returns:
        jnp.ndarray: Output tensor, shape (batch_size, depth, height, width, channels).
    """
    heads: int = 4
    dim_head: int = 32
    use_linear_attention: bool = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, D, H, W, C = x.shape
        normed_x = nn.LayerNorm(epsilon=1e-5, use_bias=False, dtype=self.dtype)(x)
        attn = LinearAttention3D(self.heads, self.dim_head, dtype=self.dtype) if self.use_linear_attention else Attention3D(self.heads, self.dim_head, dtype=self.dtype)
        out = attn(normed_x)
        assert out.shape == (B, D, H, W, C)
        return out + x

class UNet3D(nn.Module):
    """A 3D UNet for diffusion-based generation of voxel embeddings.

    Implements a UNet with downsampling and upsampling paths, incorporating
    time-conditioned residual blocks and attention mechanisms. Used to predict
    noise or clean data in the diffusion process.

    Attributes:
        dim (int): Base channel dimension for the UNet.
        init_dim (Optional[int]): Initial channel dimension. If None, uses dim.
        out_dim (Optional[int]): Output channel dimension. If None, matches input
            channels or doubles if learned_variance is True.
        dim_mults (Tuple[int, ...]): Multipliers for channel dimensions in
            downsampling/upsampling.
        resnet_block_groups (int): Number of groups for group normalization in
            ResnetBlock3D (default: 8).
        learned_variance (bool): If True, predicts variance alongside mean
            (default: False).
        dtype (Any): Data type for computations (default: jnp.float32).

    Args:
        x (jnp.ndarray): Input voxel embeddings, shape
            (batch_size, depth, height, width, embedding_dim).
        time (jnp.ndarray): Timestep indices, shape (batch_size,).

    Returns:
        jnp.ndarray: Predicted embeddings, shape
            (batch_size, depth, height, width, out_dim).
    """
    dim: int
    init_dim: Optional[int] = None
    out_dim: Optional[int] = None
    dim_mults: Tuple[int, ...]
    resnet_block_groups: int = 8
    learned_variance: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, time):
        B, D, H, W, C = x.shape
        init_dim = self.dim if self.init_dim is None else self.init_dim
        hs = []
        h = nn.Conv(
            features=init_dim, kernel_size=(7, 7, 7), padding=((3, 3), (3, 3), (3, 3)), name='init.conv_0', dtype=self.dtype
        )(x)
        hs.append(h)
        time_emb = SinusoidalPosEmb(self.dim, dtype=self.dtype)(time)
        time_emb = nn.Dense(features=self.dim * 4, dtype=self.dtype, name='time_mlp.dense_0')(time_emb)
        time_emb = nn.Dense(features=self.dim * 4, dtype=self.dtype, name='time_mlp.dense_1')(nn.gelu(time_emb))
        num_resolutions = len(self.dim_mults)
        for ind in range(num_resolutions):
            dim_in = h.shape[-1]
            h = ResnetBlock3D(dim=dim_in, groups=self.resnet_block_groups, dtype=self.dtype, name=f'down_{ind}.resblock_0')(h, time_emb)
            hs.append(h)
            h = ResnetBlock3D(dim=dim_in, groups=self.resnet_block_groups, dtype=self.dtype, name=f'down_{ind}.resblock_1')(h, time_emb)
            h = AttnBlock3D(dtype=self.dtype, name=f'down_{ind}.attnblock_0')(h)
            hs.append(h)
            if ind < num_resolutions - 1:
                h = Downsample3D(dim=self.dim * self.dim_mults[ind], dtype=self.dtype, name=f'down_{ind}.downsample_0')(h)
        mid_dim = self.dim * self.dim_mults[-1]
        h = nn.Conv(
            features=mid_dim, kernel_size=(3, 3, 3), padding=((1, 1), (1, 1), (1, 1)), dtype=self.dtype, name=f'down_{num_resolutions-1}.conv_0'
        )(h)
        h = ResnetBlock3D(dim=mid_dim, groups=self.resnet_block_groups, dtype=self.dtype, name='mid.resblock_0')(h, time_emb)
        h = AttnBlock3D(use_linear_attention=False, dtype=self.dtype, name='mid.attenblock_0')(h)
        h = ResnetBlock3D(dim=mid_dim, groups=self.resnet_block_groups, dtype=self.dtype, name='mid.resblock_1')(h, time_emb)
        for ind in reversed(range(num_resolutions)):
            dim_in = self.dim * self.dim_mults[ind]
            dim_out = self.dim * self.dim_mults[ind-1] if ind > 0 else init_dim
            h = jnp.concatenate([h, hs.pop()], axis=-1)
            h = ResnetBlock3D(dim=dim_in, groups=self.resnet_block_groups, dtype=self.dtype, name=f'up_{ind}.resblock_0')(h, time_emb)
            h = jnp.concatenate([h, hs.pop()], axis=-1)
            h = ResnetBlock3D(dim=dim_in, groups=self.resnet_block_groups, dtype=self.dtype, name=f'up_{ind}.resblock_1')(h, time_emb)
            h = AttnBlock3D(dtype=self.dtype, name=f'up_{ind}.attnblock_0')(h)
            if ind > 0:
                h = Upsample3D(dim=dim_out, dtype=self.dtype, name=f'up_{ind}.upsample_0')(h)
        h = nn.Conv(
            features=init_dim, kernel_size=(3, 3, 3), padding=((1, 1), (1, 1), (1, 1)), dtype=self.dtype, name='up_0.conv_0'
        )(h)
        h = jnp.concatenate([h, hs.pop()], axis=-1)
        out = ResnetBlock3D(dim=self.dim, groups=self.resnet_block_groups, dtype=self.dtype, name='final.resblock_0')(h, time_emb)
        default_out_dim = C * (1 if not self.learned_variance else 2)
        out_dim = default_out_dim if self.out_dim is None else self.out_dim
        return nn.Conv(
            out_dim, kernel_size=(1, 1, 1), dtype=self.dtype, name='final.conv_0'
        )(out)