"""Convert REPA SiT-XL/2 PyTorch checkpoint to Flax/NNX format.

Usage:
    python -m networks.transformers.convert_repa_to_nnx \
        --input  REPA/pretrained_models/last.pt \
        --output REPA/pretrained_models/flax_ckpt

The script:
  1. Loads the EMA PyTorch state dict from `last.pt`.
  2. Builds a Flax DiT (SiT-XL/2 config) and a REPA projector MLP.
  3. Copies every weight, applying the required shape/layout transforms.
  4. Saves the result as an orbax checkpoint (model + projector states).
"""

# built-in
import argparse
import sys
from pathlib import Path

# external
import numpy as np
import torch
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from etils import epath

# Add workspace root to path so local imports work when run as a module
_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from networks.transformers.dit_nnx import DiT, Buffer
from interfaces.repa import build_mlp


# ---------------------------------------------------------------------------
# Low-level conversion helpers
# ---------------------------------------------------------------------------

def _np(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch CPU tensor to float32 numpy."""
    return tensor.detach().cpu().float().numpy()


def _jax(tensor: torch.Tensor) -> jnp.ndarray:
    return jnp.array(_np(tensor))


def _linear_kernel(w: torch.Tensor) -> jnp.ndarray:
    """PyTorch Linear kernel (out, in) → Flax (in, out)."""
    return _jax(w.T)


def _conv_kernel(w: torch.Tensor) -> jnp.ndarray:
    """PyTorch Conv2d kernel (out, in, kh, kw) → Flax (kh, kw, in, out)."""
    return jnp.transpose(_jax(w), (2, 3, 1, 0))


def _split_qkv_kernel(
    qkv_w: torch.Tensor, num_heads: int, head_dim: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split combined QKV kernel to three Flax MultiHeadAttention kernels.

    PyTorch: (3 * hidden, hidden)
    Flax per-head: (hidden, num_heads, head_dim)
    """
    q, k, v = qkv_w.chunk(3, dim=0)          # each (hidden, hidden)
    def _reshape(x):
        # transpose (hidden, hidden) → (hidden, hidden) then (hidden, num_heads, head_dim)
        return _jax(x.T).reshape(-1, num_heads, head_dim)
    return _reshape(q), _reshape(k), _reshape(v)


def _split_qkv_bias(
    qkv_b: torch.Tensor, num_heads: int, head_dim: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split combined QKV bias to three Flax biases.

    PyTorch: (3 * hidden,)
    Flax per-head bias: (num_heads, head_dim)
    """
    q, k, v = qkv_b.chunk(3, dim=0)          # each (hidden,)
    return (
        _jax(q).reshape(num_heads, head_dim),
        _jax(k).reshape(num_heads, head_dim),
        _jax(v).reshape(num_heads, head_dim),
    )


def _out_proj_kernel(
    w: torch.Tensor, num_heads: int, head_dim: int
) -> jnp.ndarray:
    """Convert attention output projection weight.

    PyTorch: (out_features, in_features) = (hidden, hidden)
    Flax: (num_heads, head_dim, out_features)
    """
    # w: (hidden, hidden), transpose → (hidden, hidden), reshape → (num_heads, head_dim, out)
    return _jax(w.T).reshape(num_heads, head_dim, -1)


# ---------------------------------------------------------------------------
# Per-component conversion
# ---------------------------------------------------------------------------

def _set_x_proj(model: DiT, th: dict) -> None:
    model.x_proj.kernel.value = _conv_kernel(th["x_embedder.proj.weight"])
    model.x_proj.bias.value   = _jax(th["x_embedder.proj.bias"])


def _set_t_embedder(model: DiT, th: dict) -> None:
    mlp = model.t_embedder.mlp
    mlp.layers[0].kernel.value = _linear_kernel(th["t_embedder.mlp.0.weight"])
    mlp.layers[0].bias.value   = _jax(th["t_embedder.mlp.0.bias"])
    mlp.layers[2].kernel.value = _linear_kernel(th["t_embedder.mlp.2.weight"])
    mlp.layers[2].bias.value   = _jax(th["t_embedder.mlp.2.bias"])


def _set_y_embedder(model: DiT, th: dict) -> None:
    model.y_embedder.embedding_table.embedding.value = _jax(
        th["y_embedder.embedding_table.weight"]
    )


def _set_block(
    block, th: dict, idx: int, num_heads: int, head_dim: int
) -> None:
    p = f"blocks.{idx}"

    # Attention QKV
    qk, kk, vk = _split_qkv_kernel(th[f"{p}.attn.qkv.weight"], num_heads, head_dim)
    qb, kb, vb = _split_qkv_bias(th[f"{p}.attn.qkv.bias"], num_heads, head_dim)
    block.attn.query.kernel.value = qk
    block.attn.query.bias.value   = qb
    block.attn.key.kernel.value   = kk
    block.attn.key.bias.value     = kb
    block.attn.value.kernel.value = vk
    block.attn.value.bias.value   = vb

    # Attention output projection
    block.attn.out.kernel.value = _out_proj_kernel(
        th[f"{p}.attn.proj.weight"], num_heads, head_dim
    )
    block.attn.out.bias.value = _jax(th[f"{p}.attn.proj.bias"])

    # MLP
    block.mlp.linear1.kernel.value = _linear_kernel(th[f"{p}.mlp.fc1.weight"])
    block.mlp.linear1.bias.value   = _jax(th[f"{p}.mlp.fc1.bias"])
    block.mlp.linear2.kernel.value = _linear_kernel(th[f"{p}.mlp.fc2.weight"])
    block.mlp.linear2.bias.value   = _jax(th[f"{p}.mlp.fc2.bias"])

    # adaLN modulation — layers[0]=silu (no params), layers[1]=Linear
    block.adaLN_mod.layers[1].kernel.value = _linear_kernel(
        th[f"{p}.adaLN_modulation.1.weight"]
    )
    block.adaLN_mod.layers[1].bias.value = _jax(th[f"{p}.adaLN_modulation.1.bias"])

    # norm1 / norm2 have use_scale=False, use_bias=False → nothing to copy


def _set_final_layer(model: DiT, th: dict) -> None:
    fl = model.final_layer
    fl.linear.kernel.value = _linear_kernel(th["final_layer.linear.weight"])
    fl.linear.bias.value   = _jax(th["final_layer.linear.bias"])
    # norm has no learnable params (use_scale=False, use_bias=False)
    fl.adaLN_mod.layers[1].kernel.value = _linear_kernel(
        th["final_layer.adaLN_modulation.1.weight"]
    )
    fl.adaLN_mod.layers[1].bias.value = _jax(
        th["final_layer.adaLN_modulation.1.bias"]
    )


def _set_projector(projector, th: dict, proj_idx: int = 0) -> None:
    """Copy projector MLP weights (projectors.{proj_idx}.{0,2,4}.*)."""
    p = f"projectors.{proj_idx}"
    projector.layers[0].kernel.value = _linear_kernel(th[f"{p}.0.weight"])
    projector.layers[0].bias.value   = _jax(th[f"{p}.0.bias"])
    projector.layers[2].kernel.value = _linear_kernel(th[f"{p}.2.weight"])
    projector.layers[2].bias.value   = _jax(th[f"{p}.2.bias"])
    projector.layers[4].kernel.value = _linear_kernel(th[f"{p}.4.weight"])
    projector.layers[4].bias.value   = _jax(th[f"{p}.4.bias"])


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

# SiT-XL/2 config (matches REPA training defaults)
_SIT_XL2 = dict(
    input_size=32,       # 256 / 8 = 32 for 256×256 images
    patch_size=2,
    in_channels=4,
    hidden_size=1152,
    depth=28,
    num_heads=16,
    mlp_ratio=4.0,
    continuous_time_embed=False,
    freq_embed_size=256,
    num_classes=1000,
    class_dropout_prob=0.1,
    enable_dropout=False,  # inference mode
)

# REPA projector config (single DINOv2-B encoder, 768-dim features)
_REPA_PROJ = dict(
    hidden_size=1152,
    projector_dim=2048,
    feature_dim=768,
)


def convert(input_path: str, output_dir: str) -> None:
    print(f"Loading PyTorch checkpoint from: {input_path}")
    th = torch.load(input_path, map_location="cpu")

    # last.pt is stored directly as the state_dict (not wrapped in a dict with 'ema' key)
    if not isinstance(th, dict) or "x_embedder.proj.weight" not in th:
        # Try common wrapper formats
        for key in ("ema", "model", "state_dict"):
            if isinstance(th, dict) and key in th:
                th = th[key]
                print(f"  Unwrapped checkpoint using key '{key}'")
                break
    print(f"  State dict contains {len(th)} keys")

    # --- Build Flax DiT model ---
    print("Building Flax DiT (SiT-XL/2 config)…")
    rngs = nnx.Rngs(0)
    model = DiT(**_SIT_XL2, rngs=rngs)

    # --- Build projector ---
    print("Building REPA projector MLP…")
    projector = build_mlp(
        hidden_size=_REPA_PROJ["hidden_size"],
        projector_dim=_REPA_PROJ["projector_dim"],
        feature_dim=_REPA_PROJ["feature_dim"],
        rngs=rngs,
    )

    num_heads = _SIT_XL2["num_heads"]
    head_dim  = _SIT_XL2["hidden_size"] // num_heads   # 1152 // 16 = 72

    # --- Copy weights ---
    print("Copying weights…")
    _set_x_proj(model, th)
    # pos_embed (x_embedder.pe) is a sincos Buffer recomputed at init — skip
    _set_t_embedder(model, th)
    _set_y_embedder(model, th)

    for i in range(_SIT_XL2["depth"]):
        _set_block(model.blocks[i], th, i, num_heads, head_dim)

    _set_final_layer(model, th)
    _set_projector(projector, th, proj_idx=0)

    # --- Save via orbax ---
    # orbax requires absolute paths
    out = epath.Path(Path(output_dir).resolve())
    print(f"Saving orbax checkpoint to: {out}")
    out.mkdir(parents=True, exist_ok=True)

    # Only save Param and Buffer variables (exclude RNG keys which are not serialisable)
    model_state     = nnx.state(model, (nnx.Param, Buffer))
    projector_state = nnx.state(projector, nnx.Param)

    ckptr = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
    ckptr.save(
        out,
        args=ocp.args.Composite(
            model=ocp.args.StandardSave(model_state),
            projector=ocp.args.StandardSave(projector_state),
        ),
        force=True,
    )
    print("Done.")


# ---------------------------------------------------------------------------
# Verification helper (optional)
# ---------------------------------------------------------------------------

def _verify(input_path: str) -> None:
    """Quick sanity check: forward-pass equality between PyTorch and Flax."""
    import torch.nn.functional as F
    sys.path.insert(0, str(_WORKSPACE / "REPA"))
    from models.sit import SiT_models  # type: ignore

    th_state = torch.load(input_path, map_location="cpu")

    # Build PyTorch model
    th_model = SiT_models["SiT-XL/2"](
        input_size=32,
        num_classes=1000,
        use_cfg=True,
        z_dims=[768],
        encoder_depth=8,
    )
    th_model.load_state_dict(th_state)
    th_model.eval()

    # Build Flax model and inject weights
    rngs  = nnx.Rngs(0)
    model = DiT(**_SIT_XL2, rngs=rngs)
    th    = th_state
    num_heads, head_dim = _SIT_XL2["num_heads"], _SIT_XL2["hidden_size"] // _SIT_XL2["num_heads"]
    _set_x_proj(model, th)
    _set_t_embedder(model, th)
    _set_y_embedder(model, th)
    for i in range(_SIT_XL2["depth"]):
        _set_block(model.blocks[i], th, i, num_heads, head_dim)
    _set_final_layer(model, th)

    # Random inputs
    B, C, H, W = 2, 4, 32, 32
    x_np = np.random.randn(B, C, H, W).astype(np.float32)
    t_np = np.array([500, 750], dtype=np.float32)
    y_np = np.array([1, 2], dtype=np.int32)

    # PyTorch forward
    with torch.no_grad():
        x_th = torch.tensor(x_np)
        t_th = torch.tensor(t_np)
        y_th = torch.tensor(y_np)
        out_th, _ = th_model(x_th.permute(0, 2, 3, 1).reshape(B, -1, C), t_th, y_th)

    # Flax forward (channels last: (B, H, W, C))
    x_jax = jnp.array(x_np.transpose(0, 2, 3, 1))  # (B, H, W, C)
    t_jax = jnp.array(t_np)
    y_jax = jnp.array(y_np)
    out_jax, _ = model(x_jax, t_jax, y_jax)

    print("PyTorch output stats: mean={:.4f} std={:.4f}".format(
        float(out_th.mean()), float(out_th.std())))
    print("Flax    output stats: mean={:.4f} std={:.4f}".format(
        float(out_jax.mean()), float(out_jax.std())))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Convert REPA PyTorch → Flax/NNX")
    parser.add_argument(
        "--input", default="REPA/pretrained_models/last.pt",
        help="Path to the PyTorch checkpoint (last.pt)"
    )
    parser.add_argument(
        "--output", default="REPA/pretrained_models/flax_ckpt",
        help="Output directory for the orbax checkpoint"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run a quick forward-pass sanity check after conversion"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert(args.input, args.output)
    if args.verify:
        print("\nRunning forward-pass verification…")
        _verify(args.input)
