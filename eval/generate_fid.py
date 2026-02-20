"""Generate 4096 images with SiT-XL/2 REPA (Flax checkpoint) and compute FID.

Usage on Kaggle (TPU v5e-8 or GPU):
    python generate_fid.py \
        --ckpt-dir  /kaggle/input/repa-flax/flax_ckpt \
        --ref-dir   /kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/test \
        --out-dir   /kaggle/working/generated \
        --num-images 4096 \
        --batch-size 64 \
        --num-steps 250 \
        --cfg-scale 1.8 \
        --guidance-high 0.7 \
        --seed 0

Expected FID for correct weights: ~2.0-3.0 (SDE 250 steps, cfg=1.8 on 4K samples).
Official 50K FID = 1.42 (from REPA paper with optimal settings).
"""

# ── built-in ──────────────────────────────────────────────────────────────────
import argparse
import math
import os
import sys
from pathlib import Path

# ── numeric / image ───────────────────────────────────────────────────────────
import numpy as np
from PIL import Image
from tqdm import tqdm

# ── JAX / Flax ────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from etils import epath

# ── PyTorch (for VAE decode + FID) ────────────────────────────────────────────
import torch
from diffusers import AutoencoderKL          # pip install diffusers
from torchvision import transforms
import torchvision.transforms.functional as TF

# ── diffuse_nnx model ─────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_WORKSPACE = _HERE.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from networks.transformers.dit_nnx import DiT, Buffer


# ── SiT-XL/2 config matching the converted checkpoint ────────────────────────
_MODEL_CFG = dict(
    input_size=32,            # 256×256 / 8 = 32 latent grid
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
    enable_dropout=False,     # inference
)

# VAE scale factor used by REPA
_LATENT_SCALE = 0.18215


# ─────────────────────────────────────────────────────────────────────────────
# 1. Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_dir: str) -> DiT:
    """Load DiT weights from an orbax checkpoint produced by convert_repa_to_nnx.py."""
    print(f"Loading Flax checkpoint from: {ckpt_dir}")
    rngs  = nnx.Rngs(0)
    model = DiT(**_MODEL_CFG, rngs=rngs)

    abstract_state = nnx.state(model, (nnx.Param, Buffer))
    ckptr  = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
    out    = epath.Path(Path(ckpt_dir).resolve())
    result = ckptr.restore(
        out,
        args=ocp.args.Composite(model=ocp.args.StandardRestore(abstract_state)),
    )
    nnx.update(model, result.model)
    print(f"  DiT params loaded.  "
          f"Param count: {sum(v.size for v in jax.tree.leaves(nnx.state(model, nnx.Param))):,}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. SDE sampler  (exact JAX port of REPA's euler_maruyama_sampler)
# ─────────────────────────────────────────────────────────────────────────────

def _get_score_from_velocity(v: jnp.ndarray, x: jnp.ndarray, t: float) -> jnp.ndarray:
    """Convert velocity v_t to score ∇ log p_t for the linear path.

    Linear transport: x_t = (1-t)x + t·ε
        α_t = 1-t,  dα/dt = -1
        σ_t = t,    dσ/dt =  1
    => score = (-(1-t)·v - x) / t
    """
    ndim_extra = v.ndim - 1
    t_b = jnp.array(t).reshape((1,) + (1,) * ndim_extra)
    return (-((1.0 - t_b) * v + x)) / t_b


def _model_step(
    model: DiT,
    x: jnp.ndarray,
    t_scalar: float,
    y: jnp.ndarray,
    y_null: jnp.ndarray,
    cfg_scale: float,
    guidance_high: float,
    guidance_low: float,
) -> jnp.ndarray:
    """Evaluate model + CFG, return drift (without diffusion noise term)."""
    B = x.shape[0]
    t_vec = jnp.ones(B) * t_scalar
    use_cfg = cfg_scale > 1.0 and guidance_low <= t_scalar <= guidance_high

    if use_cfg:
        x_in = jnp.concatenate([x, x], axis=0)
        y_in = jnp.concatenate([y, y_null], axis=0)
        t_in = jnp.ones(2 * B) * t_scalar
    else:
        x_in, y_in, t_in = x, y, t_vec

    # raw velocity from DiT  (channels-last: (B, H, W, C))
    v_out, _ = model(x_in, t_in, y_in)

    diffusion = 2.0 * t_scalar

    if use_cfg:
        v_cond, v_uncond = v_out[:B], v_out[B:]
        s_cond   = _get_score_from_velocity(v_cond,   x, t_scalar)
        s_uncond = _get_score_from_velocity(v_uncond,  x, t_scalar)
        d_cond   = v_cond   - 0.5 * diffusion * s_cond
        d_uncond = v_uncond - 0.5 * diffusion * s_uncond
        drift = d_uncond + cfg_scale * (d_cond - d_uncond)
    else:
        s     = _get_score_from_velocity(v_out, x, t_scalar)
        drift = v_out - 0.5 * diffusion * s

    return drift


def euler_maruyama_sample(
    model: DiT,
    shape: tuple,
    y: jnp.ndarray,
    *,
    rng: jnp.ndarray,
    num_steps: int = 250,
    cfg_scale: float = 1.8,
    guidance_high: float = 0.7,
    guidance_low: float = 0.0,
) -> jnp.ndarray:
    """Stochastic Euler-Maruyama sampler, JAX port of REPA's euler_maruyama_sampler.

    Returns latents of shape `shape` (channels-last, B×H×W×C).
    """
    rng, init_key = jax.random.split(rng)
    x = jax.random.normal(init_key, shape)          # start from Gaussian noise

    y_null = jnp.ones(shape[0], dtype=jnp.int32) * 1000  # null class for CFG

    # time grid: [1.0, ..., 0.04] then append 0.0   (same as REPA)
    t_main  = np.linspace(1.0, 0.04, num_steps, dtype=np.float64)
    t_steps = np.concatenate([t_main, [0.0]])

    # ── stochastic steps (all but last) ──
    for i in range(len(t_steps) - 2):
        t_cur  = float(t_steps[i])
        t_next = float(t_steps[i + 1])
        dt     = t_next - t_cur
        diffusion = 2.0 * t_cur

        drift = _model_step(
            model, x, t_cur, y, y_null, cfg_scale, guidance_high, guidance_low
        )
        rng, noise_key = jax.random.split(rng)
        eps  = jax.random.normal(noise_key, x.shape)
        x    = x + drift * dt + math.sqrt(diffusion) * eps * math.sqrt(abs(dt))

    # ── deterministic last step (returns conditional mean) ──
    t_cur  = float(t_steps[-2])
    t_next = float(t_steps[-1])
    dt     = t_next - t_cur
    drift  = _model_step(
        model, x, t_cur, y, y_null, cfg_scale, guidance_high, guidance_low
    )
    x = x + drift * dt

    return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. VAE decode  (PyTorch diffusers — GPU if available, else CPU)
# ─────────────────────────────────────────────────────────────────────────────

def load_vae(vae_name: str = "stabilityai/sd-vae-ft-ema") -> AutoencoderKL:
    print(f"Loading VAE: {vae_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(vae_name).to(device)
    vae.eval()
    print(f"  VAE on {device}")
    return vae


@torch.no_grad()
def decode_latents(vae: AutoencoderKL, latents_jax: jnp.ndarray) -> np.ndarray:
    """Decode a batch of JAX latents (B,H,W,C channels-last) → uint8 RGB (B,H,W,3)."""
    device = next(vae.parameters()).device
    # JAX channels-last → PyTorch channels-first
    lat = np.asarray(latents_jax)              # (B, H, W, C)
    lat = torch.from_numpy(lat).permute(0, 3, 1, 2).float().to(device)  # (B,C,H,W)
    lat = lat / _LATENT_SCALE                  # undo REPA's normalisation
    images = vae.decode(lat).sample            # (B, 3, H, W) in [-1, 1]
    images = (images / 2 + 0.5).clamp(0, 1)   # [0, 1]
    images = (images * 255).byte().permute(0, 2, 3, 1).cpu().numpy()  # (B,H,W,3) uint8
    return images


# ─────────────────────────────────────────────────────────────────────────────
# 4. Reference images from ImageNet test set
# ─────────────────────────────────────────────────────────────────────────────

def collect_ref_images(ref_dir: str, num_images: int, out_dir: str, seed: int = 0) -> str:
    """Copy/resize `num_images` random ImageNet test images to `out_dir/ref/`."""
    ref_out = Path(out_dir) / "ref"
    ref_out.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(Path(ref_dir).rglob("*.JPEG")) + sorted(Path(ref_dir).rglob("*.jpg"))
    if len(all_imgs) == 0:
        raise FileNotFoundError(f"No JPEG images found under {ref_dir}")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(all_imgs, size=min(num_images, len(all_imgs)), replace=False)

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(256),
    ])

    print(f"Preparing {len(chosen)} reference images → {ref_out}")
    for i, p in enumerate(tqdm(chosen, desc="Ref images")):
        img = Image.open(p).convert("RGB")
        img = transform(img)
        img.save(ref_out / f"{i:06d}.png")

    return str(ref_out)


# ─────────────────────────────────────────────────────────────────────────────
# 5. FID computation  (scipy-based, works without extra GPU)
# ─────────────────────────────────────────────────────────────────────────────

def _inception_features(image_dir: str, batch_size: int = 64, device: str = "cpu") -> np.ndarray:
    """Extract Inception-v3 pool3 features for all images in a directory."""
    import torchvision.models as tvm
    from torchvision.models import Inception_V3_Weights

    model = tvm.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()       # return pool3 features (2048-d)
    model.aux_logits = False
    model.eval().to(device)

    preprocess = transforms.Compose([
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    paths = sorted(Path(image_dir).glob("*.png")) + sorted(Path(image_dir).glob("*.jpg"))
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc=f"Inception ({Path(image_dir).name})"):
        batch = torch.stack([
            preprocess(Image.open(p).convert("RGB")) for p in paths[i : i + batch_size]
        ]).to(device)
        with torch.no_grad():
            f = model(batch)
        feats.append(f.cpu().numpy())

    return np.concatenate(feats, axis=0)   # (N, 2048)


def compute_fid(real_dir: str, fake_dir: str, batch_size: int = 64) -> float:
    """Compute FID between images in `real_dir` and `fake_dir` using scipy."""
    from scipy.linalg import sqrtm

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Extracting Inception features for real images …")
    real_feats = _inception_features(real_dir, batch_size, device)
    print("Extracting Inception features for generated images …")
    fake_feats = _inception_features(fake_dir, batch_size, device)

    mu_r, sigma_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)

    diff  = mu_r - mu_f
    # sqrtm of sigma_r @ sigma_f
    covmean, _ = sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.imag(covmean), 0, atol=1e-3):
            print("Warning: imaginary component in sqrtm")
        covmean = np.real(covmean)

    fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))
    return fid


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def generate_images(
    model: DiT,
    vae: AutoencoderKL,
    out_dir: str,
    num_images: int,
    batch_size: int,
    num_steps: int,
    cfg_scale: float,
    guidance_high: float,
    guidance_low: float,
    seed: int,
    num_classes: int = 1000,
) -> str:
    gen_out = Path(out_dir) / "gen"
    gen_out.mkdir(parents=True, exist_ok=True)

    # Latent grid: 256×256 image → 32×32×4 latent (patch_size=2 → 16×16 tokens)
    latent_h = latent_w = _MODEL_CFG["input_size"]   # 32
    latent_c = _MODEL_CFG["in_channels"]              # 4

    total_gen = 0
    rng_key   = jax.random.PRNGKey(seed)

    # sample random class labels covering all 1000 ImageNet classes evenly
    labels_all = np.tile(np.arange(num_classes), math.ceil(num_images / num_classes))[:num_images]
    np.random.default_rng(seed).shuffle(labels_all)

    print(f"\nGenerating {num_images} images  (batch={batch_size}, steps={num_steps}, "
          f"cfg={cfg_scale}, guidance_high={guidance_high}) …")

    for start in tqdm(range(0, num_images, batch_size), desc="Generating"):
        end  = min(start + batch_size, num_images)
        bsz  = end - start

        y    = jnp.array(labels_all[start:end], dtype=jnp.int32)
        shape = (bsz, latent_h, latent_w, latent_c)

        rng_key, sample_key = jax.random.split(rng_key)
        latents = euler_maruyama_sample(
            model, shape, y,
            rng=sample_key,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            guidance_high=guidance_high,
            guidance_low=guidance_low,
        )
        images = decode_latents(vae, latents)   # (B, 256, 256, 3) uint8

        for i, img_arr in enumerate(images):
            idx = total_gen + i
            Image.fromarray(img_arr).save(gen_out / f"{idx:06d}.png")

        total_gen += bsz

    print(f"Saved {total_gen} generated images to {gen_out}")
    return str(gen_out)


def main():
    parser = argparse.ArgumentParser(description="SiT-XL/2 REPA Flax → generate + FID")

    parser.add_argument("--ckpt-dir",       default="REPA/pretrained_models/flax_ckpt",
                        help="Path to orbax checkpoint directory")
    parser.add_argument("--ref-dir",
                        default="/kaggle/input/competitions/imagenet-object-localization-challenge"
                                "/ILSVRC/Data/CLS-LOC/test",
                        help="Directory containing reference ImageNet test images")
    parser.add_argument("--out-dir",        default="fid_eval",
                        help="Output directory for generated images and reference subset")
    parser.add_argument("--num-images",     type=int,   default=4096)
    parser.add_argument("--batch-size",     type=int,   default=64,
                        help="Sampling batch size (lower if OOM)")
    parser.add_argument("--num-steps",      type=int,   default=250)
    parser.add_argument("--cfg-scale",      type=float, default=1.8)
    parser.add_argument("--guidance-high",  type=float, default=0.7)
    parser.add_argument("--guidance-low",   type=float, default=0.0)
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--vae",            default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--skip-generate",  action="store_true",
                        help="Skip generation, only compute FID (assumes images already exist)")
    parser.add_argument("--inception-batch-size", type=int, default=64)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")

    # ── Step 1: load model ──────────────────────────────────────────────────
    model = load_model(args.ckpt_dir)

    # ── Step 2: load VAE ────────────────────────────────────────────────────
    vae = load_vae(args.vae)

    # ── Step 3: generate images ─────────────────────────────────────────────
    gen_dir = str(Path(args.out_dir) / "gen")
    if not args.skip_generate:
        gen_dir = generate_images(
            model, vae,
            out_dir=args.out_dir,
            num_images=args.num_images,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            guidance_high=args.guidance_high,
            guidance_low=args.guidance_low,
            seed=args.seed,
        )

    # ── Step 4: prepare reference images ────────────────────────────────────
    ref_dir = collect_ref_images(
        args.ref_dir, args.num_images, args.out_dir, seed=args.seed
    )

    # ── Step 5: compute FID ─────────────────────────────────────────────────
    print("\nComputing FID …")
    fid = compute_fid(ref_dir, gen_dir, batch_size=args.inception_batch_size)
    print(f"\n{'='*50}")
    print(f"  FID ({args.num_images // 1000}K) = {fid:.4f}")
    print(f"{'='*50}")
    print("Reference: official REPA SiT-XL/2 50K FID ≈ 1.42 (SDE 250 steps, cfg=1.8)")
    print("Expected ~4K FID with correct weights: ~2–5 (higher due to fewer samples)")


if __name__ == "__main__":
    main()
