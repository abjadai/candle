"""
export_to_onnx.py — Export Candle deduplication checkpoints to ONNX.

Accepts PyTorch Lightning .ckpt files directly — the same format produced
by the Candle training scripts (candle_pl.py / candle_pl_distill.py).
A Lightning checkpoint is a regular Python dict with at minimum:

    {
        'state_dict': { 'encoder.emb.tok_emb.weight': ..., ... },
        'optimizer_states': [...],
        'epoch': ...,
        ...
    }

Only 'state_dict' is used for export; optimizer states and training
metadata are ignored.  Teacher weights (present in distillation checkpoints)
are also stripped automatically.

Usage
-----
# Export the full (6-layer) model:
python export_to_onnx.py --checkpoint best_full_model.ckpt --output full_model.onnx

# Export the distilled (2-layer) model:
python export_to_onnx.py --checkpoint best_distilled_model.ckpt --output distilled_model.onnx --distilled

# Validate the exported model against PyTorch output:
python export_to_onnx.py --checkpoint best_full_model.ckpt --output full_model.onnx --validate

Place this script inside the candle-main/ directory so it can import
candle_pl and candle_tokenizer directly.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Imports from the candle repo (must run from candle-main/)
# ---------------------------------------------------------------------------
try:
    from candle_pl import CandleModel
    from candle_tokenizer import CandleTokenizer
except ImportError:
    sys.exit(
        "❌  Could not import candle_pl / candle_tokenizer.\n"
        "    Run this script from inside the candle-main/ directory."
    )

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    sys.exit(
        "❌  Missing dependencies.  Install them with:\n"
        "    pip install onnx onnxruntime"
    )

# ---------------------------------------------------------------------------
# Vocabulary — must match predict_cli.py exactly
# ---------------------------------------------------------------------------
LETTERS = [
    ' ', 'ش', 'ؤ', 'ء', 'ذ', 'إ', 'أ', 'ا', 'ض', 'ع', 'ح', 'ص', 'ط', 'ى',
    'ظ', 'ب', 'د', 'ف', 'غ', 'ه', 'ج', 'ك', 'ل', 'م', 'ن', 'ة', 'ق', 'ر',
    'س', 'ت', 'ث', 'و', 'خ', 'ي', 'ز', 'آ', 'ئ',
]

ACTUAL_MAX_SEQ_LEN = 256   # training sequence length
MODEL_MAX_SEQ_LEN  = 1024  # positional embedding table size
D_MODEL            = 512
N_HEADS            = 16
DROP_PROB          = 0.0   # disable dropout at export time
BLANK_SYMBOL       = '_'


# ---------------------------------------------------------------------------
# ONNX-safe attention patch
# ---------------------------------------------------------------------------
# Two issues prevent a clean ONNX export from the default transformer.py:
#
# 1. sdpa_kernel() context manager — the Flash / Efficient attention backends
#    do not trace through torch.onnx.export and produce NaN outputs.
#
# 2. Boolean attn_mask — F.scaled_dot_product_attention expects an *additive*
#    float mask (0.0 / -inf), but make_pad_mask() returns a BoolTensor.
#    PyTorch handles this internally at runtime but the ONNX tracer captures
#    the raw boolean tensor, which causes NaN in ONNXRuntime.
#
# Fix: monkey-patch MultiHeadAttention.forward on the model instance to use
# plain F.scaled_dot_product_attention with the mask converted to float,
# bypassing the sdpa_kernel context manager entirely.  We restore the
# original method after export so the model object is left unchanged.

from transformer import MultiHeadAttention   # noqa: E402  (local import)
import types                                  # noqa: E402


def _onnx_safe_mha_forward(self, q, k, v, mask=None):
    """Drop-in replacement for MultiHeadAttention.forward that is ONNX-safe."""
    q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
    q, k, v = self.split(q), self.split(k), self.split(v)

    # Convert boolean mask → additive float mask (0.0 keeps, -1e9 blocks).
    # We use -1e9 rather than -inf: ONNX Runtime computes softmax([-inf,...,-inf])
    # as NaN for fully-masked PAD rows, whereas softmax([-1e9,...,-1e9]) ≈ uniform
    # and stays finite.  The LayerNorm after attention normalises the difference away.
    float_mask = None
    if mask is not None:
        float_mask = torch.zeros_like(mask, dtype=q.dtype)
        float_mask = float_mask.masked_fill(mask == 0, -1e9)

    # Plain SDPA — no sdpa_kernel context manager
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=float_mask
    )
    out = self.concat(out)
    out = self.w_concat(out)
    return out


def _patch_attention(model: nn.Module) -> list:
    """Replace forward on every MultiHeadAttention layer; return originals."""
    originals = []
    for module in model.modules():
        if isinstance(module, MultiHeadAttention):
            originals.append((module, module.forward))
            module.forward = types.MethodType(_onnx_safe_mha_forward, module)
    return originals


def _unpatch_attention(originals: list) -> None:
    """Restore original forward methods."""
    for module, original_forward in originals:
        module.forward = original_forward


# ---------------------------------------------------------------------------
# ONNX wrapper
# ---------------------------------------------------------------------------

class CandleONNXWrapper(nn.Module):
    """
    Wraps CandleModel so torch.onnx.export sees a clean
    (src: int64[N, T]) → logits: float32[N, T, V] interface.
    """

    def __init__(self, model: CandleModel):
        super().__init__()
        self.model = model

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return self.model(src)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, n_layers: int) -> CandleModel:
    tokenizer = CandleTokenizer(LETTERS, max_seq_len=ACTUAL_MAX_SEQ_LEN)
    model = CandleModel(
        tokenizer,
        blank_symbol=BLANK_SYMBOL,
        max_seq_len=MODEL_MAX_SEQ_LEN,
        d_model=D_MODEL,
        n_layers=n_layers,
        n_heads=N_HEADS,
        drop_prob=DROP_PROB,
        learnable_pos_emb=False,
    )

    print(f"📂 Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # A PyTorch Lightning .ckpt is a dict; 'state_dict' holds model weights.
    # Other keys like 'optimizer_states', 'lr_schedulers', 'epoch',
    # 'global_step', and 'pytorch-lightning_version' are ignored.
    if "state_dict" not in ckpt:
        raise KeyError(
            f"'state_dict' key not found in checkpoint.\n"
            f"Top-level keys present: {list(ckpt.keys())}"
        )
    ckpt_keys = list(ckpt.keys())
    print(f"   Checkpoint keys : {ckpt_keys}")
    print(f"   Epoch           : {ckpt.get('epoch', 'n/a')}")
    print(f"   Global step     : {ckpt.get('global_step', 'n/a')}")

    # Strip teacher weights present in distillation checkpoints
    state_dict = {
        k: v for k, v in ckpt["state_dict"].items()
        if "teacher" not in k
    }
    print(f"   State-dict keys : {len(state_dict)} tensors loaded")

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"⚠️  Missing keys   : {missing}")
    if unexpected:
        print(f"⚠️  Unexpected keys: {unexpected}")

    model.eval()
    return model


def make_dummy_input(batch_size: int = 2,
                     seq_len: int = 32,
                     pad_token_id: int = 0) -> torch.Tensor:
    """
    A small random token-id tensor with a few PAD tokens so the mask
    logic is exercised during tracing.
    """
    src = torch.randint(1, 40, (batch_size, seq_len), dtype=torch.long)
    # Pad the last few positions of each sequence
    src[:, -4:] = pad_token_id
    return src


def export(model: CandleModel, output_path: str) -> None:
    wrapper = CandleONNXWrapper(model)
    dummy_src = make_dummy_input(pad_token_id=model.pad_token_id)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Patch all MultiHeadAttention layers to use the ONNX-safe forward,
    # then restore originals immediately after export.
    originals = _patch_attention(model)
    print(f"   Patched {len(originals)} MultiHeadAttention layer(s) for ONNX export")

    try:
        print(f"⚙️  Exporting to {output_path} ...")
        torch.onnx.export(
            wrapper,
            (dummy_src,),
            str(output_path),
            input_names=["src"],
            output_names=["logits"],
            dynamic_axes={
                "src":    {0: "batch_size", 1: "seq_len"},
                "logits": {0: "batch_size", 1: "seq_len"},
            },
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,   # force legacy TorchScript exporter — the dynamo
                            # exporter (default in PyTorch ≥2.9) ignores patches
                            # applied to instance methods and re-traces the
                            # original module graph
        )
    finally:
        _unpatch_attention(originals)

    print(f"✅ Saved: {output_path}")


def validate(model: CandleModel, output_path: str) -> None:
    """
    Check that the ONNX model and the PyTorch model agree numerically
    on a fresh random input.

    The patched (ONNX-safe) forward is used for the PyTorch reference run
    because that is what the ONNX graph was traced from.  Comparing against
    the unpatched model would show a large diff on GPU (Flash Attention handles
    the boolean mask differently from the additive float mask used in the graph).
    """
    print("🔍 Validating ONNX model against PyTorch ...")

    # --- ONNX graph check ---
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("   ONNX graph check: OK")

    # --- Numerical comparison ---
    src = make_dummy_input(batch_size=3, seq_len=24,
                           pad_token_id=model.pad_token_id)

    # Patch for the reference run so it matches the traced graph exactly
    originals = _patch_attention(model)
    try:
        with torch.no_grad():
            pt_logits = model(src).numpy()
    finally:
        _unpatch_attention(originals)

    session = ort.InferenceSession(output_path)
    ort_logits = session.run(None, {"src": src.numpy()})[0]

    max_diff = float(abs(pt_logits - ort_logits).max())
    print(f"   Max absolute difference (PyTorch vs ONNX): {max_diff:.2e}")

    threshold = 1e-4
    if max_diff < threshold:
        print(f"   ✅ Outputs match (threshold {threshold})")
    else:
        print(f"   ⚠️  Outputs differ by more than {threshold} — check export")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a Candle deduplication checkpoint to ONNX."
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to the PyTorch Lightning .ckpt checkpoint file.",
    )
    p.add_argument(
        "--output", required=True,
        help="Destination path for the exported .onnx file.",
    )
    p.add_argument(
        "--distilled", action="store_true",
        help="Set this flag for the 2-layer distilled model (default: 6-layer full model).",
    )
    p.add_argument(
        "--validate", action="store_true",
        help="After export, validate ONNX output against PyTorch numerically.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_layers = 2 if args.distilled else 6
    variant  = "distilled (2-layer)" if args.distilled else "full (6-layer)"
    print(f"🚀 Exporting {variant} model")

    model = load_model(args.checkpoint, n_layers)
    export(model, args.output)

    if args.validate:
        validate(model, args.output)

    print()
    print("Done!  To package for candle_deduplicator, zip the .onnx file:")
    print(f"  zip model.zip {args.output}")
    print("Then upload model.zip to your GitHub Release and update")
    print("_DOWNLOAD_URL in candle_deduplicator/models.py.")


if __name__ == "__main__":
    main()
