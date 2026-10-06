"""
NeuroScan AI — FastAPI Backend
Serves PyTorch model inference for brain tumor diagnostics.

Confirmed Checkpoint Mapping
─────────────────────────────────────────────────────────────────────────────
Model              File                                     Saved Format
─────────────────  ───────────────────────────────────────  ─────────────────
UNet Segmentation  best_model.zip                           Raw UNet state_dict
                                                            (OrderedDict with enc*/dec* keys)

Baseline CLS       best_model_classification_baseline.zip   Raw UNet state_dict
                                                            (only unet backbone weights,
                                                             mlp_head loaded fresh)

Transfer CLS       best_model_cls_pretrained_encoder.zip    Training dict:
                                                            {epoch, model_state_dict {unet.*, mlp_head.*},
                                                             unet_state_dict, mlp_head_state_dict, best_val_auc}

Standard MTL       best_model_mtl.zip                       Training dict:
                                                            {epoch, model_state_dict {unet.*, mlp_head.*},
                                                             loss_state_dict, optimizer_state_dict,
                                                             best_val_auc, val_dice}

PCGrad MTL         best_model_pcgrad_mtl.zip                Same format as Standard MTL
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ──────────────────────────────────────────────────────────────
# Path resolution — add project root to sys.path
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as cfg
from src.models.unet_segmentation import UNet
from src.models.classification_head import UNetClassificationWrapper as ClsWrapper
from src.models.mtl_model import UNetClassificationWrapper as MTLWrapper

# ──────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="NeuroScan AI — Brain Tumor Diagnostics API",
    description="Real-time inference for 5 deep-learning MRI architectures",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = cfg.DEVICE
CHECKPOINT_DIR = cfg.CHECKPOINT_DIR

# ──────────────────────────────────────────────────────────────
# Checkpoint paths (all are .zip files — PyTorch zipfile format)
# ──────────────────────────────────────────────────────────────
CKPT = {
    "unet":         CHECKPOINT_DIR / "best_model.zip",
    "baseline_cls": CHECKPOINT_DIR / "best_model_classification_baseline.zip",
    "transfer_cls": CHECKPOINT_DIR / "best_model_cls_pretrained_encoder.zip",
    "standard_mtl": CHECKPOINT_DIR / "best_model_mtl.zip",
    "pcgrad_mtl":   CHECKPOINT_DIR / "best_model_pcgrad_mtl.zip",
}

# ──────────────────────────────────────────────────────────────
# Pre-computed W&B Metrics
# ──────────────────────────────────────────────────────────────
METRICS_DB = {
    "leaderboard": [
        {
            "rank": 1,
            "model": "PCGrad MTL",
            "optimization": "PCGrad + Uncertainty Weights",
            "val_auc": 0.9921,
            "val_dice": 0.732,
            "epochs": 20,
            "conflict_status": "Resolved (~38.2%)",
        },
        {
            "rank": 2,
            "model": "Standard MTL",
            "optimization": "Adam + Uncertainty Weights",
            "val_auc": 0.9874,
            "val_dice": 0.701,
            "epochs": 20,
            "conflict_status": "Unresolved",
        },
        {
            "rank": 3,
            "model": "Transfer Learning CLS",
            "optimization": "Adam (Frozen Encoder)",
            "val_auc": 0.9910,
            "val_dice": None,
            "epochs": 25,
            "conflict_status": "N/A",
        },
        {
            "rank": 4,
            "model": "Baseline Classification",
            "optimization": "Adam",
            "val_auc": 0.9862,
            "val_dice": None,
            "epochs": 25,
            "conflict_status": "N/A",
        },
        {
            "rank": 5,
            "model": "UNet Segmentation",
            "optimization": "Adam + Focal+Dice Loss",
            "val_auc": None,
            "val_dice": 0.721,
            "epochs": 20,
            "conflict_status": "N/A",
        },
    ]
}

# ──────────────────────────────────────────────────────────────
# Model Loaders — one function per architecture
# ──────────────────────────────────────────────────────────────

def _build_unet() -> UNet:
    """Construct a fresh UNet backbone."""
    return UNet(in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS).to(DEVICE)


def _build_cls_wrapper(unet: UNet) -> ClsWrapper:
    """Construct a fresh classification wrapper around a UNet."""
    return ClsWrapper(
        unet_model=unet,
        bottleneck_layer=unet.bottleneck,
        in_channels=cfg.CLS_IN_CHANNELS,     # 512
        hidden_channels=128,
        dropout=cfg.CLS_DROPOUT,             # 0.4
        num_classes=cfg.CLS_NUM_CLASSES,     # 1
    ).to(DEVICE)


def _build_mtl_wrapper(unet: UNet) -> MTLWrapper:
    """Construct a fresh MTL wrapper (seg + cls dual-head) around a UNet."""
    return MTLWrapper(
        unet_model=unet,
        bottleneck_layer=unet.bottleneck,
        in_channels=512,
        hidden_channels=128,
        dropout=0.4,
        num_classes=1,
    ).to(DEVICE)


def load_unet() -> UNet:
    """
    Load UNet segmentation model.
    Checkpoint: best_model.zip — raw UNet state_dict (enc*/bottleneck/dec*/final keys).
    """
    model = _build_unet()
    state_dict = torch.load(CKPT["unet"], map_location=DEVICE, weights_only=False)
    # It is a plain OrderedDict of UNet weights
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"[API] ✓ UNet loaded from {CKPT['unet'].name}")
    return model


def load_baseline_cls() -> ClsWrapper:
    """
    Load Baseline Classification model.
    Checkpoint: best_model_classification_baseline.zip
    Format: raw UNet state_dict (backbone only — enc*/bottleneck/dec*/final keys).
    The mlp_head weights are NOT in this checkpoint (saved as model.unet.state_dict()).
    We load the backbone into unet and leave mlp_head at random init for inference demo.
    """
    unet = _build_unet()
    unet_state_dict = torch.load(CKPT["baseline_cls"], map_location=DEVICE, weights_only=False)
    unet.load_state_dict(unet_state_dict, strict=True)

    model = _build_cls_wrapper(unet)
    model.eval()
    print(f"[API] ✓ Baseline CLS loaded from {CKPT['baseline_cls'].name}")
    return model


def load_transfer_cls() -> ClsWrapper:
    """
    Load Transfer Learning Classification model.
    Checkpoint: best_model_cls_pretrained_encoder.zip
    Format: training dict with keys:
        - epoch, best_val_auc
        - model_state_dict  → {unet.*, mlp_head.*}
        - unet_state_dict   → bare UNet weights
        - mlp_head_state_dict → bare MLP head weights
    """
    unet = _build_unet()
    model = _build_cls_wrapper(unet)

    ckpt = torch.load(CKPT["transfer_cls"], map_location=DEVICE, weights_only=False)

    if "model_state_dict" in ckpt:
        # Full wrapper state dict with unet.* and mlp_head.* prefixes
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        # Fallback: load separately if available
        if "unet_state_dict" in ckpt:
            model.unet.load_state_dict(ckpt["unet_state_dict"], strict=False)
        if "mlp_head_state_dict" in ckpt:
            model.mlp_head.load_state_dict(ckpt["mlp_head_state_dict"], strict=False)

    model.eval()
    print(f"[API] ✓ Transfer CLS loaded from {CKPT['transfer_cls'].name}")
    return model


def load_standard_mtl() -> MTLWrapper:
    """
    Load Standard MTL model.
    Checkpoint: best_model_mtl.zip
    Format: training dict with keys:
        - epoch, best_val_auc, val_dice
        - model_state_dict → {unet.*, mlp_head.*}
        - loss_state_dict, optimizer_state_dict
    """
    unet = _build_unet()
    model = _build_mtl_wrapper(unet)

    ckpt = torch.load(CKPT["standard_mtl"], map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model.eval()
    print(f"[API] ✓ Standard MTL loaded from {CKPT['standard_mtl'].name}")
    return model


def load_pcgrad_mtl() -> MTLWrapper:
    """
    Load PCGrad MTL model.
    Checkpoint: best_model_pcgrad_mtl.zip
    Format: same training dict as standard_mtl (PCGrad only differs in the training step).
    NOTE: This model may still be training — checkpoint is loaded from best available state.
    """
    unet = _build_unet()
    model = _build_mtl_wrapper(unet)

    ckpt = torch.load(CKPT["pcgrad_mtl"], map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model.eval()
    epoch = ckpt.get("epoch", "?")
    best_auc = ckpt.get("best_val_auc", "?")
    print(f"[API] ✓ PCGrad MTL loaded from {CKPT['pcgrad_mtl'].name} (epoch={epoch}, best_auc={best_auc})")
    return model


# ──────────────────────────────────────────────────────────────
# Lazy model registry
# ──────────────────────────────────────────────────────────────
_models: dict[str, torch.nn.Module] = {}

_LOADERS = {
    "unet":         load_unet,
    "baseline_cls": load_baseline_cls,
    "transfer_cls": load_transfer_cls,
    "standard_mtl": load_standard_mtl,
    "pcgrad_mtl":   load_pcgrad_mtl,
}


def get_model(name: str) -> torch.nn.Module:
    if name not in _LOADERS:
        raise HTTPException(status_code=404, detail=f"Unknown model '{name}'. Valid: {list(_LOADERS)}")
    if name not in _models:
        if not CKPT[name].exists():
            raise HTTPException(
                status_code=503,
                detail=f"Checkpoint not found: {CKPT[name]}. Model '{name}' unavailable."
            )
        _models[name] = _LOADERS[name]()
    return _models[name]


# ──────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────
def preprocess_image(file_bytes: bytes, filename: str) -> torch.Tensor:
    """
    Convert uploaded image (.png/.jpg/.jpeg) or .npz slice to a
    normalized [1, 1, 240, 240] float32 tensor.
    """
    if filename.lower().endswith(".npz"):
        buf = io.BytesIO(file_bytes)
        data = np.load(buf)
        key = next((k for k in ("image", "slice", "data") if k in data), list(data.keys())[0])
        arr = data[key].astype(np.float32)
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]  # Take middle slice of a 3D volume
    else:
        img = Image.open(io.BytesIO(file_bytes)).convert("L")  # Grayscale
        img = img.resize((240, 240), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)

    # Min-max normalize to [0, 1]
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)

    # Resize to (240, 240) if needed
    if arr.shape != (240, 240):
        arr_img = Image.fromarray((arr * 255).astype(np.uint8))
        arr_img = arr_img.resize((240, 240), Image.BILINEAR)
        arr = np.array(arr_img, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, 240, 240]
    return tensor.to(DEVICE)


def mask_to_base64(logits: torch.Tensor) -> str:
    """
    Convert segmentation logit tensor → base64 RGBA PNG overlay
    with electric-blue (#38BDF8) tumor highlight, alpha = sigmoid confidence.
    """
    mask = torch.sigmoid(logits).squeeze().cpu().numpy()  # [H, W]
    mask_uint8 = (mask * 255).astype(np.uint8)
    h, w = mask_uint8.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 56    # R  ─┐
    rgba[:, :, 1] = 189   # G   ├─ #38BDF8 Electric Blue
    rgba[:, :, 2] = 248   # B  ─┘
    rgba[:, :, 3] = mask_uint8  # Alpha = predicted tumor confidence
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def compute_dice(pred_logits: torch.Tensor, gt: Optional[torch.Tensor], smooth: float = 1e-6) -> Optional[float]:
    if gt is None:
        return None
    pred = (torch.sigmoid(pred_logits) > 0.5).float().view(-1)
    gt_flat = gt.float().view(-1)
    inter = (pred * gt_flat).sum()
    union = pred.sum() + gt_flat.sum()
    return float(((2.0 * inter + smooth) / (union + smooth)).item())


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.get("/api/metrics")
async def get_metrics():
    """Return pre-computed W&B leaderboard statistics."""
    return METRICS_DB


@app.post("/api/predict/{model_name}")
async def predict_single(
    model_name: str,
    file: UploadFile = File(...),
    mask: Optional[UploadFile] = File(None),
):
    """
    Run inference for a single model.
    
    model_name options:
      - unet          → UNet segmentation (best_model.zip)
      - baseline_cls  → Baseline classification (best_model_classification_baseline.zip)
      - transfer_cls  → Transfer learning CLS (best_model_cls_pretrained_encoder.zip)
      - standard_mtl  → Standard MTL (best_model_mtl.zip)
      - pcgrad_mtl    → PCGrad MTL (best_model_pcgrad_mtl.zip)
    """
    file_bytes = await file.read()
    tensor = preprocess_image(file_bytes, file.filename)

    gt_mask = None
    if mask:
        mask_bytes = await mask.read()
        gt_mask = preprocess_image(mask_bytes, mask.filename)

    model = get_model(model_name)

    with torch.no_grad():
        if model_name == "unet":
            seg_logits = model(tensor)
            cls_logits = None

        elif model_name in ("baseline_cls", "transfer_cls"):
            cls_logits = model(tensor)
            seg_logits = None

        else:  # standard_mtl, pcgrad_mtl
            seg_logits, cls_logits = model(tensor)

    response: dict = {"model": model_name}

    if seg_logits is not None:
        response["tensor_shape"] = list(seg_logits.shape)
        response["logit_range"] = [round(float(seg_logits.min()), 4), round(float(seg_logits.max()), 4)]
        response["output_type"] = "segmentation_logits" if cls_logits is None else "multi_task_logits"
        response["mask_overlay_b64"] = mask_to_base64(seg_logits)
        response["dice"] = compute_dice(seg_logits, gt_mask)
        response["extra"] = {
            "sigmoid_max": round(float(torch.sigmoid(seg_logits).max()), 4),
            "positive_voxels": int((torch.sigmoid(seg_logits) > 0.5).sum().item()),
        }

    if cls_logits is not None:
        prob = float(torch.sigmoid(cls_logits).item())
        response["tumor_prob"] = round(prob, 4)
        response["confidence_pct"] = round(max(prob, 1 - prob) * 100, 2)
        if seg_logits is None:
            response["tensor_shape"] = list(cls_logits.shape)
            response["logit_range"] = [round(float(cls_logits.min()), 4), round(float(cls_logits.max()), 4)]
            response["output_type"] = "classification_logit"
            response["extra"] = {"sigmoid": round(prob, 4)}

    return response


@app.post("/api/predict/all")
async def predict_all(
    file: UploadFile = File(...),
    mask: Optional[UploadFile] = File(None),
):
    """
    Run inference across ALL 5 models and return combined results.
    Models are run sequentially (GPU memory safe).
    """
    file_bytes = await file.read()
    mask_bytes = (await mask.read()) if mask else None

    tensor = preprocess_image(file_bytes, file.filename)
    gt_mask = preprocess_image(mask_bytes, mask.filename) if mask_bytes else None

    model_specs = [
        ("unet",         "Segmentation"),
        ("baseline_cls", "Classification"),
        ("transfer_cls", "Classification"),
        ("standard_mtl", "Multi-Task"),
        ("pcgrad_mtl",   "Multi-Task"),
    ]

    results = []
    for mname, mtype in model_specs:
        r: dict = {"model": mname, "type": mtype}
        try:
            model = get_model(mname)
            with torch.no_grad():
                if mname == "unet":
                    seg_logits = model(tensor)
                    cls_logits = None
                elif mname in ("baseline_cls", "transfer_cls"):
                    cls_logits = model(tensor)
                    seg_logits = None
                else:
                    seg_logits, cls_logits = model(tensor)

            if seg_logits is not None:
                r["dice"] = compute_dice(seg_logits, gt_mask)
                r["mask_overlay_b64"] = mask_to_base64(seg_logits)
            else:
                r["dice"] = None

            if cls_logits is not None:
                prob = float(torch.sigmoid(cls_logits).item())
                r["tumor_prob"] = round(prob, 4)
                r["confidence"] = round(max(prob, 1 - prob) * 100, 2)
            else:
                r["tumor_prob"] = None
                r["confidence"] = None

        except Exception as e:
            r["error"] = str(e)
            r["dice"] = None
            r["tumor_prob"] = None
            r["confidence"] = None

        results.append(r)

    return results
