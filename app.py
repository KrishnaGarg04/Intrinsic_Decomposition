"""
Material Editor — Streamlit App
Physics-based metal/plastic editing using albedo/shading/specularity decomposition
+ Auto segmentation (GrabCut + saliency) + LoRA diffusion refinement
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import torch
import random

st.set_page_config(page_title="Material Editor", layout="wide", page_icon="🔩")

st.markdown("""
<style>
[data-testid="stSidebar"] { min-width: 320px; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# AUTO SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_saliency(img_uint8):
    """
    Pure-OpenCV saliency via frequency-tuned method (no contrib needed).
    Computes the difference between each pixel colour and the mean image colour
    in Lab space, blurred at multiple scales — a lightweight but effective proxy.
    """
    h, w = img_uint8.shape[:2]
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2Lab).astype(np.float32)

    # Global mean colour
    mean_lab = lab.mean(axis=(0, 1), keepdims=True)

    # Saliency = distance from global mean, smoothed
    diff = np.linalg.norm(lab - mean_lab, axis=2)
    sal = cv2.GaussianBlur(diff, (0, 0), max(h, w) * 0.02 + 1)

    # Boost centre (objects are more often centred in product photos)
    cy, cx = h / 2, w / 2
    Y, X = np.ogrid[:h, :w]
    centre_weight = np.exp(-((X - cx)**2 / (0.35*w)**2 + (Y - cy)**2 / (0.35*h)**2))
    sal = sal * (0.6 + 0.4 * centre_weight)

    # Normalise to [0, 1]
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)
    return sal.astype(np.float32)


def auto_segment(img_arr):
    """
    Fully automatic foreground segmentation using GrabCut seeded by
    a saliency-based bounding box — no clicks required.
    Returns a float32 mask (H, W) with 1 = foreground, 0 = background.
    """
    img_uint8 = (np.clip(img_arr, 0, 1) * 255).astype(np.uint8)
    h, w = img_uint8.shape[:2]

    # ── Step 1: saliency map (no contrib required) ────────────────────────
    sal_map = compute_saliency(img_uint8)

    sal_map = sal_map.astype(np.float32)
    sal_map = cv2.GaussianBlur(sal_map, (0, 0), min(h, w) * 0.02)

    # ── Step 2: bounding box from top-K% saliency pixels ─────────────────
    thresh = np.percentile(sal_map, 70)
    binary = (sal_map > thresh).astype(np.uint8)

    # find tightest bounding box with a small margin
    ys, xs = np.where(binary)
    if len(ys) == 0:
        # fallback: centre 60% of image
        margin_y, margin_x = int(h * 0.2), int(w * 0.2)
        rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
    else:
        pad = 10
        x0 = max(xs.min() - pad, 0)
        y0 = max(ys.min() - pad, 0)
        x1 = min(xs.max() + pad, w - 1)
        y1 = min(ys.max() + pad, h - 1)
        rect = (x0, y0, x1 - x0, y1 - y0)

    # ── Step 3: GrabCut inside that bounding box ──────────────────────────
    gc_mask  = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_uint8, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # pixels marked as probable/definite foreground
    fg_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.float32)

    # ── Step 4: clean up with morphology ─────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # ── Step 5: keep only the largest connected component ─────────────────
    fg_u8 = (fg_mask * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fg_mask = (labels == largest).astype(np.float32)

    return fg_mask


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS-BASED DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════

FREQ_AMPLIFY = 1.5

def decompose_image(img_arr):
    img = img_arr.astype(np.float32)
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    shading = cv2.GaussianBlur(gray, (0, 0), 5)[..., None]
    shading = np.repeat(shading, 3, axis=2)
    shading = np.clip(shading, 0.05, 1.0)
    albedo = np.clip(img / (shading + 1e-6), 0, 1)
    specularity = np.clip(img - shading * albedo, 0, 1)
    return albedo, shading, specularity

def compute_freq_bands(arr):
    # Ensure we are working with at least (H, W, 1)
    if arr.ndim == 2:
        arr = arr[..., None]
    
    # Force the blur results to keep the channel dimension
    low_raw = cv2.GaussianBlur(arr, (0, 0), 8)
    if low_raw.ndim == 2: low_raw = low_raw[..., None]
        
    mid_blur = cv2.GaussianBlur(arr, (0, 0), 2)
    if mid_blur.ndim == 2: mid_blur = mid_blur[..., None]

    low  = low_raw
    mid_raw  = mid_blur - low
    high_raw = arr - mid_blur
    
    mid  = mid_raw  / FREQ_AMPLIFY
    high = high_raw / FREQ_AMPLIFY
    return mid, high

def soft_tonemap(x):
    return np.where(x > 1.0, x / (1.0 + x), x)


# ═══════════════════════════════════════════════════════════════════════════
# METALLIC PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

METAL_TINTS = {
    "silver":   ([1.00, 1.00, 1.00], [1.00, 1.00, 1.00]),
    "gold":     ([1.00, 0.90, 0.60], [1.00, 0.85, 0.40]),
    "copper":   ([1.00, 0.75, 0.55], [1.00, 0.65, 0.45]),
    "steel":    ([0.85, 0.95, 1.00], [0.80, 0.90, 1.00]),
    "titanium": ([0.90, 1.00, 0.85], [0.85, 0.95, 0.80]),
}

def desaturate(img, amount):
    lum = (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2])[..., None]
    return img * (1 - amount) + lum * amount

def apply_metallic_albedo(A, A_high, A_mid, desat, darken, scratch_str, tint_color, tint_str):
    out = desaturate(A, desat)
    out = out * darken
    if A_high is not None:
        out = out + scratch_str * A_high
    else:
        noise = (np.random.rand(*out.shape).astype(np.float32) - 0.5) * scratch_str * 0.3
        out = out + noise
    tint = np.array(tint_color, dtype=np.float32).reshape(1, 1, 3)
    out = out * (1 - tint_str) + out * tint * tint_str
    return np.clip(out, 0, 1)

def apply_metallic_shading(Sd, Sd_mid, Sd_high, mid_alpha, high_alpha, amp_mid):
    out = Sd.copy()
    if Sd_mid is not None:
        out = out - mid_alpha * amp_mid * np.abs(Sd_mid)
    if Sd_high is not None:
        out = out + high_alpha * 0.4 * Sd_high
    return np.clip(out, 0, 1)

def apply_metallic_specularity(Sp, Sp_high, Sp_mid, sp_high_alpha, sp_mid_alpha, amp):
    out = Sp.copy()
    if Sp_high is not None:
        out = out + sp_high_alpha * amp * Sp_high
    if Sp_mid is not None:
        out = out + sp_mid_alpha * (amp * 0.6) * Sp_mid
    return np.clip(soft_tonemap(out), 0, 1.4)

def apply_metal(img_arr, mask, tint_key, strength, params):
    A, Sd, Sp = decompose_image(img_arr)
    A_mid,  A_high  = compute_freq_bands(A[..., 0])
    Sd_mid, Sd_high = compute_freq_bands(Sd[..., 0])
    Sp_mid, Sp_high = compute_freq_bands(Sp[..., 0])

    def expand(b):
        if b is not None and b.ndim == 2:
            return np.stack([b]*3, axis=-1)
        return b

    A_high  = expand(A_high)
    Sd_mid  = expand(Sd_mid)
    Sd_high = expand(Sd_high)
    Sp_high = expand(Sp_high)
    Sp_mid  = expand(Sp_mid)

    tint_color = METAL_TINTS[tint_key][0]

    desat      = params["desat"]      * strength
    darken     = 1.0 - (1.0 - params["darken"]) * strength
    scratch    = params["scratch"]    * strength
    sd_mid     = params["sd_mid"]     * strength
    sd_high    = params["sd_high"]    * strength
    amp_mid    = params["amp_mid"]
    sp_high    = params["sp_high"]    * strength
    sp_mid_v   = params["sp_mid"]     * strength
    amp_sp     = params["amp_sp"]
    tint_str   = params["tint_str"]   * strength

    A_edit  = apply_metallic_albedo(A, A_high, A_mid, desat, darken, scratch, tint_color, tint_str)
    Sd_edit = apply_metallic_shading(Sd, Sd_mid, Sd_high, sd_mid, sd_high, amp_mid)
    Sp_edit = apply_metallic_specularity(Sp, Sp_high, Sp_mid, sp_high, sp_mid_v, amp_sp)

    edited = soft_tonemap(np.clip(A_edit * Sd_edit + Sp_edit, 0, 2.0))

    soft_mask = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)[..., None]
    result = soft_mask * edited + (1 - soft_mask) * img_arr
    return np.clip(result, 0, 1)


# ═══════════════════════════════════════════════════════════════════════════
# PLASTIC PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

PLASTIC_TINTS = {
    "white":  [1.00, 1.00, 1.00],
    "black":  [0.10, 0.10, 0.10],
    "red":    [0.85, 0.15, 0.15],
    "blue":   [0.15, 0.35, 0.85],
    "green":  [0.10, 0.75, 0.20],
    "yellow": [1.00, 0.85, 0.10],
    "orange": [0.85, 0.45, 0.10],
    "purple": [0.55, 0.15, 0.85],
    "cyan":   [0.10, 0.75, 0.75],
    "pearl":  [0.95, 0.95, 0.95],
}

def saturate(img, amount):
    lum = (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2])[..., None]
    return np.clip(lum + (img - lum) * amount, 0, 1)

def apply_plastic_albedo(A, sat_amount, brighten, tint_color, tint_str):
    out = saturate(A, sat_amount)
    out = np.clip(out * brighten, 0, 1)
    tint = np.array(tint_color, dtype=np.float32).reshape(1, 1, 3)
    out = np.clip(out * (1 - tint_str) + tint * tint_str, 0, 1)
    return out

def apply_plastic_shading(Sd, Sd_mid, Sd_high, mid_alpha, high_alpha, amp_mid):
    out = Sd.copy()
    if Sd_mid is not None:
        out = out - mid_alpha * amp_mid * np.abs(Sd_mid) * 0.3
    if Sd_high is not None:
        out = out + high_alpha * 0.08 * Sd_high
    return np.clip(out, 0, 1)

def apply_plastic_specularity(Sp, Sp_high, Sp_mid, sp_high_alpha, sp_mid_alpha, amp, hw, mask):
    h, w = hw
    out = Sp * 0.25
    if Sp_mid is not None:
        out = out + sp_mid_alpha * amp * np.clip(Sp_mid, 0, None) * 1.5
    if Sp_high is not None:
        out = out + sp_high_alpha * amp * np.clip(Sp_high, 0, None) * 0.3
    cx = random.uniform(0.25, 0.75)
    cy = random.uniform(0.25, 0.75)
    sx = random.uniform(0.15, 0.40)
    sy = random.uniform(0.15, 0.40)
    ys = np.linspace(0, 1, h)[:, None]
    xs = np.linspace(0, 1, w)[None, :]
    blob = np.exp(-((xs-cx)**2/(2*sx**2) + (ys-cy)**2/(2*sy**2)))
    highlight = np.stack([blob, blob, blob], axis=-1)
    out = out + highlight * random.uniform(0.15, 0.40) * mask[..., None]
    return np.clip(soft_tonemap(out), 0, 1.4)

def apply_plastic(img_arr, mask, tint_key, strength, params):
    A, Sd, Sp = decompose_image(img_arr)
    Sd_mid, Sd_high = compute_freq_bands(Sd[..., 0])
    Sp_mid, Sp_high = compute_freq_bands(Sp[..., 0])

    def expand(b):
        if b is not None and b.ndim == 2:
            return np.stack([b]*3, axis=-1)
        return b

    Sd_mid  = expand(Sd_mid)
    Sd_high = expand(Sd_high)
    Sp_high = expand(Sp_high)
    Sp_mid  = expand(Sp_mid)

    tint_color = PLASTIC_TINTS[tint_key]
    tint_str   = params["tint_str"]  * strength
    sat        = 1.0 + (params["sat"] - 1.0) * strength
    brighten   = 1.0 + (params["brighten"] - 1.0) * strength
    sd_mid     = params["sd_mid"]    * strength
    sd_high    = params["sd_high"]   * strength
    amp_mid    = params["amp_mid"]
    sp_high    = params["sp_high"]   * strength
    sp_mid_v   = params["sp_mid"]    * strength
    amp_sp     = params["amp_sp"]

    A_edit  = apply_plastic_albedo(A, sat, brighten, tint_color, tint_str)
    Sd_edit = apply_plastic_shading(Sd, Sd_mid, Sd_high, sd_mid, sd_high, amp_mid)
    Sp_edit = apply_plastic_specularity(Sp, Sp_high, Sp_mid, sp_high, sp_mid_v, amp_sp,
                                        (img_arr.shape[0], img_arr.shape[1]), mask)

    edited = soft_tonemap(np.clip(A_edit * Sd_edit + Sp_edit, 0, 2.0))
    soft_mask = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)[..., None]
    result = soft_mask * edited + (1 - soft_mask) * img_arr
    return np.clip(result, 0, 1)


# ═══════════════════════════════════════════════════════════════════════════
# LORA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_lora():
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            safety_checker=None,
        )
        pipe.load_lora_weights("/Volumes/T7 Shield/material_lora")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        pipe = pipe.to(device)
        return pipe, True
    except Exception:
        return None, False

METAL_PROMPTS = {
    "silver":   "mirror chrome silver metallic surface, perfect reflections, highly polished metal",
    "gold":     "polished gold metal surface, warm golden reflections, luxury material",
    "copper":   "aged copper metal with patina, oxidized surface, antique metallic",
    "steel":    "brushed cold steel surface, blue reflections, industrial metal",
    "titanium": "titanium metal surface, matte reflective, aerospace material",
}

PLASTIC_PROMPTS = {
    "white":  "smooth white plastic surface, clean soft highlights, product design",
    "black":  "glossy black plastic, soft specular highlight, consumer electronics",
    "red":    "vivid red plastic material, smooth surface, soft reflections",
    "blue":   "blue plastic material, smooth matte surface, soft highlights",
    "green":  "green plastic surface, smooth clean material",
    "yellow": "yellow plastic material, bright smooth surface",
    "orange": "orange plastic surface, smooth vivid material",
    "purple": "purple plastic material, smooth soft highlights",
    "cyan":   "cyan plastic surface, bright smooth material",
    "pearl":  "pearl white plastic, subtle iridescent highlights, premium material",
}

def to_pil(arr):
    return Image.fromarray((np.clip(arr, 0, 1)*255).astype(np.uint8))

def img_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════
# DEFAULT PHYSICS PARAMS
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_METAL = {
    "desat": 0.65, "darken": 0.85, "scratch": 0.28,
    "sd_mid": 0.32, "sd_high": 0.25, "amp_mid": 0.70,
    "sp_high": 1.50, "sp_mid": 0.45, "amp_sp": 1.10,
    "tint_str": 0.22,
}

DEFAULT_PLASTIC = {
    "tint_str": 0.38, "sat": 1.30, "brighten": 1.05,
    "sd_mid": 0.10, "sd_high": 0.05, "amp_mid": 0.35,
    "sp_high": 0.20, "sp_mid": 0.60, "amp_sp": 0.90,
}


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("### Metal type")
    metal_type = st.selectbox("", list(METAL_TINTS.keys()), label_visibility="collapsed")

    st.markdown("### Plastic color")
    plastic_type = st.selectbox("", list(PLASTIC_TINTS.keys()), label_visibility="collapsed", key="ptype")

    st.markdown("---")
    st.markdown("### 🤖 LoRA Diffusion")
    use_lora = st.toggle("Enable LoRA refinement", value=False)

    lora_contribution = st.slider(
        "LoRA contribution", 0, 100, 35, 1,
        format="%d%%",
        help="0% = pure physics CV result · 100% = fully LoRA-guided output",
        disabled=not use_lora,
    )
    lora_blend    = lora_contribution / 100.0
    diff_strength = 0.30 + lora_blend * 0.55
    guidance      = 9.0

    prompt = st.text_area(
        "Prompt (guides LoRA)",
        value="",
        height=100,
        placeholder=(
            "Describe the desired material appearance…\n"
            "e.g. brushed gold surface, warm studio lighting, photorealistic"
        ),
        disabled=not use_lora,
    )

    if use_lora and not prompt.strip():
        if st.session_state.get("slider_val", 0) > 0:
            prompt = METAL_PROMPTS.get(metal_type, "shiny metallic surface")
        else:
            prompt = PLASTIC_PROMPTS.get(plastic_type, "smooth plastic surface")
        st.caption(f"\u2139\ufe0f Auto-prompt: *{prompt[:55]}\u2026*")

    neg_prompt = "matte, flat, dull, blurry, deformed"

    st.markdown("---")
    st.markdown("### Physics params")
    with st.expander("Advanced metal params"):
        DEFAULT_METAL["desat"]    = st.slider("Desaturation",   0.0, 1.0, DEFAULT_METAL["desat"],   0.05)
        DEFAULT_METAL["scratch"]  = st.slider("Scratch detail", 0.0, 0.6, DEFAULT_METAL["scratch"], 0.05)
        DEFAULT_METAL["sp_high"]  = st.slider("Specularity",    0.5, 2.5, DEFAULT_METAL["sp_high"], 0.1)
        DEFAULT_METAL["tint_str"] = st.slider("Tint strength",  0.0, 0.5, DEFAULT_METAL["tint_str"],0.05)

    with st.expander("Advanced plastic params"):
        DEFAULT_PLASTIC["sat"]      = st.slider("Saturation boost", 1.0, 2.0, DEFAULT_PLASTIC["sat"],      0.05)
        DEFAULT_PLASTIC["tint_str"] = st.slider("Color tint",       0.0, 0.7, DEFAULT_PLASTIC["tint_str"], 0.05)
        DEFAULT_PLASTIC["sp_mid"]   = st.slider("Highlight width",  0.1, 1.0, DEFAULT_PLASTIC["sp_mid"],   0.05)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════════════════

st.title("🔩 Material Editor")
st.caption("Physics-based albedo/shading/specularity decomposition · Auto segmentation (GrabCut) · LoRA refinement")

uploaded = st.file_uploader("Drop image here", type=["jpg","jpeg","png"], label_visibility="collapsed")

if uploaded:
    original_pil = Image.open(uploaded).convert("RGB")
    MAX = 640
    w, h = original_pil.size
    if w > MAX or h > MAX:
        scale = min(MAX/w, MAX/h)
        original_pil = original_pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    orig_arr = np.array(original_pil).astype(np.float32) / 255.0

    # ── Auto-segment on new image upload ──────────────────────────────────
    if st.session_state.get("img_key") != uploaded.name:
        st.session_state.img_key = uploaded.name
        if "result" in st.session_state:
            del st.session_state["result"]

        with st.spinner("🔍 Auto-segmenting foreground object..."):
            mask = auto_segment(orig_arr)
            st.session_state.mask = mask
        st.success(f"✅ Segmentation complete — {int(mask.sum())} foreground pixels detected")

    # ── Segmentation display + manual override ────────────────────────────
    st.markdown("---")
    st.markdown("### Step 1 — Segmentation")

    col_img, col_ctrl = st.columns([2, 1])

    with col_img:
        viz = orig_arr.copy()
        mask_f = st.session_state.mask
        viz[mask_f < 0.5] = viz[mask_f < 0.5] * 0.35          # dim background
        viz[mask_f >= 0.5, 1] = np.clip(                       # green tint on FG
            viz[mask_f >= 0.5, 1] + 0.12, 0, 1
        )
        coverage = 100 * mask_f.mean()
        st.image(to_pil(viz),
                 caption=f"Green = segmented object  |  Coverage: {coverage:.1f}%",
                 use_container_width=True)

    with col_ctrl:
        st.markdown("**Auto-segmentation is active.**")
        st.caption("GrabCut + saliency runs automatically. Use the controls below to refine if needed.")

        st.markdown("---")

        if st.button("🔄 Re-run auto-segment", use_container_width=True):
            with st.spinner("Re-segmenting..."):
                mask = auto_segment(orig_arr)
                st.session_state.mask = mask
            st.rerun()

        if st.button("📐 Use full image (no mask)", use_container_width=True):
            st.session_state.mask = np.ones(
                (original_pil.height, original_pil.width), dtype=np.float32
            )
            st.rerun()

        st.markdown("---")
        st.markdown("**Manual mask (optional)**")
        st.caption("Draw a mask and upload to override auto-segmentation.")
        mask_upload = st.file_uploader("Upload mask PNG (white=object)", type=["png"],
                                       label_visibility="collapsed", key="mask_upload")
        if mask_upload is not None:
            # Use file id to avoid reprocessing on every rerun (prevents infinite loop)
            file_id = mask_upload.file_id
            if st.session_state.get("mask_file_id") != file_id:
                manual_mask_pil = Image.open(mask_upload).convert("L").resize(
                    (original_pil.width, original_pil.height), Image.NEAREST
                )
                st.session_state.mask = (np.array(manual_mask_pil) > 127).astype(np.float32)
                st.session_state.mask_file_id = file_id
                st.success("✅ Manual mask applied.")
            else:
                st.success("✅ Manual mask active.")

    # ── Material slider ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Step 2 — Material")

    lcol, scol, rcol = st.columns([1, 8, 1])
    with lcol:
        st.markdown("<div style='padding-top:30px;font-size:12px;color:#888;'>← Plastic</div>",
                    unsafe_allow_html=True)
    with scol:
        slider_val = st.slider("", -100, 100, 0, 1,
                               label_visibility="collapsed", key="slider_val")
    with rcol:
        st.markdown("<div style='padding-top:30px;font-size:12px;color:#888;'>Metallic →</div>",
                    unsafe_allow_html=True)

    pct = abs(slider_val)
    if slider_val < 0:
        st.caption(f"**Plastic — {plastic_type}** ({pct}%)")
    elif slider_val > 0:
        st.caption(f"**Metallic — {metal_type}** ({pct}%)")
    else:
        st.caption("**Original** — no effect")

    if st.button("✨ Apply material", type="primary", use_container_width=True):
        strength = abs(slider_val) / 100.0
        mask = st.session_state.mask

        with st.spinner("Applying physics-based material..."):
            if slider_val == 0:
                result_arr = orig_arr.copy()

            elif slider_val > 0:
                result_arr = apply_metal(orig_arr, mask, metal_type, strength, DEFAULT_METAL.copy())

                if use_lora and strength > 0.15:
                    pipe, loaded = load_lora()
                    if loaded and pipe:
                        try:
                            with st.spinner("Running LoRA diffusion (~30s)..."):
                                lora_in = to_pil(result_arr)
                                lora_out = pipe(
                                    prompt=prompt,
                                    negative_prompt=neg_prompt,
                                    image=lora_in,
                                    strength=diff_strength,
                                    num_inference_steps=30,
                                    guidance_scale=guidance,
                                ).images[0].resize(original_pil.size)
                            lora_arr = np.array(lora_out).astype(np.float32) / 255.0
                            soft_mask = cv2.GaussianBlur(mask, (21, 21), 0)[..., None]
                            blend = lora_blend * strength
                            result_arr = result_arr*(1-blend) + lora_arr*blend
                            result_arr = np.clip(
                                orig_arr*(1-soft_mask) + result_arr*soft_mask, 0, 1
                            )
                            st.success("LoRA refinement applied")
                        except Exception as e:
                            st.warning(f"LoRA failed, using CV only: {e}")
                    else:
                        st.warning("LoRA weights not found — using physics CV only")

            else:
                result_arr = apply_plastic(orig_arr, mask, plastic_type, strength, DEFAULT_PLASTIC.copy())

        st.session_state.result = to_pil(result_arr)

    # ── Output ─────────────────────────────────────────────────────────────
    if "result" in st.session_state:
        st.markdown("---")
        st.markdown("### Result")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(original_pil, caption="Original", use_container_width=True)
        with c2:
            mask_viz = orig_arr.copy()
            mask_viz[st.session_state.mask < 0.5] *= 0.3
            st.image(to_pil(mask_viz), caption="Object mask", use_container_width=True)
        with c3:
            st.image(st.session_state.result, caption="Edited", use_container_width=True)

        mat_label = metal_type if slider_val >= 0 else plastic_type
        st.download_button(
            "⬇️ Download PNG",
            data=img_bytes(st.session_state.result),
            file_name=f"material_{mat_label}_{pct}pct.png",
            mime="image/png",
            use_container_width=True,
        )

else:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Upload**")
        st.caption("Any object photo")
    with c2:
        st.markdown("**2. Auto-Segment**")
        st.caption("GrabCut isolates the object automatically — no clicks needed")
    with c3:
        st.markdown("**3. Slide**")
        st.caption("← Plastic &nbsp; | &nbsp; Metallic →", unsafe_allow_html=True)
    st.info("Upload an image to begin — segmentation runs automatically")