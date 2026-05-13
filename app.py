# app.py — Document Restoration Studio

import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
from main import (
    restore_document, get_image_histogram,
    ocr_analyze, compute_ocr_diff,
    generate_confidence_heatmap, check_tesseract
)

st.set_page_config(
    page_title="Document Restoration Studio",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size:2rem; font-weight:700; margin-bottom:0; }
    .section-header {
        font-size:.85rem; font-weight:600; color:#888;
        text-transform:uppercase; letter-spacing:1px;
        margin-top:1.2rem; margin-bottom:.3rem;
    }
    hr.divider { border:none; border-top:1px solid #333; margin:.6rem 0; }
    .metric-card {
        background:#1e1e2e; border:1px solid #333; border-radius:10px;
        padding:14px 18px; text-align:center;
    }
    .metric-val { font-size:2rem; font-weight:700; }
    .metric-lbl { font-size:.75rem; color:#888; margin-top:2px; }
    .pill-green {
        display:inline-block; background:#14532d; color:#86efac;
        border-radius:20px; padding:2px 10px; font-size:.78rem; margin:2px;
    }
    .pill-red {
        display:inline-block; background:#7f1d1d; color:#fca5a5;
        border-radius:20px; padding:2px 10px; font-size:.78rem; margin:2px;
        text-decoration:line-through;
    }
    .pill-blue {
        display:inline-block; background:#1e3a5f; color:#93c5fd;
        border-radius:20px; padding:2px 10px; font-size:.78rem; margin:2px;
    }
    .step-row {
        display:flex; align-items:flex-start; gap:12px;
        padding:10px 14px; border-radius:8px;
        background:#1e1e2e; border:1px solid #2d2d3d; margin-bottom:8px;
    }
    .step-num {
        background:#3b82f6; color:white; border-radius:50%;
        width:24px; height:24px; display:flex; align-items:center;
        justify-content:center; font-size:.75rem; font-weight:700;
        flex-shrink:0; margin-top:2px;
    }
    .step-name { font-weight:600; font-size:.9rem; }
    .step-detail { color:#888; font-size:.8rem; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📄 Document Restoration Studio</p>', unsafe_allow_html=True)
st.caption("Restore documents and measure improvement with OCR-based readability scoring.")
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─── File Upload ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a document image (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"]
)
if uploaded_file is None:
    st.session_state.pop("temp_path", None)
    st.session_state.pop("upload_id", None)
    st.info("👆 Upload an image to get started.")
    st.stop()

# Persist temp file across Streamlit reruns
file_id = getattr(uploaded_file, "file_id", uploaded_file.name)
if st.session_state.get("upload_id") != file_id:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        st.session_state["temp_path"] = tmp.name
    st.session_state["upload_id"] = file_id
    st.session_state["ocr_done"]   = False

temp_path    = st.session_state["temp_path"]
base_path    = os.path.splitext(temp_path)[0]
output_path  = f"{base_path}_restored.png"

original_bgr  = cv2.imread(temp_path)
original_rgb  = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Controls")
    st.markdown('<p class="section-header">📋 Preset Profiles</p>', unsafe_allow_html=True)
    preset = st.selectbox("Start from a preset",
        ["Custom", "Scanned Document", "Old / Yellowed Paper",
         "Low-Light Photo", "Faint Text", "High Contrast Print"])

    PRESETS = {
        "Scanned Document": dict(
            color_mode="grayscale",
            normalize_background=True, norm_kernel_size=15,
            denoise_method="bilateral", bilateral_d=9,
            bilateral_sigma_color=75, bilateral_sigma_space=75,
            median_ksize=5, nlmeans_h=10, gaussian_ksize=5,
            contrast_method="clahe", clahe_clip_limit=3.0, clahe_tile_size=8, gamma=1.0,
            sharpen=True, sharpen_strength=1.5, sharpen_blur_sigma=3.0,
            threshold_method="combined", adaptive_block_size=15, adaptive_C=10,
            morph_open=True, morph_open_ksize=2, morph_close=True, morph_close_ksize=3,
            morph_dilate=False, morph_dilate_ksize=2, morph_erode=False, morph_erode_ksize=2,
            deskew=False, remove_borders=False, border_size=10,
            invert_output=False, brightness=0, edge_enhance=False,
        ),
        "Old / Yellowed Paper": dict(
            color_mode="grayscale",
            normalize_background=True, norm_kernel_size=21,
            denoise_method="nlmeans", nlmeans_h=12,
            bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75,
            median_ksize=5, gaussian_ksize=5,
            contrast_method="clahe", clahe_clip_limit=4.0, clahe_tile_size=8, gamma=1.0,
            sharpen=True, sharpen_strength=1.8, sharpen_blur_sigma=2.0,
            threshold_method="sauvola", adaptive_block_size=21, adaptive_C=10,
            morph_open=True, morph_open_ksize=2, morph_close=True, morph_close_ksize=2,
            morph_dilate=False, morph_dilate_ksize=2, morph_erode=False, morph_erode_ksize=2,
            deskew=True, remove_borders=True, border_size=8,
            invert_output=False, brightness=10, edge_enhance=False,
        ),
        "Low-Light Photo": dict(
            color_mode="grayscale",
            normalize_background=False, norm_kernel_size=15,
            denoise_method="bilateral", bilateral_d=11,
            bilateral_sigma_color=100, bilateral_sigma_space=100,
            median_ksize=5, nlmeans_h=10, gaussian_ksize=5,
            contrast_method="gamma", clahe_clip_limit=3.0, clahe_tile_size=8, gamma=1.8,
            sharpen=True, sharpen_strength=1.4, sharpen_blur_sigma=2.0,
            threshold_method="adaptive", adaptive_block_size=19, adaptive_C=8,
            morph_open=True, morph_open_ksize=2, morph_close=True, morph_close_ksize=3,
            morph_dilate=False, morph_dilate_ksize=2, morph_erode=False, morph_erode_ksize=2,
            deskew=False, remove_borders=False, border_size=10,
            invert_output=False, brightness=20, edge_enhance=False,
        ),
        "Faint Text": dict(
            color_mode="grayscale",
            normalize_background=True, norm_kernel_size=11,
            denoise_method="gaussian", gaussian_ksize=3,
            bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75,
            median_ksize=5, nlmeans_h=10,
            contrast_method="hist_eq", clahe_clip_limit=3.0, clahe_tile_size=8, gamma=1.0,
            sharpen=True, sharpen_strength=2.0, sharpen_blur_sigma=1.5,
            threshold_method="sauvola", adaptive_block_size=13, adaptive_C=5,
            morph_open=False, morph_open_ksize=2, morph_close=True, morph_close_ksize=2,
            morph_dilate=True, morph_dilate_ksize=2, morph_erode=False, morph_erode_ksize=2,
            deskew=False, remove_borders=False, border_size=10,
            invert_output=False, brightness=30, edge_enhance=True,
        ),
        "High Contrast Print": dict(
            color_mode="grayscale",
            normalize_background=False, norm_kernel_size=15,
            denoise_method="median", median_ksize=3,
            bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75,
            nlmeans_h=10, gaussian_ksize=5,
            contrast_method="none", clahe_clip_limit=3.0, clahe_tile_size=8, gamma=1.0,
            sharpen=False, sharpen_strength=1.0, sharpen_blur_sigma=3.0,
            threshold_method="otsu", adaptive_block_size=15, adaptive_C=10,
            morph_open=True, morph_open_ksize=2, morph_close=False, morph_close_ksize=2,
            morph_dilate=False, morph_dilate_ksize=2, morph_erode=False, morph_erode_ksize=2,
            deskew=False, remove_borders=False, border_size=10,
            invert_output=False, brightness=0, edge_enhance=False,
        ),
    }

    use_preset = preset != "Custom"
    pv = PRESETS.get(preset, {})
    def pget(k, d): return pv.get(k, d) if use_preset else d

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🎨 Color Mode</p>', unsafe_allow_html=True)
    color_mode = st.radio("Process as", ["grayscale", "color"],
                           index=["grayscale","color"].index(pget("color_mode","grayscale")),
                           horizontal=True, disabled=use_preset)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🔧 Corrections</p>', unsafe_allow_html=True)
    deskew         = st.checkbox("Auto Deskew", value=pget("deskew", False), disabled=use_preset)
    remove_borders = st.checkbox("Remove Border Artifacts", value=pget("remove_borders", False), disabled=use_preset)
    border_size    = st.slider("Border Strip (px)", 2, 40, pget("border_size", 10), disabled=use_preset or not remove_borders)
    brightness     = st.slider("Brightness Adjustment", -100, 100, pget("brightness", 0), disabled=use_preset)
    invert_output  = st.checkbox("Invert Output", value=pget("invert_output", False), disabled=use_preset)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🌅 Background Normalization</p>', unsafe_allow_html=True)
    normalize_background = st.checkbox("Normalize Background", value=pget("normalize_background", True), disabled=use_preset)
    norm_kernel_size     = st.slider("Kernel Size", 5, 51, pget("norm_kernel_size", 15), step=2,
                                      disabled=use_preset or not normalize_background)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🔇 Noise Reduction</p>', unsafe_allow_html=True)
    denoise_method = st.selectbox("Method",
        ["bilateral","median","nlmeans","gaussian","none"],
        index=["bilateral","median","nlmeans","gaussian","none"].index(pget("denoise_method","bilateral")),
        disabled=use_preset)
    bilateral_d           = st.slider("Filter Diameter", 3, 15, pget("bilateral_d", 9), step=2,
                                       disabled=use_preset or denoise_method != "bilateral")
    bilateral_sigma_color = st.slider("Sigma Color", 10, 200, pget("bilateral_sigma_color", 75),
                                       disabled=use_preset or denoise_method != "bilateral")
    bilateral_sigma_space = st.slider("Sigma Space", 10, 200, pget("bilateral_sigma_space", 75),
                                       disabled=use_preset or denoise_method != "bilateral")
    median_ksize          = st.slider("Median Kernel", 3, 11, pget("median_ksize", 5), step=2,
                                       disabled=use_preset or denoise_method != "median")
    nlmeans_h             = st.slider("NL-Means Strength", 1, 30, pget("nlmeans_h", 10),
                                       disabled=use_preset or denoise_method != "nlmeans")
    gaussian_ksize        = st.slider("Gaussian Kernel", 3, 15, pget("gaussian_ksize", 5), step=2,
                                       disabled=use_preset or denoise_method != "gaussian")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🌓 Contrast Enhancement</p>', unsafe_allow_html=True)
    contrast_method  = st.selectbox("Method", ["clahe","hist_eq","gamma","none"],
        index=["clahe","hist_eq","gamma","none"].index(pget("contrast_method","clahe")),
        disabled=use_preset)
    clahe_clip_limit = st.slider("CLAHE Clip Limit", 0.5, 8.0, pget("clahe_clip_limit", 3.0),
                                  step=0.5, disabled=use_preset or contrast_method != "clahe")
    clahe_tile_size  = st.slider("CLAHE Tile Size", 4, 16, pget("clahe_tile_size", 8),
                                  step=2, disabled=use_preset or contrast_method != "clahe")
    gamma            = st.slider("Gamma", 0.2, 3.0, pget("gamma", 1.0), step=0.1,
                                  disabled=use_preset or contrast_method != "gamma")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🔬 Sharpening</p>', unsafe_allow_html=True)
    sharpen            = st.checkbox("Enable Sharpening", value=pget("sharpen", True), disabled=use_preset)
    sharpen_strength   = st.slider("Strength", 1.0, 3.0, pget("sharpen_strength", 1.5),
                                    step=0.1, disabled=use_preset or not sharpen)
    sharpen_blur_sigma = st.slider("Radius (sigma)", 0.5, 5.0, pget("sharpen_blur_sigma", 3.0),
                                    step=0.5, disabled=use_preset or not sharpen)
    edge_enhance       = st.checkbox("Edge Enhancement", value=pget("edge_enhance", False), disabled=use_preset)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🎯 Thresholding</p>', unsafe_allow_html=True)
    threshold_method    = st.selectbox("Method", ["adaptive","otsu","combined","sauvola","none"],
        index=["adaptive","otsu","combined","sauvola","none"].index(pget("threshold_method","adaptive")),
        disabled=use_preset)
    adaptive_block_size = st.slider("Block Size (odd)", 5, 51, pget("adaptive_block_size", 15),
                                     step=2, disabled=use_preset or threshold_method in ("otsu","none"))
    adaptive_C          = st.slider("Constant C", 0, 30, pget("adaptive_C", 10),
                                     disabled=use_preset or threshold_method in ("otsu","none"))

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🔩 Morphology</p>', unsafe_allow_html=True)
    morph_open         = st.checkbox("Opening",  value=pget("morph_open", True),    disabled=use_preset)
    morph_open_ksize   = st.slider("Open Kernel",   1, 7, pget("morph_open_ksize", 2),   disabled=use_preset or not morph_open)
    morph_close        = st.checkbox("Closing",  value=pget("morph_close", True),   disabled=use_preset)
    morph_close_ksize  = st.slider("Close Kernel",  1, 7, pget("morph_close_ksize", 3),  disabled=use_preset or not morph_close)
    morph_dilate       = st.checkbox("Dilation", value=pget("morph_dilate", False), disabled=use_preset)
    morph_dilate_ksize = st.slider("Dilate Kernel", 1, 5, pget("morph_dilate_ksize", 2), disabled=use_preset or not morph_dilate)
    morph_erode        = st.checkbox("Erosion",  value=pget("morph_erode", False),  disabled=use_preset)
    morph_erode_ksize  = st.slider("Erode Kernel",  1, 5, pget("morph_erode_ksize", 2),  disabled=use_preset or not morph_erode)

# ─── Build params dict ───────────────────────────────────────────────────────
def v(key, default):
    return pv.get(key, default) if use_preset else default

params = dict(
    color_mode=v("color_mode", color_mode),
    deskew=v("deskew", deskew), remove_borders=v("remove_borders", remove_borders),
    border_size=v("border_size", border_size), brightness=v("brightness", brightness),
    invert_output=v("invert_output", invert_output),
    normalize_background=v("normalize_background", normalize_background),
    norm_kernel_size=v("norm_kernel_size", norm_kernel_size),
    denoise_method=v("denoise_method", denoise_method),
    bilateral_d=v("bilateral_d", bilateral_d),
    bilateral_sigma_color=v("bilateral_sigma_color", bilateral_sigma_color),
    bilateral_sigma_space=v("bilateral_sigma_space", bilateral_sigma_space),
    median_ksize=v("median_ksize", median_ksize), nlmeans_h=v("nlmeans_h", nlmeans_h),
    gaussian_ksize=v("gaussian_ksize", gaussian_ksize),
    contrast_method=v("contrast_method", contrast_method),
    clahe_clip_limit=v("clahe_clip_limit", clahe_clip_limit),
    clahe_tile_size=v("clahe_tile_size", clahe_tile_size), gamma=v("gamma", gamma),
    sharpen=v("sharpen", sharpen), sharpen_strength=v("sharpen_strength", sharpen_strength),
    sharpen_blur_sigma=v("sharpen_blur_sigma", sharpen_blur_sigma),
    edge_enhance=v("edge_enhance", edge_enhance),
    threshold_method=v("threshold_method", threshold_method),
    adaptive_block_size=v("adaptive_block_size", adaptive_block_size),
    adaptive_C=v("adaptive_C", adaptive_C),
    morph_open=v("morph_open", morph_open), morph_open_ksize=v("morph_open_ksize", morph_open_ksize),
    morph_close=v("morph_close", morph_close), morph_close_ksize=v("morph_close_ksize", morph_close_ksize),
    morph_dilate=v("morph_dilate", morph_dilate), morph_dilate_ksize=v("morph_dilate_ksize", morph_dilate_ksize),
    morph_erode=v("morph_erode", morph_erode), morph_erode_ksize=v("morph_erode_ksize", morph_erode_ksize),
)

# ─── Helper: render matplotlib fig to Streamlit ──────────────────────────────
def _fig_to_st(fig):
    """Safely convert a matplotlib figure to bytes and show in Streamlit."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)                    # ← KEY FIX: reset pointer before reading
    st.image(buf.getvalue(), use_container_width=True)
    plt.close(fig)
    buf.close()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_restore, tab_compare, tab_ocr, tab_hist, tab_info = st.tabs([
    "🖼️ Restore", "🔀 Compare", "📝 OCR Diff & Readability", "📊 Histogram", "ℹ️ Pipeline"
])

# ── Tab 1: Restore ────────────────────────────────────────────────────────────
with tab_restore:
    col_orig, col_out = st.columns(2)
    with col_orig:
        st.subheader("Original")
        h, w = original_rgb.shape[:2]
        st.image(original_rgb, use_container_width=True)
        st.caption(f"{w}×{h} px  ·  {uploaded_file.name}  ·  {round(uploaded_file.size/1024,1)} KB")
    with col_out:
        st.subheader("Restored")
        result_slot = st.empty()
        # Show last restored result if it exists
        if os.path.exists(output_path):
            prev = cv2.imread(output_path)
            if prev is not None:
                result_slot.image(
                    cv2.cvtColor(prev, cv2.COLOR_BGR2RGB) if len(prev.shape)==3 else prev,
                    use_container_width=True
                )

    run = st.button("▶ Run Restoration", type="primary", use_container_width=True)
    if run:
        with st.spinner("Processing…"):
            result = restore_document(temp_path, show_steps=False, params=params)
        if result is not None:
            disp = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) if len(result.shape)==3 else result
            result_slot.image(disp, use_container_width=True, clamp=True)
            st.success("✅ Restored successfully!")
            c1, c2 = st.columns(2)
            with c1:
                with open(output_path, "rb") as f:
                    st.download_button("⬇ Download PNG", f, "restored.png", "image/png", use_container_width=True)
            with c2:
                ok, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok:
                    st.download_button("⬇ Download JPEG", buf.tobytes(), "restored.jpg", "image/jpeg", use_container_width=True)
        else:
            st.error("❌ Failed to process image.")

# ── Tab 2: Compare ────────────────────────────────────────────────────────────
with tab_compare:
    st.subheader("Side-by-Side Comparison")
    if not os.path.exists(output_path):
        st.info("Run restoration first (▶ Run Restoration).")
    else:
        res_bgr = cv2.imread(output_path)
        if res_bgr is not None:
            res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB) if len(res_bgr.shape)==3 else res_bgr
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original**")
                st.image(original_rgb, use_container_width=True)
            with c2:
                st.markdown("**Restored**")
                st.image(res_rgb, use_container_width=True)
            with st.expander("🔎 Difference Heatmap"):
                orig_g = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
                res_g  = cv2.resize(
                    cv2.cvtColor(res_bgr, cv2.COLOR_BGR2GRAY) if len(res_bgr.shape)==3 else res_bgr,
                    (orig_g.shape[1], orig_g.shape[0])
                )
                diff_map = cv2.applyColorMap(cv2.absdiff(orig_g, res_g), cv2.COLORMAP_JET)
                st.image(cv2.cvtColor(diff_map, cv2.COLOR_BGR2RGB),
                         caption="Red = large pixel change", use_container_width=True)

# ── Tab 3: OCR Diff ───────────────────────────────────────────────────────────
with tab_ocr:
    st.subheader("📝 OCR-Based Readability Analysis")

    if not check_tesseract():
        st.error("⚠️ Tesseract OCR engine not found.")
        st.markdown("""
**Step 1** — [Download Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki) and install to default path.

**Step 2** — `pip install pytesseract`

**Step 3** — Make sure the path at the top of `main.py` matches your install location.
        """)
        st.stop()

    if not os.path.exists(output_path):
        st.info("Run restoration first (▶ Run Restoration) to enable OCR analysis.")
        st.stop()

    with st.expander("⚙️ OCR Tuning", expanded=False):
        ocr_conf_threshold = st.slider("Minimum word confidence to show (%)", 0, 80, 0,
            help="Filter out low-confidence words from the diff. 0 = show all.")

    run_ocr = st.button("🔍 Run OCR Analysis", type="primary", use_container_width=True)

    if run_ocr or st.session_state.get("ocr_done"):
        if run_ocr:
            with st.spinner("Running OCR on original image… (tries 20 configurations)"):
                ocr_before = ocr_analyze(original_gray)
            with st.spinner("Running OCR on restored image…"):
                res_img    = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
                ocr_after  = ocr_analyze(res_img)
            diff = compute_ocr_diff(ocr_before, ocr_after)
            st.session_state.update({
                "ocr_before": ocr_before, "ocr_after": ocr_after,
                "ocr_diff": diff, "ocr_done": True,
            })
        else:
            ocr_before = st.session_state["ocr_before"]
            ocr_after  = st.session_state["ocr_after"]
            diff       = st.session_state["ocr_diff"]

        if ocr_before.get("error") and not ocr_before["words"]:
            st.error(f"OCR error: {ocr_before['error']}")
            st.stop()

        # ── Score Cards ───────────────────────────────────────────────────────
        st.markdown("### Readability Score")
        delta_color = "#22c55e" if diff["score_delta"] >= 0 else "#ef4444"
        delta_label = f"{'▲' if diff['score_delta'] >= 0 else '▼'} {abs(diff['score_delta']):.1f} pts"
        cards = [
            (f"{diff['score_before']:.1f}", "Readability Before", "#3b82f6"),
            (f"{diff['score_after']:.1f}",  "Readability After",  "#22c55e"),
            (delta_label,                    "Score Δ",            delta_color),
            (f"{diff['word_count_after']:,}","Words Detected",     "#a78bfa"),
            (f"{diff['high_conf_after']:.0f}%","High-Conf Words",  "#f59e0b"),
        ]
        for col, (val, lbl, color) in zip(st.columns(5), cards):
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-val" style="color:{color};">{val}</div>'
                    f'<div class="metric-lbl">{lbl}</div></div>',
                    unsafe_allow_html=True
                )
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gauge Chart ───────────────────────────────────────────────────────
        st.markdown("### Confidence Gauge")
        plt.close("all")   # clear any leftover figures
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5),
                                  subplot_kw=dict(projection="polar"))
        fig.patch.set_facecolor("#0e1117")
        for ax, (score, label, color) in zip(axes, [
            (diff["score_before"], "Original", "#3b82f6"),
            (diff["score_after"],  "Restored", "#22c55e"),
        ]):
            angle      = np.pi * (1 - score / 100)
            theta      = np.linspace(0, np.pi, 200)
            ax.plot(theta, np.ones_like(theta), color="#1e293b", linewidth=18, zorder=1)
            fill_theta = np.linspace(0, np.pi * (score / 100), 200)
            ax.plot(fill_theta, np.ones_like(fill_theta), color=color,
                    linewidth=18, zorder=2, solid_capstyle="round")
            ax.plot([angle], [0.82], "o", markersize=10, color=color, zorder=3)
            ax.plot([angle, angle], [0, 0.75], color="white", linewidth=2, zorder=3)
            ax.set(ylim=(0, 1.1)); ax.set_theta_zero_location("W")
            ax.set_theta_direction(-1); ax.set_thetamin(0); ax.set_thetamax(180)
            ax.set_rticks([]); ax.set_xticks([])
            ax.spines["polar"].set_visible(False)
            ax.set_facecolor("#0e1117")
            ax.text(np.pi/2, 0.35, f"{score:.1f}", ha="center", va="center",
                     fontsize=22, fontweight="bold", color=color)
            ax.text(np.pi/2, 0.08, label, ha="center", va="center", fontsize=11, color="#aaa")
            for t, lbl in [(0,"100"),(np.pi/2,"50"),(np.pi,"0")]:
                ax.text(t, 1.12, lbl, ha="center", va="center", fontsize=8, color="#555")
        plt.tight_layout(pad=0.5)
        _fig_to_st(fig)

        # ── Confidence Distribution ───────────────────────────────────────────
        st.markdown("### Confidence Distribution")
        plt.close("all")
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        fig2.patch.set_facecolor("#0e1117"); ax2.set_facecolor("#0e1117")
        ax2.hist(ocr_before["confidences"], bins=range(0,105,5), alpha=0.6,
                  color="#3b82f6", label="Original", edgecolor="none")
        ax2.hist(ocr_after["confidences"],  bins=range(0,105,5), alpha=0.6,
                  color="#22c55e", label="Restored", edgecolor="none")
        ax2.axvline(70, color="#f59e0b", linestyle="--", linewidth=1,
                     label="High-conf threshold (70)")
        ax2.set_xlabel("Word Confidence (%)", color="#aaa")
        ax2.set_ylabel("Word Count", color="#aaa")
        ax2.tick_params(colors="#aaa")
        for sp in ax2.spines.values(): sp.set_edgecolor("#333")
        ax2.legend(facecolor="#1e1e2e", edgecolor="#333", labelcolor="white", fontsize=9)
        plt.tight_layout()
        _fig_to_st(fig2)

        # ── Confidence Heatmaps ───────────────────────────────────────────────
        st.markdown("### Per-Word Confidence Heatmap")
        st.caption("🟢 Green = high confidence · 🔴 Red = low confidence")
        hc1, hc2 = st.columns(2)
        with hc1:
            st.markdown("**Original**")
            st.image(cv2.cvtColor(generate_confidence_heatmap(original_gray, ocr_before),
                                   cv2.COLOR_BGR2RGB), use_container_width=True)
        with hc2:
            st.markdown("**Restored**")
            rg = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
            if rg is not None:
                st.image(cv2.cvtColor(generate_confidence_heatmap(rg, ocr_after),
                                       cv2.COLOR_BGR2RGB), use_container_width=True)

        # ── Word Summary Pills ────────────────────────────────────────────────
        st.markdown("### Word-Level Summary")
        conf_thr      = ocr_conf_threshold
        cm_after_map  = {w.lower(): c for w, c in zip(ocr_after["words"],  ocr_after["confidences"])}
        cm_before_map = {w.lower(): c for w, c in zip(ocr_before["words"], ocr_before["confidences"])}

        unlocked_f  = [w for w in diff["unlocked_words"]  if cm_after_map.get(w,  0) >= conf_thr]
        lost_f      = [w for w in diff["lost_words"]       if cm_before_map.get(w, 0) >= conf_thr]
        improved_f  = [w for w in diff["improved_words"]   if cm_after_map.get(w,  0) >= conf_thr]

        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            st.markdown(f"**🟢 Unlocked ({len(unlocked_f)})**")
            st.caption("Newly readable after restoration")
            if unlocked_f:
                st.markdown(" ".join(
                    f'<span class="pill-green" title="{cm_after_map.get(w,0)}% conf">'
                    f'{w} <small>{cm_after_map.get(w,0)}%</small></span>'
                    for w in unlocked_f[:80]), unsafe_allow_html=True)
            else:
                st.caption("None")

        with wc2:
            st.markdown(f"**🔴 Lost ({len(lost_f)})**")
            st.caption("Words lost after restoration")
            if lost_f:
                st.markdown(" ".join(
                    f'<span class="pill-red">{w} <small>{cm_before_map.get(w,0)}%</small></span>'
                    for w in lost_f[:80]), unsafe_allow_html=True)
            else:
                st.caption("None — no regression ✓")

        with wc3:
            st.markdown(f"**🔵 Improved ({len(improved_f)})**")
            st.caption("Same words, higher confidence")
            if improved_f:
                st.markdown(" ".join(
                    f'<span class="pill-blue" '
                    f'title="{cm_before_map.get(w,0)}% → {cm_after_map.get(w,0)}%">'
                    f'{w} <small>+{cm_after_map.get(w,0)-cm_before_map.get(w,0)}%</small></span>'
                    for w in improved_f[:80]), unsafe_allow_html=True)
            else:
                st.caption("None")

        # ── Text Diff ─────────────────────────────────────────────────────────
        st.markdown("### Full Text Diff")
        st.caption("🟢 Green = new word  ·  🔴 Red strikethrough = removed word")
        if diff["text_diff_html"].strip():
            st.markdown(
                f'<div style="background:#111827;border:1px solid #374151;border-radius:8px;'
                f'padding:16px;line-height:2;font-family:monospace;font-size:.9rem;'
                f'color:#e5e7eb;max-height:400px;overflow-y:auto;">'
                f'{diff["text_diff_html"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("No text extracted. Ensure the image contains readable text.")

        with st.expander("📄 Raw Extracted Text"):
            r1, r2 = st.columns(2)
            with r1:
                st.markdown("**Original**")
                st.text_area("", diff["full_text_before"], height=250,
                              key="raw_before", label_visibility="collapsed")
            with r2:
                st.markdown("**Restored**")
                st.text_area("", diff["full_text_after"], height=250,
                              key="raw_after", label_visibility="collapsed")

        # ── Export ────────────────────────────────────────────────────────────
        report = "\n".join([
            "DOCUMENT RESTORATION — OCR READABILITY REPORT", "="*50,
            f"Score Before     : {diff['score_before']:.1f} / 100",
            f"Score After      : {diff['score_after']:.1f} / 100",
            f"Score Delta      : {diff['score_delta']:+.1f} pts",
            f"Words Before     : {diff['word_count_before']}",
            f"Words After      : {diff['word_count_after']}",
            f"High-Conf Before : {diff['high_conf_before']}%",
            f"High-Conf After  : {diff['high_conf_after']}%",
            "",
            f"Unlocked ({len(diff['unlocked_words'])}): " + ", ".join(diff["unlocked_words"]),
            f"Lost ({len(diff['lost_words'])}): "         + ", ".join(diff["lost_words"]),
            f"Improved ({len(diff['improved_words'])}): " + ", ".join(diff["improved_words"]),
            "", "--- ORIGINAL TEXT ---", diff["full_text_before"],
            "", "--- RESTORED TEXT ---",  diff["full_text_after"],
        ])
        st.download_button("⬇ Download OCR Report (.txt)", report,
                            "ocr_report.txt", "text/plain", use_container_width=True)

# ── Tab 4: Histogram ──────────────────────────────────────────────────────────
with tab_hist:
    st.subheader("Pixel Intensity Histogram")
    plt.close("all")   # ← prevent stale figure leaks
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0e1117")

    restored_gray_hist = None
    if os.path.exists(output_path):
        _tmp = cv2.imread(output_path)
        if _tmp is not None:
            restored_gray_hist = cv2.cvtColor(_tmp, cv2.COLOR_BGR2GRAY) if len(_tmp.shape)==3 else _tmp

    for ax, (src, label, color) in zip(axes, [
        (original_gray,     "Original", "#4fc3f7"),
        (restored_gray_hist, "Restored", "#81c784"),
    ]):
        ax.set_facecolor("#0e1117")
        if src is None:
            ax.set_title("Run restoration first", color="white", fontsize=11)
            for sp in ax.spines.values(): sp.set_edgecolor("#333")
            continue
        hist = cv2.calcHist([src], [0], None, [256], [0, 256]).flatten()
        ax.fill_between(range(256), hist, alpha=0.5, color=color)
        ax.plot(hist, color=color, linewidth=1.2)
        # Mean and median lines
        mean_val   = float(np.mean(src))
        median_val = float(np.median(src))
        ax.axvline(mean_val,   color="#f59e0b", linestyle="--", linewidth=1,
                    label=f"Mean {mean_val:.0f}")
        ax.axvline(median_val, color="#f87171", linestyle=":",  linewidth=1,
                    label=f"Median {median_val:.0f}")
        ax.set_title(label, color="white", fontsize=13)
        ax.set_xlabel("Pixel Value", color="#aaa")
        ax.set_ylabel("Count", color="#aaa")
        ax.tick_params(colors="#aaa")
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
        ax.legend(facecolor="#1e1e2e", edgecolor="#333", labelcolor="white", fontsize=8)

    plt.tight_layout()
    _fig_to_st(fig)   # ← uses buf.seek(0) fix

    if restored_gray_hist is not None:
        with st.expander("📊 Statistics"):
            sc1, sc2, sc3, sc4 = st.columns(4)
            stats = [
                ("Orig Mean",   f"{np.mean(original_gray):.1f}"),
                ("Rest Mean",   f"{np.mean(restored_gray_hist):.1f}"),
                ("Orig Std",    f"{np.std(original_gray):.1f}"),
                ("Rest Std",    f"{np.std(restored_gray_hist):.1f}"),
            ]
            for col, (lbl, val) in zip([sc1,sc2,sc3,sc4], stats):
                col.metric(lbl, val)

# ── Tab 5: Pipeline Info ──────────────────────────────────────────────────────
with tab_info:
    st.subheader("Active Pipeline Configuration")
    st.caption(f"Preset: **{preset}**  ·  Color mode: **{params['color_mode']}**")
    st.markdown("")

    # Build steps list with full details
    steps = []
    if params["deskew"]:
        steps.append(("📐 Auto Deskew", "Detects and corrects document rotation angle"))
    if params["normalize_background"]:
        steps.append(("🌅 Background Normalization",
                       f"Kernel {params['norm_kernel_size']}×{params['norm_kernel_size']} px — removes shadows & uneven lighting"))
    if params["brightness"] != 0:
        steps.append(("💡 Brightness Adjustment", f"Offset: {params['brightness']:+d}"))

    denoise_details = {
        "bilateral": f"d={params['bilateral_d']}, σ_color={params['bilateral_sigma_color']}, σ_space={params['bilateral_sigma_space']}",
        "median":    f"Kernel {params['median_ksize']}×{params['median_ksize']}",
        "nlmeans":   f"Strength h={params['nlmeans_h']}",
        "gaussian":  f"Kernel {params['gaussian_ksize']}×{params['gaussian_ksize']}",
        "none":      "Disabled",
    }
    steps.append(("🔇 Noise Reduction",
                   f"Method: {params['denoise_method']}  ·  {denoise_details[params['denoise_method']]}"))

    if params["contrast_method"] != "none":
        contrast_details = {
            "clahe":    f"Clip limit {params['clahe_clip_limit']}, tile {params['clahe_tile_size']}×{params['clahe_tile_size']}",
            "hist_eq":  "Global histogram equalisation",
            "gamma":    f"γ = {params['gamma']}  ({'brighter' if params['gamma']<1 else 'darker'})",
        }
        steps.append(("🌓 Contrast Enhancement",
                       f"Method: {params['contrast_method']}  ·  {contrast_details.get(params['contrast_method'], '')}"))

    if params["sharpen"]:
        steps.append(("🔬 Sharpening",
                       f"Strength {params['sharpen_strength']}, radius σ={params['sharpen_blur_sigma']}"))
    if params["edge_enhance"]:
        steps.append(("🔮 Edge Enhancement", "Laplacian boost — sharpens fine edges"))

    if params["threshold_method"] != "none":
        thresh_details = {
            "adaptive": f"Block {params['adaptive_block_size']}, C={params['adaptive_C']}",
            "otsu":     "Automatic global threshold",
            "combined": f"Adaptive ∩ Otsu  (block {params['adaptive_block_size']}, C={params['adaptive_C']})",
            "sauvola":  f"Local Sauvola  (block {params['adaptive_block_size']})",
        }
        steps.append(("🎯 Binarization / Thresholding",
                       f"Method: {params['threshold_method']}  ·  {thresh_details.get(params['threshold_method'], '')}"))

    if params["morph_open"]:
        steps.append(("✂️ Morphological Opening",
                       f"Kernel {params['morph_open_ksize']}×{params['morph_open_ksize']} — removes speckle noise"))
    if params["morph_close"]:
        steps.append(("🔗 Morphological Closing",
                       f"Kernel {params['morph_close_ksize']}×{params['morph_close_ksize']} — fills gaps in text strokes"))
    if params["morph_dilate"]:
        steps.append(("➕ Dilation",
                       f"Kernel {params['morph_dilate_ksize']}×{params['morph_dilate_ksize']} — thickens strokes"))
    if params["morph_erode"]:
        steps.append(("➖ Erosion",
                       f"Kernel {params['morph_erode_ksize']}×{params['morph_erode_ksize']} — thins strokes"))
    if params["remove_borders"]:
        steps.append(("🖼️ Border Removal", f"Strips {params['border_size']} px from each edge"))
    if params["invert_output"]:
        steps.append(("🔄 Invert Output", "Flips black ↔ white"))

    for i, (name, detail) in enumerate(steps, 1):
        st.markdown(
            f'<div class="step-row">'
            f'<div class="step-num">{i}</div>'
            f'<div><div class="step-name">{name}</div>'
            f'<div class="step-detail">{detail}</div></div></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.caption(f"Total active steps: **{len(steps)}**  ·  Change sidebar controls and click ▶ Run Restoration to apply.")