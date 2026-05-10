# app.py — Enhanced Document Restoration App

import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from main import restore_document, get_image_histogram

st.set_page_config(
    page_title="Document Restoration Studio",
    page_icon="📄",
    layout="wide"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; margin-bottom: 0; }
    .section-header {
        font-size: 1rem; font-weight: 600; color: #666;
        text-transform: uppercase; letter-spacing: 1px;
        margin-top: 1.2rem; margin-bottom: 0.3rem;
    }
    .badge {
        display: inline-block; padding: 2px 10px;
        border-radius: 12px; font-size: 0.75rem; font-weight: 600;
        background: #e8f4f8; color: #1a6e8a; margin-left: 8px;
    }
    hr.divider { border: none; border-top: 1px solid #e0e0e0; margin: 0.8rem 0; }
    .stSlider > label { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ─── Title ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📄 Document Restoration Studio</p>', unsafe_allow_html=True)
st.caption("Upload a document image and fine-tune every step of the restoration pipeline.")
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─── File Upload ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a document image (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is None:
    st.info("👆 Upload an image to get started.")
    st.stop()

# Save to temp file
with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    tmp.write(uploaded_file.read())
    temp_path = tmp.name

original_bgr = cv2.imread(temp_path)
original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

# ─── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Controls")

    # ── Preset Profiles ──────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📋 Preset Profiles</p>', unsafe_allow_html=True)
    preset = st.selectbox(
        "Start from a preset",
        ["Custom", "Scanned Document", "Old / Yellowed Paper",
         "Low-Light Photo", "Faint Text", "High Contrast Print"]
    )

    PRESETS = {
        "Scanned Document": dict(
            normalize_background=True, norm_kernel_size=15,
            denoise_method="bilateral", bilateral_d=9,
            bilateral_sigma_color=75, bilateral_sigma_space=75,
            contrast_method="clahe", clahe_clip_limit=3.0, clahe_tile_size=8,
            sharpen=True, sharpen_strength=1.5, sharpen_blur_sigma=3.0,
            threshold_method="combined", adaptive_block_size=15, adaptive_C=10,
            morph_open=True, morph_open_ksize=2,
            morph_close=True, morph_close_ksize=3,
            deskew=False, invert_output=False, brightness=0,
            edge_enhance=False, remove_borders=False,
        ),
        "Old / Yellowed Paper": dict(
            normalize_background=True, norm_kernel_size=21,
            denoise_method="nlmeans", nlmeans_h=12,
            contrast_method="clahe", clahe_clip_limit=4.0, clahe_tile_size=8,
            sharpen=True, sharpen_strength=1.8, sharpen_blur_sigma=2.0,
            threshold_method="sauvola", adaptive_block_size=21, adaptive_C=10,
            morph_open=True, morph_open_ksize=2,
            morph_close=True, morph_close_ksize=2,
            deskew=True, invert_output=False, brightness=10,
            edge_enhance=False, remove_borders=True, border_size=8,
        ),
        "Low-Light Photo": dict(
            normalize_background=False, norm_kernel_size=15,
            denoise_method="bilateral", bilateral_d=11,
            bilateral_sigma_color=100, bilateral_sigma_space=100,
            contrast_method="gamma", gamma=1.8,
            sharpen=True, sharpen_strength=1.4, sharpen_blur_sigma=2.0,
            threshold_method="adaptive", adaptive_block_size=19, adaptive_C=8,
            morph_open=True, morph_open_ksize=2,
            morph_close=True, morph_close_ksize=3,
            deskew=False, invert_output=False, brightness=20,
            edge_enhance=False, remove_borders=False,
        ),
        "Faint Text": dict(
            normalize_background=True, norm_kernel_size=11,
            denoise_method="gaussian", gaussian_ksize=3,
            contrast_method="hist_eq",
            sharpen=True, sharpen_strength=2.0, sharpen_blur_sigma=1.5,
            threshold_method="sauvola", adaptive_block_size=13, adaptive_C=5,
            morph_open=False, morph_open_ksize=2,
            morph_close=True, morph_close_ksize=2,
            morph_dilate=True, morph_dilate_ksize=2,
            deskew=False, invert_output=False, brightness=30,
            edge_enhance=True, remove_borders=False,
        ),
        "High Contrast Print": dict(
            normalize_background=False, norm_kernel_size=15,
            denoise_method="median", median_ksize=3,
            contrast_method="none",
            sharpen=False, sharpen_strength=1.0, sharpen_blur_sigma=3.0,
            threshold_method="otsu", adaptive_block_size=15, adaptive_C=10,
            morph_open=True, morph_open_ksize=2,
            morph_close=False, morph_close_ksize=2,
            deskew=False, invert_output=False, brightness=0,
            edge_enhance=False, remove_borders=False,
        ),
    }

    use_preset = preset != "Custom"
    pv = PRESETS.get(preset, {})

    def pget(key, default):
        return pv.get(key, default) if use_preset else default

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Color Mode ────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">🎨 Color Mode</p>', unsafe_allow_html=True)
    color_mode = st.radio(
        "Process as",
        ["grayscale", "color"],
        index=0,
        horizontal=True,
        disabled=use_preset
    )

    # ── Corrections ───────────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🔧 Corrections</p>', unsafe_allow_html=True)

    deskew = st.checkbox(
        "Auto Deskew (rotate to straighten text)",
        value=pget("deskew", False),
        disabled=use_preset
    )
    remove_borders = st.checkbox(
        "Remove Border Artifacts",
        value=pget("remove_borders", False),
        disabled=use_preset
    )
    if remove_borders:
        border_size = st.slider("Border Strip (px)", 2, 40,
                                 pget("border_size", 10), disabled=use_preset)
    else:
        border_size = 10

    brightness = st.slider(
        "Brightness Adjustment", -100, 100,
        pget("brightness", 0),
        disabled=use_preset
    )
    invert_output = st.checkbox(
        "Invert Output (white-on-black → black-on-white)",
        value=pget("invert_output", False),
        disabled=use_preset
    )

    # ── Background Normalization ──────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🌅 Background Normalization</p>', unsafe_allow_html=True)
    normalize_background = st.checkbox(
        "Normalize Background",
        value=pget("normalize_background", True),
        disabled=use_preset
    )
    norm_kernel_size = st.slider(
        "Normalization Kernel Size", 5, 51,
        pget("norm_kernel_size", 15), step=2,
        help="Larger values remove broader illumination gradients.",
        disabled=use_preset or not normalize_background
    )

    # ── Noise Reduction ───────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🔇 Noise Reduction</p>', unsafe_allow_html=True)
    denoise_method = st.selectbox(
        "Denoising Method",
        ["bilateral", "median", "nlmeans", "gaussian", "none"],
        index=["bilateral", "median", "nlmeans", "gaussian", "none"].index(
            pget("denoise_method", "bilateral")),
        help="Bilateral preserves edges. NL-Means is best but slower.",
        disabled=use_preset
    )

    if denoise_method == "bilateral":
        bilateral_d = st.slider("Filter Diameter (d)", 3, 15,
                                 pget("bilateral_d", 9), step=2, disabled=use_preset)
        bilateral_sigma_color = st.slider("Sigma Color", 10, 200,
                                           pget("bilateral_sigma_color", 75), disabled=use_preset)
        bilateral_sigma_space = st.slider("Sigma Space", 10, 200,
                                           pget("bilateral_sigma_space", 75), disabled=use_preset)
    else:
        bilateral_d, bilateral_sigma_color, bilateral_sigma_space = 9, 75, 75

    if denoise_method == "median":
        median_ksize = st.slider("Kernel Size", 3, 11,
                                  pget("median_ksize", 5), step=2, disabled=use_preset)
    else:
        median_ksize = 5

    if denoise_method == "nlmeans":
        nlmeans_h = st.slider("Filter Strength (h)", 1, 30,
                               pget("nlmeans_h", 10), disabled=use_preset)
    else:
        nlmeans_h = 10

    if denoise_method == "gaussian":
        gaussian_ksize = st.slider("Kernel Size", 3, 15,
                                    pget("gaussian_ksize", 5), step=2, disabled=use_preset)
    else:
        gaussian_ksize = 5

    # ── Contrast Enhancement ──────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🌓 Contrast Enhancement</p>', unsafe_allow_html=True)
    contrast_method = st.selectbox(
        "Contrast Method",
        ["clahe", "hist_eq", "gamma", "none"],
        index=["clahe", "hist_eq", "gamma", "none"].index(
            pget("contrast_method", "clahe")),
        help="CLAHE is best for uneven lighting. Gamma brightens/darkens globally.",
        disabled=use_preset
    )

    if contrast_method == "clahe":
        clahe_clip_limit = st.slider("Clip Limit", 0.5, 8.0,
                                      pget("clahe_clip_limit", 3.0), step=0.5,
                                      disabled=use_preset)
        clahe_tile_size = st.slider("Tile Size", 4, 16,
                                     pget("clahe_tile_size", 8), step=2,
                                     disabled=use_preset)
    else:
        clahe_clip_limit, clahe_tile_size = 3.0, 8

    if contrast_method == "gamma":
        gamma = st.slider("Gamma Value", 0.2, 3.0,
                           pget("gamma", 1.0), step=0.1,
                           help="< 1.0 = brighter, > 1.0 = darker",
                           disabled=use_preset)
    else:
        gamma = 1.0

    # ── Sharpening ────────────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🔬 Sharpening</p>', unsafe_allow_html=True)
    sharpen = st.checkbox("Enable Sharpening",
                           value=pget("sharpen", True), disabled=use_preset)
    sharpen_strength = st.slider(
        "Sharpening Strength", 1.0, 3.0,
        pget("sharpen_strength", 1.5), step=0.1,
        help="How aggressively to sharpen (1.0 = off, 2.0 = strong).",
        disabled=use_preset or not sharpen
    )
    sharpen_blur_sigma = st.slider(
        "Sharpening Radius (sigma)", 0.5, 5.0,
        pget("sharpen_blur_sigma", 3.0), step=0.5,
        disabled=use_preset or not sharpen
    )
    edge_enhance = st.checkbox(
        "Edge Enhancement (Laplacian boost)",
        value=pget("edge_enhance", False),
        disabled=use_preset
    )

    # ── Thresholding ──────────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🎯 Binarization / Thresholding</p>', unsafe_allow_html=True)
    threshold_method = st.selectbox(
        "Threshold Method",
        ["adaptive", "otsu", "combined", "sauvola", "none"],
        index=["adaptive", "otsu", "combined", "sauvola", "none"].index(
            pget("threshold_method", "adaptive")),
        help="Sauvola is excellent for old/degraded documents.",
        disabled=use_preset
    )

    adaptive_block_size = st.slider(
        "Block Size (odd)", 5, 51,
        pget("adaptive_block_size", 15), step=2,
        help="Neighbourhood size for local thresholding.",
        disabled=use_preset or threshold_method in ("otsu", "none")
    )
    adaptive_C = st.slider(
        "Constant C", 0, 30,
        pget("adaptive_C", 10),
        help="Subtracted from neighbourhood mean. Higher = more aggressive.",
        disabled=use_preset or threshold_method in ("otsu", "none")
    )

    # ── Morphology ────────────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🔩 Morphological Operations</p>', unsafe_allow_html=True)
    morph_open = st.checkbox("Opening (remove speckles)",
                              value=pget("morph_open", True), disabled=use_preset)
    morph_open_ksize = st.slider("Open Kernel Size", 1, 7,
                                  pget("morph_open_ksize", 2),
                                  disabled=use_preset or not morph_open)

    morph_close = st.checkbox("Closing (fill gaps in text)",
                               value=pget("morph_close", True), disabled=use_preset)
    morph_close_ksize = st.slider("Close Kernel Size", 1, 7,
                                   pget("morph_close_ksize", 3),
                                   disabled=use_preset or not morph_close)

    morph_dilate = st.checkbox("Dilation (thicken strokes)",
                                value=pget("morph_dilate", False), disabled=use_preset)
    morph_dilate_ksize = st.slider("Dilate Kernel Size", 1, 5,
                                    pget("morph_dilate_ksize", 2),
                                    disabled=use_preset or not morph_dilate)

    morph_erode = st.checkbox("Erosion (thin strokes)",
                               value=pget("morph_erode", False), disabled=use_preset)
    morph_erode_ksize = st.slider("Erode Kernel Size", 1, 5,
                                   pget("morph_erode_ksize", 2),
                                   disabled=use_preset or not morph_erode)

# ─── Main Panel ───────────────────────────────────────────────────────────────
tab_restore, tab_compare, tab_hist, tab_info = st.tabs(
    ["🖼️ Restore", "🔀 Side-by-Side Compare", "📊 Histogram", "ℹ️ Pipeline Info"]
)

# Gather params from sidebar
params = dict(
    color_mode=color_mode if not use_preset else "grayscale",
    deskew=deskew if not use_preset else pv.get("deskew", False),
    remove_borders=remove_borders if not use_preset else pv.get("remove_borders", False),
    border_size=border_size if not use_preset else pv.get("border_size", 10),
    brightness=brightness if not use_preset else pv.get("brightness", 0),
    invert_output=invert_output if not use_preset else pv.get("invert_output", False),
    normalize_background=normalize_background if not use_preset else pv.get("normalize_background", True),
    norm_kernel_size=norm_kernel_size if not use_preset else pv.get("norm_kernel_size", 15),
    denoise_method=denoise_method if not use_preset else pv.get("denoise_method", "bilateral"),
    bilateral_d=bilateral_d if not use_preset else pv.get("bilateral_d", 9),
    bilateral_sigma_color=bilateral_sigma_color if not use_preset else pv.get("bilateral_sigma_color", 75),
    bilateral_sigma_space=bilateral_sigma_space if not use_preset else pv.get("bilateral_sigma_space", 75),
    median_ksize=median_ksize if not use_preset else pv.get("median_ksize", 5),
    nlmeans_h=nlmeans_h if not use_preset else pv.get("nlmeans_h", 10),
    gaussian_ksize=gaussian_ksize if not use_preset else pv.get("gaussian_ksize", 5),
    contrast_method=contrast_method if not use_preset else pv.get("contrast_method", "clahe"),
    clahe_clip_limit=clahe_clip_limit if not use_preset else pv.get("clahe_clip_limit", 3.0),
    clahe_tile_size=clahe_tile_size if not use_preset else pv.get("clahe_tile_size", 8),
    gamma=gamma if not use_preset else pv.get("gamma", 1.0),
    sharpen=sharpen if not use_preset else pv.get("sharpen", True),
    sharpen_strength=sharpen_strength if not use_preset else pv.get("sharpen_strength", 1.5),
    sharpen_blur_sigma=sharpen_blur_sigma if not use_preset else pv.get("sharpen_blur_sigma", 3.0),
    edge_enhance=edge_enhance if not use_preset else pv.get("edge_enhance", False),
    threshold_method=threshold_method if not use_preset else pv.get("threshold_method", "adaptive"),
    adaptive_block_size=adaptive_block_size if not use_preset else pv.get("adaptive_block_size", 15),
    adaptive_C=adaptive_C if not use_preset else pv.get("adaptive_C", 10),
    morph_open=morph_open if not use_preset else pv.get("morph_open", True),
    morph_open_ksize=morph_open_ksize if not use_preset else pv.get("morph_open_ksize", 2),
    morph_close=morph_close if not use_preset else pv.get("morph_close", True),
    morph_close_ksize=morph_close_ksize if not use_preset else pv.get("morph_close_ksize", 3),
    morph_dilate=morph_dilate if not use_preset else pv.get("morph_dilate", False),
    morph_dilate_ksize=morph_dilate_ksize if not use_preset else pv.get("morph_dilate_ksize", 2),
    morph_erode=morph_erode if not use_preset else pv.get("morph_erode", False),
    morph_erode_ksize=morph_erode_ksize if not use_preset else pv.get("morph_erode_ksize", 2),
)

# ── Tab 1: Restore ─────────────────────────────────────────────────────────
with tab_restore:
    col_orig, col_out = st.columns(2)

    with col_orig:
        st.subheader("Original Image")
        h, w = original_rgb.shape[:2]
        st.image(original_rgb, use_container_width=True)
        st.caption(f"Size: {w} × {h} px  |  "
                   f"File: {uploaded_file.name}  |  "
                   f"{round(uploaded_file.size / 1024, 1)} KB")

    with col_out:
        st.subheader("Restored Image")
        restore_placeholder = st.empty()

    run = st.button("▶ Run Restoration", type="primary", use_container_width=True)

    if run:
        with st.spinner("Applying pipeline…"):
            result = restore_document(temp_path, show_steps=False, params=params)

        if result is not None:
            base_path = os.path.splitext(temp_path)[0]
            output_path = f"{base_path}_restored.png"

            if params["color_mode"] == "color" and len(result.shape) == 3:
                result_display = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            else:
                result_display = result

            with col_out:
                restore_placeholder.image(result_display, use_container_width=True,
                                           clamp=True)

            st.success("✅ Document restored successfully!")

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                with open(output_path, "rb") as f:
                    st.download_button(
                        "⬇ Download Restored (PNG)",
                        data=f,
                        file_name="restored_document.png",
                        mime="image/png",
                        use_container_width=True
                    )
            with col_dl2:
                # Also offer JPEG
                ret, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ret:
                    st.download_button(
                        "⬇ Download Restored (JPEG)",
                        data=buf.tobytes(),
                        file_name="restored_document.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
        else:
            st.error("❌ Failed to process image.")

# ── Tab 2: Side-by-Side Compare ────────────────────────────────────────────
with tab_compare:
    st.subheader("Side-by-Side Comparison")
    base_path = os.path.splitext(temp_path)[0]
    output_path = f"{base_path}_restored.png"

    if not os.path.exists(output_path):
        st.info("Run the restoration first (▶ Run Restoration) to see the comparison.")
    else:
        result_bgr = cv2.imread(output_path)
        if result_bgr is not None:
            if len(result_bgr.shape) == 3:
                result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            else:
                result_rgb = result_bgr

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original**")
                st.image(original_rgb, use_container_width=True)
            with col2:
                st.markdown("**Restored**")
                st.image(result_rgb, use_container_width=True)

            # Diff image
            with st.expander("🔎 Show Difference Map"):
                orig_gray_r = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
                res_gray_r = cv2.resize(
                    cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY) if len(result_bgr.shape) == 3
                    else result_bgr,
                    (orig_gray_r.shape[1], orig_gray_r.shape[0])
                )
                diff = cv2.absdiff(orig_gray_r, res_gray_r)
                diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
                st.image(cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB),
                         caption="Difference heatmap (red = large change)",
                         use_container_width=True)

# ── Tab 3: Histogram ───────────────────────────────────────────────────────
with tab_hist:
    st.subheader("Pixel Intensity Histogram")
    base_path = os.path.splitext(temp_path)[0]
    output_path = f"{base_path}_restored.png"

    import matplotlib.pyplot as plt
    import io

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0e1117")

    for ax, (img_src, label) in zip(
        axes,
        [(original_gray, "Original"), (None, "Restored")]
    ):
        if label == "Restored":
            if os.path.exists(output_path):
                res = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
                if res is None:
                    ax.set_title("Restored (not yet run)", color="white")
                    ax.set_facecolor("#0e1117")
                    continue
                img_src = res
            else:
                ax.set_title("Restored (not yet run)", color="white")
                ax.set_facecolor("#0e1117")
                continue

        hist = cv2.calcHist([img_src], [0], None, [256], [0, 256]).flatten()
        ax.set_facecolor("#0e1117")
        ax.fill_between(range(256), hist, alpha=0.6,
                         color="#4fc3f7" if label == "Original" else "#81c784")
        ax.plot(hist, color="#4fc3f7" if label == "Original" else "#81c784", linewidth=1)
        ax.set_title(label, color="white", fontsize=13)
        ax.set_xlabel("Pixel Value", color="#aaa")
        ax.set_ylabel("Count", color="#aaa")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor="#0e1117")
    st.image(buf.getvalue(), use_container_width=True)
    plt.close(fig)

# ── Tab 4: Pipeline Info ───────────────────────────────────────────────────
with tab_info:
    st.subheader("Active Pipeline Configuration")

    step_icons = {
        "deskew": "📐",
        "normalize_background": "🌅",
        "brightness": "💡",
        "denoise_method": "🔇",
        "contrast_method": "🌓",
        "sharpen": "🔬",
        "edge_enhance": "🔮",
        "threshold_method": "🎯",
        "morph_open": "✂️",
        "morph_close": "🔗",
        "morph_dilate": "➕",
        "morph_erode": "➖",
        "remove_borders": "🖼️",
        "invert_output": "🔄",
    }

    active_steps = []
    if params["deskew"]:
        active_steps.append(("📐 Deskew", "Auto-corrects document rotation"))
    if params["normalize_background"]:
        active_steps.append(("🌅 Background Normalization",
                              f"Kernel {params['norm_kernel_size']}×{params['norm_kernel_size']}"))
    if params["brightness"] != 0:
        active_steps.append(("💡 Brightness", f"Offset: {params['brightness']:+d}"))
    active_steps.append(("🔇 Denoise",
                          f"Method: {params['denoise_method']}" + (
                              f" | d={params['bilateral_d']}, σ_c={params['bilateral_sigma_color']}"
                              if params["denoise_method"] == "bilateral" else "")))
    if params["contrast_method"] != "none":
        extra = ""
        if params["contrast_method"] == "clahe":
            extra = f" | clip={params['clahe_clip_limit']}, tile={params['clahe_tile_size']}"
        elif params["contrast_method"] == "gamma":
            extra = f" | γ={params['gamma']}"
        active_steps.append(("🌓 Contrast Enhancement",
                              f"Method: {params['contrast_method']}{extra}"))
    if params["sharpen"]:
        active_steps.append(("🔬 Sharpening",
                              f"Strength: {params['sharpen_strength']}, σ={params['sharpen_blur_sigma']}"))
    if params["edge_enhance"]:
        active_steps.append(("🔮 Edge Enhancement", "Laplacian boost"))
    if params["threshold_method"] != "none":
        active_steps.append(("🎯 Thresholding",
                              f"Method: {params['threshold_method']}" + (
                                  f" | block={params['adaptive_block_size']}, C={params['adaptive_C']}"
                                  if params["threshold_method"] != "otsu" else "")))
    if params["morph_open"]:
        active_steps.append(("✂️ Morphological Opening",
                              f"Kernel {params['morph_open_ksize']}×{params['morph_open_ksize']}"))
    if params["morph_close"]:
        active_steps.append(("🔗 Morphological Closing",
                              f"Kernel {params['morph_close_ksize']}×{params['morph_close_ksize']}"))
    if params["morph_dilate"]:
        active_steps.append(("➕ Dilation",
                              f"Kernel {params['morph_dilate_ksize']}×{params['morph_dilate_ksize']}"))
    if params["morph_erode"]:
        active_steps.append(("➖ Erosion",
                              f"Kernel {params['morph_erode_ksize']}×{params['morph_erode_ksize']}"))
    if params["remove_borders"]:
        active_steps.append(("🖼️ Border Removal",
                              f"Strip: {params['border_size']} px"))
    if params["invert_output"]:
        active_steps.append(("🔄 Invert Output", "Flips pixel values"))

    for i, (name, detail) in enumerate(active_steps):
        st.markdown(f"**{i+1}. {name}**")
        st.caption(detail)

    st.markdown("---")
    st.caption("All parameters are live — change sidebar sliders and press ▶ Run Restoration again.")