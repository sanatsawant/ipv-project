import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# =========================================================
# Core Restoration Pipeline (Fully Parameterized)
# =========================================================

def restore_document(image_path, show_steps=True, params=None):
    """
    Enhanced document restoration pipeline.
    All parameters are tunable via the `params` dict.
    """
    default_params = {
        # Background normalization
        "normalize_background": True,
        "norm_kernel_size": 15,

        # Noise reduction
        "denoise_method": "bilateral",       # "bilateral", "median", "nlmeans", "gaussian", "none"
        "bilateral_d": 9,
        "bilateral_sigma_color": 75,
        "bilateral_sigma_space": 75,
        "median_ksize": 5,
        "nlmeans_h": 10,
        "gaussian_ksize": 5,

        # Contrast enhancement
        "contrast_method": "clahe",          # "clahe", "hist_eq", "gamma", "none"
        "clahe_clip_limit": 3.0,
        "clahe_tile_size": 8,
        "gamma": 1.0,

        # Sharpening
        "sharpen": True,
        "sharpen_strength": 1.5,             # 1.0 = no change, >1 = sharper
        "sharpen_blur_sigma": 3.0,

        # Thresholding
        "threshold_method": "adaptive",      # "adaptive", "otsu", "combined", "sauvola", "none"
        "adaptive_block_size": 15,
        "adaptive_C": 10,

        # Morphological ops
        "morph_open": True,
        "morph_open_ksize": 2,
        "morph_close": True,
        "morph_close_ksize": 3,
        "morph_dilate": False,
        "morph_dilate_ksize": 2,
        "morph_erode": False,
        "morph_erode_ksize": 2,

        # Extra features
        "deskew": False,
        "remove_borders": False,
        "border_size": 10,
        "invert_output": False,
        "color_mode": "grayscale",            # "grayscale", "color"
        "brightness": 0,                      # -100 to +100
        "edge_enhance": False,
    }

    if params:
        default_params.update(params)
    p = default_params

    # ---------------------------------------------------------
    # Step 1: Load Image
    # ---------------------------------------------------------
    if p["color_mode"] == "color":
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"Error: Could not load image at {image_path}")
            return None
        work_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(work_img)
        img = l
    else:
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            print(f"Error: Could not load image at {image_path}")
            return None
        img = original_img.copy()

    print(f"Image loaded: {original_img.shape}")

    # ---------------------------------------------------------
    # Step 2: Deskew
    # ---------------------------------------------------------
    if p["deskew"]:
        img = _deskew(img)

    # ---------------------------------------------------------
    # Step 3: Background Normalization
    # ---------------------------------------------------------
    if p["normalize_background"]:
        ks = int(p["norm_kernel_size"])
        if ks % 2 == 0:
            ks += 1
        background = cv2.morphologyEx(img, cv2.MORPH_DILATE,
                                      np.ones((ks, ks), np.uint8))
        img = cv2.divide(img, background, scale=255)

    # ---------------------------------------------------------
    # Step 4: Brightness Adjustment
    # ---------------------------------------------------------
    if p["brightness"] != 0:
        img = np.clip(img.astype(np.int32) + int(p["brightness"]), 0, 255).astype(np.uint8)

    # ---------------------------------------------------------
    # Step 5: Noise Reduction
    # ---------------------------------------------------------
    method = p["denoise_method"]
    if method == "bilateral":
        img = cv2.bilateralFilter(img,
                                   d=int(p["bilateral_d"]),
                                   sigmaColor=p["bilateral_sigma_color"],
                                   sigmaSpace=p["bilateral_sigma_space"])
    elif method == "median":
        ks = int(p["median_ksize"])
        if ks % 2 == 0:
            ks += 1
        img = cv2.medianBlur(img, ks)
    elif method == "nlmeans":
        img = cv2.fastNlMeansDenoising(img, h=p["nlmeans_h"])
    elif method == "gaussian":
        ks = int(p["gaussian_ksize"])
        if ks % 2 == 0:
            ks += 1
        img = cv2.GaussianBlur(img, (ks, ks), 0)
    # "none" → skip

    # ---------------------------------------------------------
    # Step 6: Contrast Enhancement
    # ---------------------------------------------------------
    cm = p["contrast_method"]
    if cm == "clahe":
        clahe = cv2.createCLAHE(clipLimit=p["clahe_clip_limit"],
                                  tileGridSize=(int(p["clahe_tile_size"]),
                                                int(p["clahe_tile_size"])))
        img = clahe.apply(img)
    elif cm == "hist_eq":
        img = cv2.equalizeHist(img)
    elif cm == "gamma":
        gamma = max(0.1, p["gamma"])
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                        for i in range(256)], dtype=np.uint8)
        img = cv2.LUT(img, lut)
    # "none" → skip

    # ---------------------------------------------------------
    # Step 7: Sharpening
    # ---------------------------------------------------------
    if p["sharpen"]:
        strength = p["sharpen_strength"]
        gaussian_blur = cv2.GaussianBlur(img, (0, 0), p["sharpen_blur_sigma"])
        img = cv2.addWeighted(img, strength, gaussian_blur, -(strength - 1.0), 0)
        img = np.clip(img, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------
    # Step 8: Edge Enhancement
    # ---------------------------------------------------------
    if p["edge_enhance"]:
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        laplacian = np.clip(np.abs(laplacian), 0, 255).astype(np.uint8)
        img = cv2.addWeighted(img, 1.0, laplacian, 0.5, 0)

    # ---------------------------------------------------------
    # Step 9: Thresholding
    # ---------------------------------------------------------
    tm = p["threshold_method"]
    if tm != "none":
        bs = int(p["adaptive_block_size"])
        if bs % 2 == 0:
            bs += 1
        if bs < 3:
            bs = 3

        if tm == "adaptive":
            img = cv2.adaptiveThreshold(img, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, bs, int(p["adaptive_C"]))
        elif tm == "otsu":
            _, img = cv2.threshold(img, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif tm == "combined":
            adaptive = cv2.adaptiveThreshold(img, 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, bs, int(p["adaptive_C"]))
            _, otsu = cv2.threshold(img, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = cv2.bitwise_and(adaptive, otsu)
        elif tm == "sauvola":
            img = _sauvola_threshold(img, window_size=bs)
    # "none" → no binarization

    # ---------------------------------------------------------
    # Step 10: Morphological Operations
    # ---------------------------------------------------------
    if p["morph_open"]:
        ks = int(p["morph_open_ksize"])
        kernel = np.ones((ks, ks), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    if p["morph_close"]:
        ks = int(p["morph_close_ksize"])
        kernel = np.ones((ks, ks), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    if p["morph_dilate"]:
        ks = int(p["morph_dilate_ksize"])
        kernel = np.ones((ks, ks), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)

    if p["morph_erode"]:
        ks = int(p["morph_erode_ksize"])
        kernel = np.ones((ks, ks), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)

    # ---------------------------------------------------------
    # Step 11: Remove Borders
    # ---------------------------------------------------------
    if p["remove_borders"]:
        bs = int(p["border_size"])
        img = img[bs:-bs, bs:-bs] if img.shape[0] > 2 * bs and img.shape[1] > 2 * bs else img

    # ---------------------------------------------------------
    # Step 12: Invert
    # ---------------------------------------------------------
    if p["invert_output"]:
        img = cv2.bitwise_not(img)

    # ---------------------------------------------------------
    # Step 13: Merge back to color if needed
    # ---------------------------------------------------------
    final_output = img
    if p["color_mode"] == "color":
        merged = cv2.merge([img, a, b])
        final_output = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # ---------------------------------------------------------
    # Save Output
    # ---------------------------------------------------------
    base_path = os.path.splitext(image_path)[0]
    output_path = f"{base_path}_restored.png"
    cv2.imwrite(output_path, final_output)
    print(f"Restored image saved to: {output_path}")

    return final_output


# =========================================================
# Helpers
# =========================================================

def _deskew(img):
    """Deskew a grayscale image using moments."""
    coords = np.column_stack(np.where(img < 128))
    if coords.shape[0] < 10:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _sauvola_threshold(img, window_size=15, k=0.2, R=128):
    """Sauvola local thresholding — excellent for degraded documents."""
    img_float = img.astype(np.float64)
    mean = cv2.boxFilter(img_float, cv2.CV_64F, (window_size, window_size))
    mean_sq = cv2.boxFilter(img_float ** 2, cv2.CV_64F, (window_size, window_size))
    std = np.sqrt(mean_sq - mean ** 2)
    threshold = mean * (1 + k * (std / R - 1))
    binary = np.where(img_float >= threshold, 255, 0).astype(np.uint8)
    return binary


def restore_document_simple(image_path):
    """Quick simplified pipeline with defaults."""
    return restore_document(image_path, show_steps=False)


def get_image_histogram(img_bgr_or_gray):
    """Return histogram data as a list of (bin_center, count) tuples."""
    if len(img_bgr_or_gray.shape) == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr_or_gray
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist.flatten().tolist()


# =========================================================
# Execution
# =========================================================
if __name__ == "__main__":
    restore_document('sample_document.jpg', show_steps=True)