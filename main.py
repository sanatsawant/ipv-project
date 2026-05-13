import cv2
import numpy as np
import os
import difflib
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\sanat\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# =========================================================
# Tesseract Check
# =========================================================

def check_tesseract():
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# =========================================================
# OCR Preprocessing Helpers
# =========================================================

def _upscale_for_ocr(img, min_width=1200):
    """Upscale so width >= min_width — small images kill OCR accuracy."""
    h, w = img.shape[:2]
    if w < min_width:
        scale = min_width / w
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    return img


def _preprocess_for_ocr(gray):
    """
    Returns 5 differently-preprocessed candidates.
    Tesseract runs on all of them; the best-scoring result wins.
    """
    candidates = []

    # 1. CLAHE + light denoise  → best for most scanned docs
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    c1 = clahe.apply(gray)
    c1 = cv2.fastNlMeansDenoising(c1, h=7)
    candidates.append(c1)

    # 2. Adaptive threshold     → best for uneven lighting / shadows
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    c2 = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 10
    )
    candidates.append(c2)

    # 3. Otsu threshold          → best for clean, high-contrast print
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, c3 = cv2.threshold(blurred, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(c3)

    # 4. Aggressive sharpen + high-clip CLAHE → best for faint / blurry text
    clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    c4 = clahe2.apply(gray)
    blur = cv2.GaussianBlur(c4, (0, 0), 2.0)
    c4 = cv2.addWeighted(c4, 1.8, blur, -0.8, 0)
    c4 = np.clip(c4, 0, 255).astype(np.uint8)
    candidates.append(c4)

    # 5. Background-normalised  → best for yellowed / shadowed documents
    bg  = cv2.morphologyEx(gray, cv2.MORPH_DILATE,
                            np.ones((15, 15), np.uint8))
    c5  = cv2.divide(gray, bg, scale=255)
    _, c5 = cv2.threshold(c5, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(c5)

    return candidates


def _run_tesseract(gray, config):
    """Single Tesseract call. Returns (words, confs, boxes, mean_conf)."""
    try:
        data = pytesseract.image_to_data(
            gray, output_type=pytesseract.Output.DICT, config=config
        )
    except Exception:
        return [], [], [], 0.0

    words, confs, boxes = [], [], []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = int(data["conf"][i])
        if word and conf >= 0:
            words.append(word)
            confs.append(conf)
            boxes.append((word, conf,
                          data["left"][i], data["top"][i],
                          data["width"][i], data["height"][i]))

    mean_conf = float(np.mean(confs)) if confs else 0.0
    return words, confs, boxes, mean_conf


# =========================================================
# OCR Analysis Engine
# =========================================================

def ocr_analyze(img):
    """
    Improved multi-strategy OCR engine.
    - Upscales small images (biggest accuracy gain for low-res docs)
    - Generates 5 preprocessed candidates
    - Tries 4 Tesseract PSM layouts on each candidate  (20 attempts total)
    - Returns the result with the highest average confidence
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy().astype(np.uint8)

    # Always upscale — Tesseract struggles with small text
    gray = _upscale_for_ocr(gray, min_width=1200)

    candidates = _preprocess_for_ocr(gray)

    # PSM configs to try on every candidate
    configs = [
        "--psm 6  --oem 3",   # uniform block of text   (most common)
        "--psm 3  --oem 3",   # fully automatic layout
        "--psm 4  --oem 3",   # single column
        "--psm 11 --oem 3",   # sparse text — finds stray words
    ]

    best_words, best_confs, best_boxes, best_score = [], [], [], -1.0

    for candidate in candidates:
        for config in configs:
            words, confs, boxes, mean_conf = _run_tesseract(candidate, config)
            if mean_conf > best_score and words:
                best_score = mean_conf
                best_words = words
                best_confs = confs
                best_boxes = boxes

    if not best_words:
        return _empty_ocr_result(
            "No text detected. Make sure the image contains readable text."
        )

    high_conf_frac = sum(1 for c in best_confs if c >= 70) / len(best_confs)

    try:
        # Use the CLAHE candidate for final full-text extraction (cleanest text)
        full_text = pytesseract.image_to_string(
            candidates[0], config="--psm 6 --oem 3"
        ).strip()
    except Exception:
        full_text = " ".join(best_words)

    return {
        "words":          best_words,
        "confidences":    best_confs,
        "mean_conf":      round(best_score, 2),
        "word_count":     len(best_words),
        "char_count":     sum(len(w) for w in best_words),
        "full_text":      full_text,
        "boxes":          best_boxes,
        "high_conf_frac": round(high_conf_frac, 4),
        "error":          None,
    }


def _empty_ocr_result(error_msg):
    return {
        "words": [], "confidences": [], "mean_conf": 0.0,
        "word_count": 0, "char_count": 0, "full_text": "",
        "boxes": [], "high_conf_frac": 0.0, "error": error_msg,
    }


# =========================================================
# OCR Diff Engine
# =========================================================

def compute_ocr_diff(ocr_before, ocr_after):
    words_before = set(w.lower() for w in ocr_before["words"])
    words_after  = set(w.lower() for w in ocr_after["words"])

    unlocked  = sorted(words_after  - words_before)
    lost      = sorted(words_before - words_after)
    preserved = sorted(words_before & words_after)

    def conf_map(r):
        m = {}
        for word, conf in zip(r["words"], r["confidences"]):
            key = word.lower()
            m[key] = max(m.get(key, 0), conf)
        return m

    cm_before = conf_map(ocr_before)
    cm_after  = conf_map(ocr_after)

    conf_delta, improved_words, regressed_words = {}, [], []
    for w in preserved:
        delta = cm_after.get(w, 0) - cm_before.get(w, 0)
        conf_delta[w] = delta
        if delta > 5:
            improved_words.append(w)
        elif delta < -5:
            regressed_words.append(w)

    return {
        "score_before":        ocr_before["mean_conf"],
        "score_after":         ocr_after["mean_conf"],
        "score_delta":         round(ocr_after["mean_conf"] - ocr_before["mean_conf"], 2),
        "high_conf_before":    round(ocr_before["high_conf_frac"] * 100, 1),
        "high_conf_after":     round(ocr_after["high_conf_frac"]  * 100, 1),
        "word_count_before":   ocr_before["word_count"],
        "word_count_after":    ocr_after["word_count"],
        "char_count_before":   ocr_before["char_count"],
        "char_count_after":    ocr_after["char_count"],
        "unlocked_words":      unlocked,
        "lost_words":          lost,
        "preserved_words":     preserved,
        "improved_words":      improved_words,
        "regressed_words":     regressed_words,
        "conf_delta_per_word": conf_delta,
        "text_diff_html":      _build_text_diff_html(ocr_before["full_text"],
                                                      ocr_after["full_text"]),
        "full_text_before":    ocr_before["full_text"],
        "full_text_after":     ocr_after["full_text"],
    }


def _build_text_diff_html(text_before, text_after):
    words_a  = text_before.split()
    words_b  = text_after.split()
    matcher  = difflib.SequenceMatcher(None, words_a, words_b, autojunk=False)
    parts    = []
    G = 'background:#1b5e20;color:#a5d6a7;border-radius:3px;padding:1px 5px;margin:1px 2px;'
    R = 'background:#7f1d1d;color:#fca5a5;border-radius:3px;padding:1px 5px;margin:1px 2px;text-decoration:line-through;'

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(words_a[i1:i2]))
        elif tag == "insert":
            parts += [f'<span style="{G}">{w}</span>' for w in words_b[j1:j2]]
        elif tag == "delete":
            parts += [f'<span style="{R}">{w}</span>' for w in words_a[i1:i2]]
        elif tag == "replace":
            parts += [f'<span style="{R}">{w}</span>' for w in words_a[i1:i2]]
            parts += [f'<span style="{G}">{w}</span>' for w in words_b[j1:j2]]

    return " ".join(parts)


def generate_confidence_heatmap(img, ocr_data, alpha=0.45):
    if len(img.shape) == 2:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        base = img.copy()
    layer = base.copy()
    for (word, conf, x, y, w, h) in ocr_data["boxes"]:
        ratio = conf / 100.0
        cv2.rectangle(layer, (x, y), (x + w, y + h),
                      (0, int(255 * ratio), int(255 * (1 - ratio))), -1)
    return cv2.addWeighted(base, 1 - alpha, layer, alpha, 0)


# =========================================================
# Core Restoration Pipeline
# =========================================================

def restore_document(image_path, show_steps=True, params=None):
    default_params = {
        "normalize_background": True, "norm_kernel_size": 15,
        "denoise_method": "bilateral", "bilateral_d": 9,
        "bilateral_sigma_color": 75, "bilateral_sigma_space": 75,
        "median_ksize": 5, "nlmeans_h": 10, "gaussian_ksize": 5,
        "contrast_method": "clahe", "clahe_clip_limit": 3.0,
        "clahe_tile_size": 8, "gamma": 1.0,
        "sharpen": True, "sharpen_strength": 1.5, "sharpen_blur_sigma": 3.0,
        "threshold_method": "adaptive", "adaptive_block_size": 15, "adaptive_C": 10,
        "morph_open": True, "morph_open_ksize": 2,
        "morph_close": True, "morph_close_ksize": 3,
        "morph_dilate": False, "morph_dilate_ksize": 2,
        "morph_erode": False, "morph_erode_ksize": 2,
        "deskew": False, "remove_borders": False, "border_size": 10,
        "invert_output": False, "color_mode": "grayscale",
        "brightness": 0, "edge_enhance": False,
    }
    if params:
        default_params.update(params)
    p = default_params

    if p["color_mode"] == "color":
        original_img = cv2.imread(image_path)
        if original_img is None:
            return None
        lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        img = l
    else:
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            return None
        img = original_img.copy()

    if p["deskew"]:
        img = _deskew(img)

    if p["normalize_background"]:
        ks = int(p["norm_kernel_size"])
        if ks % 2 == 0: ks += 1
        bg = cv2.morphologyEx(img, cv2.MORPH_DILATE, np.ones((ks, ks), np.uint8))
        img = cv2.divide(img, bg, scale=255)

    if p["brightness"] != 0:
        img = np.clip(img.astype(np.int32) + int(p["brightness"]), 0, 255).astype(np.uint8)

    dm = p["denoise_method"]
    if dm == "bilateral":
        img = cv2.bilateralFilter(img, d=int(p["bilateral_d"]),
                                   sigmaColor=p["bilateral_sigma_color"],
                                   sigmaSpace=p["bilateral_sigma_space"])
    elif dm == "median":
        ks = int(p["median_ksize"])
        if ks % 2 == 0: ks += 1
        img = cv2.medianBlur(img, ks)
    elif dm == "nlmeans":
        img = cv2.fastNlMeansDenoising(img, h=p["nlmeans_h"])
    elif dm == "gaussian":
        ks = int(p["gaussian_ksize"])
        if ks % 2 == 0: ks += 1
        img = cv2.GaussianBlur(img, (ks, ks), 0)

    cm = p["contrast_method"]
    if cm == "clahe":
        clahe = cv2.createCLAHE(clipLimit=p["clahe_clip_limit"],
                                  tileGridSize=(int(p["clahe_tile_size"]),) * 2)
        img = clahe.apply(img)
    elif cm == "hist_eq":
        img = cv2.equalizeHist(img)
    elif cm == "gamma":
        gv = max(0.1, p["gamma"])
        lut = np.array([((i / 255.0) ** (1.0 / gv)) * 255
                        for i in range(256)], dtype=np.uint8)
        img = cv2.LUT(img, lut)

    if p["sharpen"]:
        s = p["sharpen_strength"]
        blur = cv2.GaussianBlur(img, (0, 0), p["sharpen_blur_sigma"])
        img = np.clip(cv2.addWeighted(img, s, blur, -(s - 1.0), 0), 0, 255).astype(np.uint8)

    if p["edge_enhance"]:
        lap = np.clip(np.abs(cv2.Laplacian(img, cv2.CV_64F)), 0, 255).astype(np.uint8)
        img = cv2.addWeighted(img, 1.0, lap, 0.5, 0)

    tm = p["threshold_method"]
    if tm != "none":
        bs = int(p["adaptive_block_size"])
        if bs % 2 == 0: bs += 1
        if bs < 3: bs = 3
        if tm == "adaptive":
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, bs, int(p["adaptive_C"]))
        elif tm == "otsu":
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif tm == "combined":
            adp = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, bs, int(p["adaptive_C"]))
            _, ots = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = cv2.bitwise_and(adp, ots)
        elif tm == "sauvola":
            img = _sauvola_threshold(img, window_size=bs)

    if p["morph_open"]:
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN,
                                np.ones((int(p["morph_open_ksize"]),) * 2, np.uint8))
    if p["morph_close"]:
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                                np.ones((int(p["morph_close_ksize"]),) * 2, np.uint8))
    if p["morph_dilate"]:
        img = cv2.dilate(img, np.ones((int(p["morph_dilate_ksize"]),) * 2, np.uint8))
    if p["morph_erode"]:
        img = cv2.erode(img, np.ones((int(p["morph_erode_ksize"]),) * 2, np.uint8))

    if p["remove_borders"]:
        bs = int(p["border_size"])
        if img.shape[0] > 2 * bs and img.shape[1] > 2 * bs:
            img = img[bs:-bs, bs:-bs]

    if p["invert_output"]:
        img = cv2.bitwise_not(img)

    final_output = img
    if p["color_mode"] == "color":
        final_output = cv2.cvtColor(cv2.merge([img, a, b_ch]), cv2.COLOR_LAB2BGR)

    base_path = os.path.splitext(image_path)[0]
    cv2.imwrite(f"{base_path}_restored.png", final_output)
    return final_output


def _deskew(img):
    coords = np.column_stack(np.where(img < 128))
    if coords.shape[0] < 10:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)


def _sauvola_threshold(img, window_size=15, k=0.2, R=128):
    f = img.astype(np.float64)
    mean = cv2.boxFilter(f, cv2.CV_64F, (window_size, window_size))
    mean_sq = cv2.boxFilter(f ** 2, cv2.CV_64F, (window_size, window_size))
    std = np.sqrt(np.clip(mean_sq - mean ** 2, 0, None))
    return np.where(f >= mean * (1 + k * (std / R - 1)), 255, 0).astype(np.uint8)


def restore_document_simple(image_path):
    return restore_document(image_path, show_steps=False)


def get_image_histogram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().tolist()


if __name__ == "__main__":
    restore_document('sample_document.jpg', show_steps=True)