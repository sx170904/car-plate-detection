##############################################################
# Coded By: Lew Yun Cheng
# Streamlit demo for SSPD (Single-Shot Plate Detector) + OCR
##############################################################
import json
import time
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from model_arch import SSPD

# ---------- Page setup ----------
st.set_page_config(page_title="License Plate Detector", layout="wide")
st.title("🚗 License Plate Detection & Recognition — SSPD + EasyOCR Demo")
st.caption("UCCD3094 Group 6 — Custom SSPD detector (our contribution) + EasyOCR for character recognition")

# ---------- Model loading (cached) ----------
@st.cache_resource
def load_model():
    with open("model_config.json") as f:
        cfg = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSPD(pretrained=False, freeze_early=False, grid_size=cfg["grid_size"]).to(device)
    model.load_state_dict(torch.load("sspd_best.pth", map_location=device))
    model.eval()
    return model, device, cfg

@st.cache_resource
def load_ocr_reader():
    ######################################################
    # Adapted from https://github.com/JaidedAI/EasyOCR
    ######################################################
    import easyocr
    # gpu=False keeps the demo portable; change to True if you want GPU OCR.
    return easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)

model, device, cfg = load_model()
INPUT_SIZE = cfg["input_size"]

transform = T.Compose([
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------- Sidebar controls ----------
st.sidebar.header("Detection settings")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
iou_threshold  = st.sidebar.slider("NMS IoU threshold",    0.0, 1.0, 0.4, 0.05)
max_boxes      = st.sidebar.number_input("Max boxes to display", 1, 20, 5)

st.sidebar.markdown("---")
st.sidebar.header("Recognition settings")
enable_ocr = st.sidebar.checkbox("Enable OCR (read plate text)", value=True)
ocr_min_conf = st.sidebar.slider("Min OCR confidence", 0.0, 1.0, 0.1, 0.05,
                                  help="Characters with OCR confidence below this are dropped.")

st.sidebar.markdown("---")
grid_size = cfg['grid_size']
st.sidebar.markdown(f"**Device:** `{device.type.upper()}`")
st.sidebar.markdown(f"**Input size:** `{INPUT_SIZE}×{INPUT_SIZE}`")
st.sidebar.markdown(f"**Grid size:** `{grid_size}×{grid_size}`")


# ---------- Helpers ----------
#################################
# Coded By: Lew Yun Cheng
#################################

def clean_plate_text(raw_text: str) -> str:
    """Normalize OCR output — uppercase, keep only alphanumerics and spaces."""
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
    cleaned = ''.join(ch for ch in raw_text.upper() if ch in allowed)
    # Collapse multiple spaces
    return ' '.join(cleaned.split())


def recognize_plate(pil_img, cx, cy, w, h, reader, min_conf):
    """
    Crop the plate region from the PIL image and run EasyOCR on it.
    Returns (plate_text, mean_ocr_confidence, raw_results).
    """
    W, H = pil_img.size
    # Expand the crop by a small margin (10% each side) so OCR gets full characters
    margin = 0.05
    x1 = max(0,     int((cx - w/2 - margin * w) * W))
    y1 = max(0,     int((cy - h/2 - margin * h) * H))
    x2 = min(W - 1, int((cx + w/2 + margin * w) * W))
    y2 = min(H - 1, int((cy + h/2 + margin * h) * H))
    if x2 <= x1 or y2 <= y1:
        return "", 0.0, []
    plate_crop = np.array(pil_img)[y1:y2, x1:x2]

    # EasyOCR returns a list of (bbox, text, confidence) tuples
    results = reader.readtext(plate_crop)
    # Filter by OCR confidence, sort left-to-right
    filtered = [(bb, txt, c) for (bb, txt, c) in results if c >= min_conf]
    filtered.sort(key=lambda r: r[0][0][0])  # sort by leftmost-x of bounding box
    text = clean_plate_text(' '.join(txt for _, txt, _ in filtered))
    mean_conf = float(np.mean([c for _, _, c in filtered])) if filtered else 0.0
    return text, mean_conf, plate_crop


def draw_detections(pil_img, boxes_normalized, ocr_texts=None):
    """
    Draw boxes + optional plate text labels above each box.
    ocr_texts: list of strings matching boxes_normalized order, or None to disable.
    """
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    try:
        font       = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(14, W // 40))
        font_large = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(18, W // 28))
    except Exception:
        font = font_large = ImageFont.load_default()

    for i, (cx, cy, w, h, conf) in enumerate(boxes_normalized):
        x1 = (cx - w/2) * W; y1 = (cy - h/2) * H
        x2 = (cx + w/2) * W; y2 = (cy + h/2) * H
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Confidence label on top-left of box
        conf_label = f"plate {conf:.2f}"
        tw = draw.textlength(conf_label, font=font)
        th = font.size + 4
        draw.rectangle([x1, max(0, y1 - th), x1 + tw + 6, y1], fill="red")
        draw.text((x1 + 3, max(0, y1 - th)), conf_label, fill="white", font=font)

        # OCR text label below the box (in a larger, high-contrast style)
        if ocr_texts and i < len(ocr_texts) and ocr_texts[i]:
            text = ocr_texts[i]
            tw = draw.textlength(text, font=font_large)
            th2 = font_large.size + 6
            # Place it below the box if there's space, otherwise above
            ty = y2 + 2 if y2 + th2 < H else max(0, y1 - th - th2 - 2)
            draw.rectangle([x1, ty, x1 + tw + 10, ty + th2], fill="black")
            draw.text((x1 + 5, ty + 2), text, fill="yellow", font=font_large)
    return img


def run_and_display(pil_img):
    """Run detection (+ optional OCR) on a PIL image and render the full results UI."""
    #################################
    # Coded By: Lew Yun Cheng
    #################################

    # ----- Detection -----
    x = transform(pil_img).unsqueeze(0).to(device)
    t0 = time.time()
    decoded = model.decode(x, conf_threshold=conf_threshold,
                           iou_threshold=iou_threshold, max_boxes=int(max_boxes))[0]
    if device.type == "cuda": torch.cuda.synchronize()
    det_latency_ms = (time.time() - t0) * 1000

    boxes_np = decoded.cpu().numpy() if decoded.numel() else np.zeros((0, 5))

    # ----- OCR (if enabled and something was detected) -----
    ocr_texts = []
    ocr_confs = []
    ocr_crops = []
    ocr_latency_ms = 0.0
    if enable_ocr and len(boxes_np) > 0:
        reader = load_ocr_reader()
        t1 = time.time()
        for (cx, cy, w, h, _) in boxes_np:
            text, mconf, crop = recognize_plate(pil_img, cx, cy, w, h, reader, ocr_min_conf)
            ocr_texts.append(text)
            ocr_confs.append(mconf)
            ocr_crops.append(crop)
        ocr_latency_ms = (time.time() - t1) * 1000

    # ----- Side-by-side view -----
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input image")
        st.image(pil_img, use_container_width=True)
    with col2:
        st.subheader("Detections" + (" + Recognition" if enable_ocr else ""))
        annotated = draw_detections(pil_img, boxes_np,
                                    ocr_texts=ocr_texts if enable_ocr else None)
        st.image(annotated, use_container_width=True)

    # ----- Metric tiles -----
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Plates detected", len(boxes_np))
    m2.metric("Top detection conf.",
              f"{float(boxes_np[:, 4].max()):.3f}" if len(boxes_np) else "—")
    m3.metric("Detection latency", f"{det_latency_ms:.1f} ms")
    if enable_ocr:
        m4.metric("OCR latency", f"{ocr_latency_ms:.1f} ms")
    else:
        m4.metric("OCR latency", "off")

    # ----- Recognised plate(s) banner -----
    if enable_ocr and any(ocr_texts):
        st.subheader("📖 Recognised Plate Text")
        for i, (text, mconf) in enumerate(zip(ocr_texts, ocr_confs), 1):
            if text:
                st.success(f"**Plate {i}:**  `{text}`   (OCR confidence: {mconf:.2f})")
            else:
                st.warning(f"**Plate {i}:** Could not read text. Try lowering OCR min confidence.")

        # Show plate crops in an expander for transparency
        with st.expander("🔍 See cropped plate images fed to OCR"):
            crop_cols = st.columns(min(len(ocr_crops), 4))
            for i, (crop, text) in enumerate(zip(ocr_crops, ocr_texts)):
                if isinstance(crop, np.ndarray):
                    with crop_cols[i % len(crop_cols)]:
                        st.image(crop, caption=f"Plate {i+1}: {text or '(no text)'}",
                                 use_container_width=True)

    # ----- Detections table -----
    if len(boxes_np):
        st.subheader("Detection details")
        df_data = {
            "#":          np.arange(1, len(boxes_np) + 1),
            "cx":         boxes_np[:, 0],
            "cy":         boxes_np[:, 1],
            "w":          boxes_np[:, 2],
            "h":          boxes_np[:, 3],
            "det_conf":   boxes_np[:, 4],
        }
        if enable_ocr:
            df_data["plate_text"] = ocr_texts
            df_data["ocr_conf"]   = ocr_confs
        df = pd.DataFrame(df_data)
        fmt = {"cx": "{:.3f}", "cy": "{:.3f}", "w": "{:.3f}", "h": "{:.3f}",
               "det_conf": "{:.3f}"}
        if enable_ocr: fmt["ocr_conf"] = "{:.3f}"
        st.dataframe(df.style.format(fmt), use_container_width=True)
    else:
        st.warning("No plates detected above the confidence threshold. "
                   "Try lowering it in the sidebar.")

    # ----- Download annotated image -----
    buf = BytesIO()
    annotated.save(buf, format="PNG")
    st.download_button(
        label="⬇️  Download annotated image",
        data=buf.getvalue(),
        file_name="detection_result.png",
        mime="image/png",
    )


# ---------- Input source selection via tabs ----------
tab_upload, tab_camera = st.tabs(["📁 Upload image", "📷 Capture with camera"])

with tab_upload:
    uploaded = st.file_uploader("Upload a car image (JPG / PNG)",
                                type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        pil_img = Image.open(BytesIO(uploaded.read())).convert("RGB")
        run_and_display(pil_img)
    else:
        st.info("👆 Upload an image to get started.")
        st.markdown(
            "**Tip:** lower the confidence threshold in the sidebar if the detector "
            "misses the plate, or raise it to suppress spurious boxes. "
            "Toggle OCR in the sidebar to see/hide plate text."
        )

with tab_camera:
    st.markdown(
        "Use your device camera to capture a live photo of a car. "
        "On first use, your browser will ask for camera permission — click **Allow**."
    )
    camera_img = st.camera_input("Take a photo")
    if camera_img is not None:
        pil_img = Image.open(BytesIO(camera_img.getvalue())).convert("RGB")
        run_and_display(pil_img)
    else:
        st.info("📷 Click the camera button above to capture an image.")
