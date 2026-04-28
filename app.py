##############################################################
# Coded By: Lew Yun Cheng
# Streamlit demo for SSPD (Single-Shot Plate Detector)
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
st.title("🚗 License Plate Detection — SSPD Demo")
st.caption("UCCD3094 Group 6 — Single-Shot Plate Detector (custom CNN + pretrained ResNet-18 backbone)")

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
grid_size = cfg['grid_size']
st.sidebar.markdown(f"**Device:** `{device.type.upper()}`")
st.sidebar.markdown(f"**Input size:** `{INPUT_SIZE}×{INPUT_SIZE}`")
st.sidebar.markdown(f"**Grid size:** `{grid_size}×{grid_size}`")


# ---------- Helper: draw boxes on image ----------
def draw_detections(pil_img, boxes_normalized):
    """Draw boxes given (cx,cy,w,h,conf) in [0,1] on a PIL image."""
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(14, W // 40))
    except Exception:
        font = ImageFont.load_default()
    for cx, cy, w, h, conf in boxes_normalized:
        x1 = (cx - w/2) * W; y1 = (cy - h/2) * H
        x2 = (cx + w/2) * W; y2 = (cy + h/2) * H
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        label = f"plate {conf:.2f}"
        tw = draw.textlength(label, font=font)
        th = font.size + 4
        draw.rectangle([x1, max(0, y1 - th), x1 + tw + 6, y1], fill="red")
        draw.text((x1 + 3, max(0, y1 - th)), label, fill="white", font=font)
    return img


# ---------- Helper: run detection + render results ----------
def run_and_display(pil_img, source_key):
    """Run the detector on a PIL image and render the full results UI.

    source_key (str): unique identifier for the input source ('upload' or 'camera').
    Used to make widget IDs unique so the same widget can appear in both tabs.
    """
    # Preprocess + inference
    x = transform(pil_img).unsqueeze(0).to(device)
    t0 = time.time()
    decoded = model.decode(x, conf_threshold=conf_threshold,
                           iou_threshold=iou_threshold, max_boxes=int(max_boxes))[0]
    if device.type == "cuda": torch.cuda.synchronize()
    latency_ms = (time.time() - t0) * 1000

    boxes_np = decoded.cpu().numpy() if decoded.numel() else np.zeros((0, 5))

    # Side-by-side view
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input image")
        st.image(pil_img, use_container_width=True)
    with col2:
        st.subheader("Detections")
        annotated = draw_detections(pil_img, boxes_np)
        st.image(annotated, use_container_width=True)

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Plates detected", len(boxes_np))
    m2.metric("Top confidence",
              f"{float(boxes_np[:, 4].max()):.3f}" if len(boxes_np) else "—")
    m3.metric("Inference latency", f"{latency_ms:.1f} ms")

    # Detections table
    if len(boxes_np):
        st.subheader("Bounding boxes (normalized coordinates)")
        df = pd.DataFrame(boxes_np, columns=["cx", "cy", "w", "h", "confidence"])
        df.insert(0, "#", np.arange(1, len(df) + 1))
        st.dataframe(df.style.format({"cx": "{:.3f}", "cy": "{:.3f}",
                                      "w": "{:.3f}", "h": "{:.3f}",
                                      "confidence": "{:.3f}"}),
                     use_container_width=True)
    else:
        st.warning("No plates detected above the confidence threshold. "
                   "Try lowering it in the sidebar.")

    # Download button for the annotated image (unique key per source)
    buf = BytesIO()
    annotated.save(buf, format="PNG")
    st.download_button(
        label="⬇️  Download annotated image",
        data=buf.getvalue(),
        file_name="detection_result.png",
        mime="image/png",
        key=f"download_{source_key}",
    )


# ---------- Input source selection via tabs ----------
tab_upload, tab_camera = st.tabs(["📁 Upload image", "📷 Capture with camera"])

with tab_upload:
    uploaded = st.file_uploader("Upload a car image (JPG / PNG)",
                                type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        pil_img = Image.open(BytesIO(uploaded.read())).convert("RGB")
        run_and_display(pil_img, source_key="upload")
    else:
        st.info("👆 Upload an image to get started.")
        st.markdown(
            "**Tip:** lower the confidence threshold in the sidebar if the detector "
            "misses the plate, or raise it to suppress spurious boxes."
        )

with tab_camera:
    st.markdown(
        "Use your device camera to capture a live photo of a car. "
        "On first use, your browser will ask for camera permission — click **Allow**."
    )
    camera_img = st.camera_input("Take a photo")
    if camera_img is not None:
        pil_img = Image.open(BytesIO(camera_img.getvalue())).convert("RGB")
        run_and_display(pil_img, source_key="camera")
    else:
        st.info("📷 Click the camera button above to capture an image.")