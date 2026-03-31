import os
import io
import tempfile

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import pydicom
import onnxruntime as ort
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Pneumonia Detector",
    page_icon="🫁",
    layout="wide"
)

st.markdown("""
<style>
.result-positive {
    padding:1.2rem; border-radius:10px; text-align:center;
    font-size:1.3rem; font-weight:700; margin:1rem 0;
    background:#ff000015; border:2px solid #ff4444; color:#ff4444;
}
.result-negative {
    padding:1.2rem; border-radius:10px; text-align:center;
    font-size:1.3rem; font-weight:700; margin:1rem 0;
    background:#00ff0015; border:2px solid #00cc66; color:#00cc66;
}
</style>
""", unsafe_allow_html=True)

# ── Load ONNX model from Hugging Face ────────────────────────────────────────
@st.cache_resource
def load_model():
    with st.spinner("Downloading model… (first load only)"):
        model_path = hf_hub_download(
            repo_id="AMARTYA1005/pneumonia-detector",
            filename="best.onnx"
        )
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )
    return session

# ── Load image ────────────────────────────────────────────────────────────────
def load_image(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith('.dcm'):
        with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        dcm = pydicom.dcmread(tmp_path)
        arr = dcm.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        img = Image.fromarray(arr.astype(np.uint8)).convert('RGB')
        os.unlink(tmp_path)
        return img
    img = Image.open(uploaded_file)
    img = img.convert('RGB')   # handles RGBA, grayscale, palette images
    return img

# ── Preprocess ────────────────────────────────────────────────────────────────
def preprocess(pil_img, size=640):
    img = pil_img.resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ── NMS ───────────────────────────────────────────────────────────────────────
def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    boxes  = np.array(boxes,  dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    order  = scores.argsort()[::-1]
    kept   = []
    while order.size > 0:
        i = order[0]
        kept.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[order[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[order[1:],3])
        inter = (np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1))
        area_i = (boxes[i,2]-boxes[i,0])   * (boxes[i,3]-boxes[i,1])
        area_j = (boxes[order[1:],2]-boxes[order[1:],0]) * \
                 (boxes[order[1:],3]-boxes[order[1:],1])
        iou   = inter / (area_i + area_j - inter + 1e-8)
        order = order[1:][iou < iou_threshold]
    return kept

# ── Parse ONNX output ─────────────────────────────────────────────────────────
def parse_output(output, conf_thr=0.25, iou_thr=0.45, img_size=640):
    preds = np.squeeze(output[0]).T   # [8400, 5]
    boxes, scores = [], []
    for pred in preds:
        xc, yc, w, h, conf = float(pred[0]), float(pred[1]), \
                              float(pred[2]), float(pred[3]), float(pred[4])
        if conf < conf_thr:
            continue
        x1 = max(0, int((xc - w/2) * img_size))
        y1 = max(0, int((yc - h/2) * img_size))
        x2 = min(img_size, int((xc + w/2) * img_size))
        y2 = min(img_size, int((yc + h/2) * img_size))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
    kept = nms(boxes, scores, iou_thr)
    return [{'x1': boxes[i][0], 'y1': boxes[i][1],
             'x2': boxes[i][2], 'y2': boxes[i][3],
             'conf': scores[i]} for i in kept]

# ── Draw boxes ────────────────────────────────────────────────────────────────
def draw_boxes(pil_img, boxes):
    img  = pil_img.copy().convert('RGB')
    draw = ImageDraw.Draw(img)
    W, H = img.size

    for b in boxes:
        x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']

        # Draw border lines manually — avoids ALL rectangle() version issues
        lw = 4  # line width
        # Top line
        draw.rectangle([x1, y1, x2, y1+lw], fill=(255, 70, 70))
        # Bottom line
        draw.rectangle([x1, y2-lw, x2, y2], fill=(255, 70, 70))
        # Left line
        draw.rectangle([x1, y1, x1+lw, y2], fill=(255, 70, 70))
        # Right line
        draw.rectangle([x2-lw, y1, x2, y2], fill=(255, 70, 70))

        # Label
        label   = f"Pneumonia {b['conf']:.0%}"
        label_w = len(label) * 8 + 10
        label_y = max(0, y1 - 28)
        draw.rectangle(
            [x1, label_y, x1 + label_w, label_y + 26],
            fill=(255, 70, 70)
        )
        draw.text((x1 + 5, label_y + 5), label, fill=(255, 255, 255))

    # Status bar
    bar_color = (255, 70, 70) if boxes else (40, 180, 100)
    status    = "PNEUMONIA DETECTED" if boxes else "NORMAL — No Pneumonia"
    draw.rectangle([0, 0, W, 44], fill=bar_color)
    draw.text((12, 12), status, fill=(255, 255, 255))

    return img

# ── Run detection ─────────────────────────────────────────────────────────────
def run_detection(session, pil_img, conf, iou):
    img_resized = pil_img.resize((640, 640))
    inp         = preprocess(img_resized)
    input_name  = session.get_inputs()[0].name
    raw_output  = session.run(None, {input_name: inp})
    boxes       = parse_output(raw_output, conf_thr=conf,
                               iou_thr=iou, img_size=640)
    return draw_boxes(img_resized, boxes), boxes

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.caption("Upload a chest X-ray (PNG, JPG, or DICOM) to detect pneumonia using AI")

with st.sidebar:
    st.header("⚙️ Settings")
    conf = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)
    iou  = st.slider("IoU threshold",        0.10, 0.90, 0.45, 0.05)
    st.markdown("---")
    st.info("Supported formats: PNG · JPG · JPEG · DICOM (.dcm)")

session = load_model()

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📤 Upload X-Ray")
    uploaded = st.file_uploader(
        "Drag and drop or click to browse",
        type=['png', 'jpg', 'jpeg', 'dcm']
    )
    if uploaded:
        orig = load_image(uploaded)
        st.image(orig.resize((640, 640)),
                 caption="Original X-Ray",
                 use_container_width=True)
        run_btn = st.button("🔍  Analyze X-Ray", use_container_width=True)
    else:
        st.info("Upload a chest X-ray image to get started.")
        run_btn = False

with col2:
    st.subheader("🧠 Detection Result")
    if uploaded and run_btn:
        with st.spinner("Running AI analysis…"):
            orig = load_image(uploaded)
            result_img, boxes = run_detection(session, orig, conf, iou)

        st.image(result_img,
                 caption="AI Analysis Result",
                 use_container_width=True)

        if boxes:
            st.markdown(
                '<div class="result-positive">⚠️ PNEUMONIA DETECTED</div>',
                unsafe_allow_html=True
            )
            st.write(f"**{len(boxes)} region(s) detected**")
            for i, b in enumerate(boxes, 1):
                st.write(
                    f"Region {i}: confidence **{b['conf']:.1%}** | "
                    f"position ({b['x1']}, {b['y1']}) | "
                    f"size {b['x2']-b['x1']} × {b['y2']-b['y1']} px"
                )
        else:
            st.markdown(
                '<div class="result-negative">✅ NORMAL — No Pneumonia Found</div>',
                unsafe_allow_html=True
            )

        buf = io.BytesIO()
        result_img.save(buf, format='PNG')
        st.download_button(
            "⬇️  Download Annotated Result",
            data=buf.getvalue(),
            file_name="pneumonia_result.png",
            mime="image/png",
            use_container_width=True
        )
    elif not uploaded:
        st.markdown("*Upload an image and click Analyze to see results.*")

st.markdown("---")
st.caption("⚠️ For educational use only. Not a substitute for professional medical diagnosis.")
