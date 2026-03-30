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
    with st.spinner("Downloading model… (first load only, ~30 sec)"):
        model_path = hf_hub_download(
            repo_id="YOUR_HF_USERNAME/pneumonia-detector",  # ← change this
            filename="best.onnx"
        )
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )
    return session

# ── Load image (PNG / JPG / DICOM) ───────────────────────────────────────────
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
    return Image.open(uploaded_file).convert('RGB')

# ── Preprocess image for ONNX ─────────────────────────────────────────────────
def preprocess(pil_img, size=640):
    img = pil_img.resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ── Parse YOLOv8 ONNX output ──────────────────────────────────────────────────
def parse_output(output, conf_threshold=0.25, iou_threshold=0.45, img_size=640):
    predictions = np.squeeze(output[0])
    predictions = predictions.T

    boxes  = []
    scores = []

    for pred in predictions:
        x_c, y_c, w, h, conf = pred
        if conf < conf_threshold:
            continue
        x1 = int((x_c - w / 2) * img_size)
        y1 = int((y_c - h / 2) * img_size)
        x2 = int((x_c + w / 2) * img_size)
        y2 = int((y_c + h / 2) * img_size)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_size, x2), min(img_size, y2)
        boxes.append([x1, y1, x2, y2])
        scores.append(float(conf))

    if not boxes:
        return []

    # NMS
    boxes  = np.array(boxes)
    scores = np.array(scores)
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
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
        area_j = (boxes[order[1:],2]-boxes[order[1:],0]) * \
                 (boxes[order[1:],3]-boxes[order[1:],1])
        iou    = inter / (area_i + area_j - inter + 1e-8)
        order  = order[1:][iou < iou_threshold]

    return [{'x1': int(boxes[i,0]), 'y1': int(boxes[i,1]),
             'x2': int(boxes[i,2]), 'y2': int(boxes[i,3]),
             'conf': float(scores[i])} for i in kept]

# ── Draw boxes with PIL ───────────────────────────────────────────────────────
def draw_boxes(pil_img, boxes):
    img  = pil_img.copy()
    draw = ImageDraw.Draw(img)
    for b in boxes:
        for t in range(4):
            draw.rectangle(
                [b['x1']+t, b['y1']+t, b['x2']-t, b['y2']-t],
                outline=(255, 70, 70)
            )
        label = f"Pneumonia {b['conf']:.0%}"
        lw    = len(label) * 8 + 10
        draw.rectangle(
            [b['x1'], b['y1']-28, b['x1']+lw, b['y1']],
            fill=(255, 70, 70)
        )
        draw.text((b['x1']+5, b['y1']-24), label, fill=(255, 255, 255))

    bar_color = (255, 70, 70)  if boxes else (40, 180, 100)
    status    = "PNEUMONIA DETECTED" if boxes else "NORMAL — No Pneumonia"
    draw.rectangle([0, 0, img.width, 44], fill=bar_color)
    draw.text((12, 12), status, fill=(255, 255, 255))
    return img

# ── Run inference ─────────────────────────────────────────────────────────────
def run_detection(session, pil_img, conf, iou):
    img_resized = pil_img.resize((640, 640))
    inp         = preprocess(img_resized)
    input_name  = session.get_inputs()[0].name
    output      = session.run(None, {input_name: inp})
    boxes       = parse_output(output, conf_threshold=conf,
                               iou_threshold=iou, img_size=640)
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
