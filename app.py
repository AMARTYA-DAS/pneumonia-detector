
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import pydicom
import io, os, tempfile

st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁", layout="wide")

st.markdown("""
<style>
.result-box { padding:1.2rem; border-radius:10px; text-align:center;
              font-size:1.3rem; font-weight:700; margin:1rem 0; }
.positive { background:#ff000015; border:2px solid #ff4444; color:#ff4444; }
.negative { background:#00ff0015; border:2px solid #00cc66; color:#00cc66; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not os.path.exists('best.pt'):
        st.error("Model file 'best.pt' not found in the app directory!")
        st.stop()
    return YOLO('best.pt')

def load_image(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith('.dcm'):
        with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        dcm = pydicom.dcmread(tmp_path)
        arr = dcm.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        pil_img = Image.fromarray(arr.astype(np.uint8)).convert('RGB')
        os.unlink(tmp_path)
        return pil_img
    return Image.open(uploaded_file).convert('RGB')

def run_detection(model, pil_img, conf, iou):
    img = np.array(pil_img.resize((640, 640)))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = model.predict(bgr, conf=conf, iou=iou, verbose=False)
    result  = results[0]
    out     = bgr.copy()
    boxes   = []

    if result.boxes and len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            c = float(box.conf[0])
            boxes.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'conf':c})
            cv2.rectangle(out, (x1,y1), (x2,y2), (50,80,255), 3)
            label = f"Pneumonia {c:.0%}"
            cv2.putText(out, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,80,255), 2)
        cv2.putText(out, "PNEUMONIA DETECTED", (15,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,80,255), 3)
    else:
        cv2.putText(out, "NORMAL", (15,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,200,80), 3)

    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), boxes

# ── UI ──────────────────────────────────────────────────────────────────────
st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.caption("Upload a chest X-ray (PNG, JPG, or DICOM) to detect pneumonia using AI")

with st.sidebar:
    st.header("⚙️ Settings")
    conf  = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)
    iou   = st.slider("IoU threshold",        0.10, 0.90, 0.45, 0.05)
    st.info("Lower confidence = more sensitive (catches more cases, more false positives)")

model = load_model()

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📤 Upload X-Ray")
    file = st.file_uploader("", type=['png','jpg','jpeg','dcm'])
    if file:
        img = load_image(file)
        st.image(img.resize((640,640)), caption="Original X-Ray", use_container_width=True)
        run_btn = st.button("🔍 Analyze X-Ray", use_container_width=True)
    else:
        st.info("Upload an X-ray image to get started")
        run_btn = False

with col2:
    st.subheader("🧠 Detection Result")
    if file and run_btn:
        with st.spinner("Analyzing..."):
            img      = load_image(file)
            out_img, boxes = run_detection(model, img, conf, iou)

        st.image(out_img, caption="AI Analysis", use_container_width=True)

        if boxes:
            st.markdown('<div class="result-box positive">⚠️ PNEUMONIA DETECTED</div>',
                        unsafe_allow_html=True)
            st.write(f"**{len(boxes)} region(s) detected**")
            for i, b in enumerate(boxes, 1):
                st.write(f"Region {i}: confidence {b['conf']:.1%} | "
                         f"position ({b['x1']},{b['y1']}) | "
                         f"size {b['x2']-b['x1']}×{b['y2']-b['y1']}px")
        else:
            st.markdown('<div class="result-box negative">✅ NORMAL — No Pneumonia Found</div>',
                        unsafe_allow_html=True)

        buf = io.BytesIO()
        out_img.save(buf, format='PNG')
        st.download_button("⬇️ Download Result", buf.getvalue(),
                           file_name="result.png", mime="image/png",
                           use_container_width=True)
    elif not file:
        st.markdown("*Upload an image and click Analyze to see results here.*")

st.markdown("---")
st.caption("⚠️ For educational use only. Not a substitute for professional medical diagnosis.")
