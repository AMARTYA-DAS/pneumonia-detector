import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import pydicom
import io, os, tempfile

st.set_page_config(
    page_title="Pneumonia Detector",
    page_icon="🫁",
    layout="wide"
)

st.markdown("""
<style>
.result-positive {
    padding: 1.2rem; border-radius: 10px; text-align: center;
    font-size: 1.3rem; font-weight: 700; margin: 1rem 0;
    background: #ff000015; border: 2px solid #ff4444; color: #ff4444;
}
.result-negative {
    padding: 1.2rem; border-radius: 10px; text-align: center;
    font-size: 1.3rem; font-weight: 700; margin: 1rem 0;
    background: #00ff0015; border: 2px solid #00cc66; color: #00cc66;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best.pt')
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    return YOLO(model_path)

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
        pil_img = Image.fromarray(arr.astype(np.uint8)).convert('RGB')
        os.unlink(tmp_path)
        return pil_img
    return Image.open(uploaded_file).convert('RGB')

# ── Draw bounding boxes using PIL only (NO cv2) ───────────────────────────────
def draw_boxes(pil_img, boxes):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    for b in boxes:
        # Draw thick red rectangle
        for t in range(4):  # thickness of 4px
            draw.rectangle(
                [b['x1']+t, b['y1']+t, b['x2']-t, b['y2']-t],
                outline=(255, 70, 70)
            )
        # Draw label background
        label = f"Pneumonia {b['conf']:.0%}"
        draw.rectangle(
            [b['x1'], b['y1']-28, b['x1']+len(label)*8+10, b['y1']],
            fill=(255, 70, 70)
        )
        draw.text(
            (b['x1']+5, b['y1']-24),
            label,
            fill=(255, 255, 255)
        )
    # Status text at top
    if boxes:
        draw.rectangle([0, 0, img.width, 42], fill=(255, 70, 70))
        draw.text((12, 10), "PNEUMONIA DETECTED", fill=(255, 255, 255))
    else:
        draw.rectangle([0, 0, img.width, 42], fill=(40, 180, 100))
        draw.text((12, 10), "NORMAL - No Pneumonia", fill=(255, 255, 255))
    return img

# ── Run YOLOv8 inference ──────────────────────────────────────────────────────
def run_detection(model, pil_img, conf, iou):
    img_resized = pil_img.resize((640, 640))
    results = model.predict(source=np.array(img_resized),
                            conf=conf, iou=iou, verbose=False)
    result  = results[0]
    boxes   = []
    if result.boxes and len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'conf': float(box.conf[0])
            })
    annotated = draw_boxes(img_resized, boxes)
    return annotated, boxes

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.caption("Upload a chest X-ray (PNG, JPG, or DICOM) to detect pneumonia using AI")

with st.sidebar:
    st.header("⚙️ Settings")
    conf = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05,
                     help="Lower = more sensitive, higher = more precise")
    iou  = st.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05,
                     help="Controls duplicate box suppression")
    st.markdown("---")
    st.info("Supports PNG, JPG, JPEG, and DICOM (.dcm) files")

model = load_model()

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📤 Upload X-Ray")
    uploaded = st.file_uploader(
        "Drag and drop or click to browse",
        type=['png', 'jpg', 'jpeg', 'dcm']
    )
    if uploaded:
        img = load_image(uploaded)
        st.image(img.resize((640, 640)),
                 caption="Original X-Ray",
                 use_container_width=True)
        run_btn = st.button("🔍 Analyze X-Ray", use_container_width=True)
    else:
        st.info("Upload a chest X-ray image to get started")
        run_btn = False

with col2:
    st.subheader("🧠 Detection Result")
    if uploaded and run_btn:
        with st.spinner("Running AI analysis..."):
            img         = load_image(uploaded)
            result_img, boxes = run_detection(model, img, conf, iou)

        st.image(result_img, caption="AI Analysis Result",
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

        # Download button
        buf = io.BytesIO()
        result_img.save(buf, format='PNG')
        st.download_button(
            "⬇️ Download Annotated Result",
            data=buf.getvalue(),
            file_name="pneumonia_result.png",
            mime="image/png",
            use_container_width=True
        )

    elif not uploaded:
        st.markdown("*Upload an image and click Analyze to see results here.*")

st.markdown("---")
st.caption("⚠️ For educational use only. Not a substitute for professional medical diagnosis.")
