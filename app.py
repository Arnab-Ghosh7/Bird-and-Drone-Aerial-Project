"""
app.py — Aerial Object Classifier
Bird vs Drone Classification + Optional YOLOv8 Detection

Run locally in VS Code:
    streamlit run app.py

Requires:
    pip install streamlit tensorflow ultralytics pillow matplotlib
"""
import tensorflow as tf
import os, io, tempfile
import numpy as np
import streamlit as st
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bird vs Drone Classifier",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Paths — update these to match where you saved your trained models ───────
#
#   Recommended folder structure next to app.py:
#     models/
#       best_model_savedmodel/    ← TF SavedModel folder (works across all Python/Keras versions)
#       yolov8_best.pt            ← YOLOv8 weights from Colab (optional)
#
_BASE = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH = os.path.join(_BASE, 'models', 'best_model_savedmodel')
YOLO_PATH       = os.path.join(_BASE, 'models', 'yolov8_best.pt')

CLASS_NAMES  = ['bird', 'drone']   # must match training order (class_indices)
IMG_SIZE     = (224, 224)
CONF_THRESH  = 0.5                 # classification decision boundary


# ─── Model Loaders ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading classifier...")
def load_classifier():
    try:
        # Load as TF SavedModel — compatible across all Keras/Python versions
        # Keep the full model object alive (not just .serve) to prevent GC issues
        model = tf.saved_model.load(CLASSIFIER_PATH)
        return model, None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner="Loading YOLOv8...")
def load_yolo():
    try:
        return YOLO(YOLO_PATH), None
    except Exception as e:
        return None, str(e)


# ─── Prediction Helpers ───────────────────────────────────────────────────────
def classify(model, img: Image.Image):
    """Run classification. Returns (label, confidence, raw_proba)."""
    arr   = np.array(img.convert('RGB').resize(IMG_SIZE), dtype='float32') / 255.0
    arr   = np.expand_dims(arr, 0)
    preds = model.serve(tf.constant(arr)).numpy()[0]
    # Support both single sigmoid output and two-neuron softmax output
    if len(preds) == 1:
        proba = float(preds[0])          # sigmoid: probability of drone
    else:
        proba = float(preds[1])          # softmax: preds[1] = drone probability
    label = CLASS_NAMES[1] if proba > CONF_THRESH else CLASS_NAMES[0]
    conf  = proba if label == 'drone' else 1.0 - proba
    return label, conf, proba


def run_detection(yolo_model, img: Image.Image, conf_thresh=0.4):
    """Run YOLOv8 on image, return results object."""
    # Write to temp file after closing it — required on Windows to avoid PermissionError
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    path = tmp.name
    tmp.close()
    img.save(path)
    try:
        res = yolo_model.predict(source=path, imgsz=640, conf=conf_thresh, verbose=False)
    finally:
        os.unlink(path)
    return res[0]


def draw_boxes(img: Image.Image, det_res):
    """Draw bounding boxes on image. Returns (PNG buffer, n_detections)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.axis('off')
    c_map = {'bird': '#4CAF50', 'drone': '#F44336'}
    n = 0

    if det_res.boxes is not None:
        for box in det_res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            name   = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            color  = c_map.get(name, 'yellow')
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3, edgecolor=color, facecolor='none'
            ))
            ax.text(x1+2, y1-6, f'{name}  {conf:.2f}',
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor=color, alpha=0.85))
            n += 1

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf, n


def prob_bar_chart(bird_pct, drone_pct) -> io.BytesIO:
    """Horizontal probability bar chart."""
    fig, ax = plt.subplots(figsize=(5, 2.2))
    ax.barh(['Bird', 'Drone'], [bird_pct, drone_pct],
            color=['#4CAF50', '#F44336'], edgecolor='white')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Probability (%)')
    ax.set_title('Class Probabilities', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for y, val in enumerate([bird_pct, drone_pct]):
        offset = -8 if val > 90 else 1
        ax.text(val + offset, y, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf


# ─── UI ──────────────────────────────────────────────────────────────────────
st.title("🦅 Aerial Object Classifier")
st.markdown(
    "Upload an aerial image to classify it as a **Bird** or **Drone**.  \n"
    "Enable YOLOv8 detection in the sidebar to draw bounding boxes."
)
st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    run_detect  = st.toggle("YOLOv8 Object Detection", value=False,
                            help="Draws bounding boxes around detected objects.")
    det_conf    = st.slider("Detection Confidence", 0.10, 0.90, 0.40, 0.05,
                            help="Minimum confidence to show a bounding box.")
    class_conf  = st.slider("Classification Threshold", 0.30, 0.80, 0.50, 0.05,
                            help="Probability above which image is classified as Drone.")
    CONF_THRESH = class_conf

    st.divider()
    st.markdown("**Model files**")
    st.caption(f"Classifier: `{os.path.basename(CLASSIFIER_PATH)}/`")
    clf_ok = os.path.isdir(CLASSIFIER_PATH)
    st.markdown("✅ Found" if clf_ok else "❌ Not found", unsafe_allow_html=False)

    if run_detect:
        st.caption(f"Detector: `{os.path.basename(YOLO_PATH)}`")
        yolo_ok = os.path.exists(YOLO_PATH)
        st.markdown("✅ Found" if yolo_ok else "❌ Not found")

    st.divider()
    st.markdown("**Dataset info**")
    st.caption("Train: 2,662 images (bird + drone)")
    st.caption("Val: 442 images")
    st.caption("Test: 215 images")

# Load classifier
classifier, clf_err = load_classifier()
if classifier is None:
    st.error(
        f"⚠️ Could not load classifier from `{CLASSIFIER_PATH}`\n\n"
        f"**Error:** {clf_err}\n\n"
        "Train the model in Colab, download the best `.keras` file, and place it at "
        f"`{CLASSIFIER_PATH}`."
    )
    st.stop()

# File uploader
uploaded = st.file_uploader(
    "Upload an image (JPG / PNG)",
    type=['jpg', 'jpeg', 'png'],
    help="Aerial image containing a bird or drone"
)

if uploaded:
    img = Image.open(uploaded).convert('RGB')

    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("Uploaded Image")
        st.image(img, use_container_width=True, caption=uploaded.name)

    with col_result:
        st.subheader("Prediction")

        with st.spinner("Classifying..."):
            label, confidence, raw_proba = classify(classifier, img)

        bird_pct  = (1.0 - raw_proba) * 100
        drone_pct = raw_proba * 100

        # Main result
        icon  = "🦅" if label == 'bird' else "🚁"
        color = "green" if label == 'bird' else "red"
        st.markdown(
            f"<h2 style='color:{color}'>{icon} {label.upper()}</h2>",
            unsafe_allow_html=True
        )
        st.metric(label="Confidence", value=f"{confidence*100:.1f}%")
        st.progress(int(confidence * 100))

        # Probability chart
        st.image(prob_bar_chart(bird_pct, drone_pct), use_container_width=True)

    # Detection section
    if run_detect:
        st.divider()
        st.subheader("🔍 YOLOv8 Detection")

        yolo_model, yolo_err = load_yolo()
        if yolo_model is None:
            st.warning(
                f"YOLOv8 weights not found at `{YOLO_PATH}`\n\n"
                f"Error: {yolo_err}\n\n"
                "Train the detection model in Colab and copy `yolov8_best.pt` here."
            )
        else:
            with st.spinner("Running detection..."):
                det_res   = run_detection(yolo_model, img, det_conf)
                det_buf, n_det = draw_boxes(img, det_res)

            det_col1, det_col2 = st.columns([2, 1])
            with det_col1:
                st.image(det_buf, caption="Detection Output", use_container_width=True)
            with det_col2:
                st.markdown(f"**Detected: {n_det} object(s)**")
                if det_res.boxes is not None and n_det > 0:
                    for i, box in enumerate(det_res.boxes, 1):
                        cls_id = int(box.cls[0])
                        conf   = float(box.conf[0])
                        name   = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                        ico    = "🦅" if name == 'bird' else "🚁"
                        st.write(f"{i}. {ico} **{name}** — {conf*100:.1f}%")
                else:
                    st.info("No objects found above the confidence threshold.")

    # Download button
    st.divider()
    result_text = (
        f"File: {uploaded.name}\n"
        f"Prediction: {label.upper()}\n"
        f"Confidence: {confidence*100:.1f}%\n"
        f"Bird probability: {bird_pct:.1f}%\n"
        f"Drone probability: {drone_pct:.1f}%\n"
    )
    st.download_button(
        label="📥 Download Result",
        data=result_text,
        file_name=f"result_{uploaded.name.rsplit('.', 1)[0]}.txt",
        mime="text/plain"
    )

else:
    # Landing state
    st.info("👆 Upload an image to get started.")
    with st.expander("How to set up"):
        st.markdown("""
        1. Train models in the Colab notebook (`Aerial_Object_Classification.ipynb`)
        2. Download the best model file from your Google Drive (`saved_models/` folder)
        3. Create a `models/` folder next to this `app.py`
        4. Place `best_model.keras` (and optionally `yolov8_best.pt`) inside `models/`
        5. Run: `streamlit run app.py`
        """)
    with st.expander("Dataset info"):
        st.markdown("""
        | Split | Bird | Drone | Total |
        |-------|------|-------|-------|
        | Train | 1,414 | 1,248 | 2,662 |
        | Valid | 217 | 225 | 442 |
        | Test | 121 | 94 | 215 |

        Detection dataset: 3,319 annotated images (YOLOv8 format)
        """)

st.divider()
st.caption("Aerial Object Classification | Deep Learning | TensorFlow + YOLOv8 + Streamlit")
