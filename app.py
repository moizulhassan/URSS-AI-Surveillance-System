import streamlit as st
import cv2
import tempfile
import os
import time
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Urban Risk Surveillance",
    layout="wide",
    page_icon="🔥"
)

# =========================
# PATHS
# =========================
FIRE_MODEL_PATH = r"C:\Urban-Risk-Surveillance\runs\detect\train5\weights\best.pt"
OBJECT_MODEL_PATH = "yolov8n.pt"
SNAPSHOT_DIR = "snapshots"

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    fire = YOLO(FIRE_MODEL_PATH)
    obj = YOLO(OBJECT_MODEL_PATH)
    return fire, obj

fire_model, object_model = load_models()

# =========================
# UI STYLES
# =========================
st.markdown("""
<style>
.big-title {
    font-size:40px;
    font-weight:800;
    color:#ff4b4b;
}
.card {
    padding:20px;
    border-radius:15px;
    background: linear-gradient(135deg,#1f2933,#111827);
    box-shadow: 0 0 25px rgba(255,75,75,0.35);
    text-align:center;
    font-size:22px;
    font-weight:800;
    color:white;
}
.low {background:#00ff99;color:black;}
.medium {background:#ffaa00;color:black;}
.high {background:#ff4b4b;color:white;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🔥 Urban Risk Surveillance Dashboard</div>", unsafe_allow_html=True)
st.write("AI-based Fire, Crowd & Object Detection System")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Controls")
source_type = st.sidebar.radio("Select Input Source", ["Upload Video", "Live Webcam"])
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.4)
save_snapshots = st.sidebar.checkbox("📸 Auto Save Fire Snapshots", True)

# =========================
# DASHBOARD
# =========================
col1, col2, col3, col4 = st.columns(4)
risk_box = col1.empty()
fire_box = col2.empty()
crowd_box = col3.empty()
other_box = col4.empty()
video_placeholder = st.empty()

# =========================
# RISK LOGIC
# =========================
def calculate_risk(fire, crowd):
    if fire > 0 and crowd > 10:
        return "HIGH RISK", "high"
    elif fire > 0 or crowd > 5:
        return "MEDIUM RISK", "medium"
    return "LOW RISK", "low"

# =========================
# DRAW LABEL (VISIBLE)
# =========================
def draw_label(img, text, x, y):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, y-h-8), (x+w+6, y), (0,0,0), -1)
    cv2.putText(img, text, (x+3, y-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# =========================
# PROCESS FRAME
# =========================
def process_frame(frame):
    fire_res = fire_model(frame, conf=confidence, verbose=False)[0]
    obj_res = object_model(frame, conf=confidence, verbose=False)[0]

    fire_count = crowd_count = other_count = 0

    # Fire
    if fire_res.boxes:
        for box in fire_res.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            fire_count += 1
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            draw_label(frame,"🔥 FIRE",x1,y1)
            if save_snapshots:
                os.makedirs(SNAPSHOT_DIR, exist_ok=True)
                cv2.imwrite(f"{SNAPSHOT_DIR}/fire_{int(time.time())}.jpg", frame)

    # Objects
    if obj_res.boxes:
        for box in obj_res.boxes:
            cls = int(box.cls[0])
            label = object_model.names[cls]
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            conf = box.conf[0]

            if label.lower() == "person":
                crowd_count += 1
                color = (0,255,255)
            else:
                other_count += 1
                color = (0,255,0)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            draw_label(frame,f"{label.upper()} {conf:.2f}",x1,y1)

    return frame, fire_count, crowd_count, other_count

# =========================
# STREAM
# =========================
def stream_video(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, f, c, o = process_frame(frame)
        risk, cls = calculate_risk(f, c)

        risk_box.markdown(f"<div class='card {cls}'>{risk}</div>", unsafe_allow_html=True)
        fire_box.markdown(f"<div class='card'>🔥 Fire: {f}</div>", unsafe_allow_html=True)
        crowd_box.markdown(f"<div class='card'>👥 Crowd: {c}</div>", unsafe_allow_html=True)
        other_box.markdown(f"<div class='card'>📦 Objects: {o}</div>", unsafe_allow_html=True)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame, use_column_width=True)

    cap.release()

# =========================
# MAIN
# =========================
if source_type == "Upload Video":
    file = st.file_uploader("📤 Upload Video", ["mp4","avi","mov"])
    if file:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(file.read())
        stream_video(cv2.VideoCapture(tmp.name))
else:
    cam = cv2.VideoCapture(0)
    if cam.isOpened():
        st.success("🎥 Webcam Running")
        stream_video(cam)
    else:
        st.error("❌ Webcam not accessible")
