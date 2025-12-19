import streamlit as st
import cv2
import numpy as np
import time
import os
from detector import FaceDetector

# ------------------ Page Config ------------------
st.set_page_config(page_title="Face Detection App", layout="centered")

st.title("🧠 Real-Time Face Detection System")
st.write("Detect faces using a Deep Learning model (SSD + ResNet).")

# ------------------ Sidebar ------------------
st.sidebar.header("Settings")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Image Upload", "Webcam"]
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

# ------------------ Model Paths (SAFE) ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel"
)

CONFIG_PATH = os.path.join(
    BASE_DIR, "models", "deploy.prototxt.txt"
)

# ------------------ Load Model ------------------
@st.cache_resource
def load_detector(conf_thresh):
    return FaceDetector(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        confidence_threshold=conf_thresh
    )

detector = load_detector(confidence_threshold)

# ------------------ IMAGE MODE ------------------
if mode == "Image Upload":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        faces = detector.detect_faces(image)

        for face in faces:
            (x1, y1, x2, y2) = face["box"]
            conf = face["confidence"]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        st.success(f"Faces detected: {len(faces)}")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")

# ------------------ WEBCAM MODE ------------------
if mode == "Webcam":
    st.warning("Press STOP to turn off webcam")

    cap = cv2.VideoCapture(0)
    frame_window = st.image([])
    stop = st.button("STOP")

    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break

        start_time = time.time()
        faces = detector.detect_faces(frame)
        fps = 1 / (time.time() - start_time)

        for face in faces:
            (x1, y1, x2, y2) = face["box"]
            conf = face["confidence"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"FPS: {int(fps)} | Faces: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        frame_window.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB"
        )

    cap.release()

# ------------------ Footer ------------------
st.markdown("---")
st.caption("Built with OpenCV DNN & Streamlit • Face Detection System")
