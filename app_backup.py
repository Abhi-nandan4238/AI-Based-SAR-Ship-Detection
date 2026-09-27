import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import time
from pathlib import Path


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="AI-Based SAR Ship Detection",
    page_icon="🚢",
    layout="wide"
)


# -------------------------------------------------
# Model Path
# -------------------------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "runs"
    / "detect"
    / "outputs"
    / "training"
    / "sar_ship_yolo-3"
    / "weights"
    / "best.pt"
)


# -------------------------------------------------
# Load Model
# -------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


model = load_model()


# -------------------------------------------------
# Title
# -------------------------------------------------

st.title("🚢 AI-Based SAR Ship Detection")

st.write(
    "YOLO11n-based ship detection system for Synthetic Aperture Radar (SAR) imagery."
)


st.divider()


# -------------------------------------------------
# Upload Image
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload SAR Image",
    type=["jpg", "jpeg", "png"]
)


# -------------------------------------------------
# Detection
# -------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded SAR Image")

    st.image(
        image,
        caption="Original SAR Image",
        use_container_width=True
    )

    if st.button("🔍 Detect Ships", type="primary"):

        with st.spinner("Detecting ships..."):

            start_time = time.time()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as temp_file:

                image.save(temp_file.name)
                temp_path = temp_file.name

            results = model.predict(
                source=temp_path,
                imgsz=640,
                conf=0.25,
                verbose=False
            )

            processing_time = time.time() - start_time

            result = results[0]

            detected_image = result.plot()

            ship_count = len(result.boxes)

            if ship_count > 0:
                confidences = result.boxes.conf.cpu().tolist()
                average_confidence = sum(confidences) / len(confidences)
            else:
                average_confidence = 0.0


        # -------------------------------------------------
        # Results
        # -------------------------------------------------

        st.divider()

        st.subheader("Detection Results")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                detected_image,
                caption="Detected Ships",
                use_container_width=True
            )

        with col2:

            st.metric(
                "Ships Detected",
                ship_count
            )

            st.metric(
                "Average Confidence",
                f"{average_confidence:.2%}"
            )

            st.metric(
                "Processing Time",
                f"{processing_time:.2f} seconds"
            )

            st.write("**Model:** YOLO11n")

            st.write("**Dataset:** SSDD")

            st.write("**Detection Class:** Ship")


else:

    st.info(
        "Upload a SAR image to start ship detection."
    )


# -------------------------------------------------
# Footer
# -------------------------------------------------

st.divider()

st.caption(
    "AI-Based SAR Ship Detection | YOLO11n | SSDD Dataset"
)