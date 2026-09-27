import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import time
from pathlib import Path
import pandas as pd


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI-Based SAR Ship Detection",
    page_icon="🚢",
    layout="wide"
)


# =========================================================
# MODEL PATH
# =========================================================

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


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


model = load_model()


# =========================================================
# HEADER
# =========================================================

st.title("🚢 AI-Based SAR Ship Detection")

st.write(
    "YOLO11n-based ship detection system for "
    "Synthetic Aperture Radar (SAR) imagery."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Detection Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

st.sidebar.write(
    f"Current threshold: **{confidence_threshold:.2f}**"
)

st.sidebar.divider()

st.sidebar.subheader("Model Information")

st.sidebar.write("**Model:** YOLO11n")
st.sidebar.write("**Dataset:** SSDD")
st.sidebar.write("**Class:** Ship")
st.sidebar.write("**Image Size:** 640 × 640")


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload SAR Image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# IMAGE PROCESSING
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("🛰️ Uploaded SAR Image")

    st.image(
        image,
        caption="Original SAR Image",
        use_container_width=True
    )

    st.divider()

    detect_button = st.button(
        "🔍 Detect Ships",
        type="primary",
        use_container_width=True
    )

    if detect_button:

        with st.spinner("Running YOLO ship detection..."):

            start_time = time.time()

            # ---------------------------------------------
            # Temporary image
            # ---------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as temp_file:

                image.save(temp_file.name)
                temp_path = temp_file.name

            # ---------------------------------------------
            # YOLO Prediction
            # ---------------------------------------------

            results = model.predict(
                source=temp_path,
                imgsz=640,
                conf=confidence_threshold,
                verbose=False
            )

            processing_time = time.time() - start_time

            result = results[0]

            # ---------------------------------------------
            # Detection Information
            # ---------------------------------------------

            ship_count = len(result.boxes)

            confidences = []

            if ship_count > 0:
                confidences = (
                    result.boxes.conf
                    .cpu()
                    .tolist()
                )

            if confidences:
                average_confidence = (
                    sum(confidences) / len(confidences)
                )

                highest_confidence = max(confidences)

            else:
                average_confidence = 0.0
                highest_confidence = 0.0

            # ---------------------------------------------
            # Plot Detection Result
            # ---------------------------------------------

            detected_image = result.plot()


        # =================================================
        # DETECTION RESULTS
        # =================================================

        st.subheader("🎯 Detection Results")

        # ---------------------------------------------
        # Metrics
        # ---------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🚢 Ships Detected",
                ship_count
            )

        with col2:
            st.metric(
                "🎯 Average Confidence",
                f"{average_confidence:.2%}"
            )

        with col3:
            st.metric(
                "🏆 Highest Confidence",
                f"{highest_confidence:.2%}"
            )

        with col4:
            st.metric(
                "⚡ Processing Time",
                f"{processing_time:.2f}s"
            )


        st.divider()


        # =================================================
        # DETECTED IMAGE
        # =================================================

        st.subheader("🛰️ Detected Ships")

        st.image(
            detected_image,
            caption="YOLO11n Ship Detection",
            use_container_width=True
        )


        # =================================================
        # DOWNLOAD RESULT
        # =================================================

        st.divider()

        st.subheader("⬇️ Download Result")

        # Convert detected image to PNG bytes

        from io import BytesIO

        output_image = Image.fromarray(detected_image)

        buffer = BytesIO()

        output_image.save(
            buffer,
            format="PNG"
        )

        st.download_button(
            label="⬇️ Download Detected Image",
            data=buffer.getvalue(),
            file_name="sar_ship_detection_result.png",
            mime="image/png",
            use_container_width=True
        )


        # =================================================
        # DETECTION DETAILS
        # =================================================

        st.divider()

        st.subheader("📋 Detection Details")


        if ship_count > 0:

            detection_data = []

            for i, box in enumerate(result.boxes):

                confidence = float(
                    box.conf[0].cpu().item()
                )

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .tolist()
                )

                x1, y1, x2, y2 = coordinates

                detection_data.append(
                    {
                        "Ship": f"Ship {i + 1}",
                        "Confidence": f"{confidence:.2%}",
                        "X1": round(x1, 1),
                        "Y1": round(y1, 1),
                        "X2": round(x2, 1),
                        "Y2": round(y2, 1)
                    }
                )


            detection_df = pd.DataFrame(
                detection_data
            )

            st.dataframe(
                detection_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No ships detected above the selected "
                "confidence threshold."
            )


        # =================================================
        # SUMMARY
        # =================================================

        st.divider()

        st.subheader("📊 Detection Summary")

        if ship_count > 0:

            st.success(
                f"{ship_count} ship(s) detected "
                f"with an average confidence of "
                f"{average_confidence:.2%}."
            )

        else:

            st.warning(
                "No ships were detected. "
                "Try lowering the confidence threshold."
            )


else:

    st.info(
        "👆 Upload a SAR image above to start ship detection."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI-Based SAR Ship Detection | "
    "YOLO11n | SSDD Dataset | "
    "SAR Remote Sensing"
)