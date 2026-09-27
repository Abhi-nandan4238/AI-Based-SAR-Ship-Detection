# AI-Based SAR Ship Detection using YOLO11n

An AI-based computer vision system for detecting and localizing ships in Synthetic Aperture Radar (SAR) imagery using YOLO11n.

This project includes dataset preparation, YOLO-based object detection, model evaluation, automated prediction, and a Streamlit web dashboard for interactive ship detection.

---

## 🚀 Project Overview

Synthetic Aperture Radar (SAR) provides Earth observation imagery that can be used for maritime monitoring and large-scale image analysis.

This project develops a deep learning pipeline that automatically detects ships in SAR images using the YOLO11n object detection model.

The system takes a SAR image as input and produces:

* Ship bounding boxes
* Ship count
* Detection confidence
* Bounding-box coordinates
* Processing time
* Visualized detection results

---

## 🎯 Objectives

* Detect ships automatically in SAR imagery.
* Train a custom YOLO11n object detection model.
* Evaluate the model using standard object detection metrics.
* Generate bounding-box predictions on unseen test images.
* Build an interactive Streamlit dashboard.
* Provide a reusable pipeline for SAR-based object detection experiments.

---

## 🛰️ What is SAR?

SAR stands for **Synthetic Aperture Radar**.

It is an active remote sensing technology that uses radar signals to generate images of the Earth's surface.

Unlike optical imagery, SAR does not depend on sunlight and can provide observations during both day and night.

---

## 🤖 Why YOLO11n?

YOLO (You Only Look Once) is an object detection model family designed to detect and localize objects in images.

This project uses YOLO11n because the task requires both:

1. Identifying whether a ship is present.
2. Localizing the ship using a bounding box.

A pretrained YOLO11n model was fine-tuned on the SSDD SAR ship dataset.

---

## 📊 Dataset

### SSDD — SAR Ship Detection Dataset

The project uses the **SSDD dataset** for training and evaluation.

Dataset information used in this project:

* Total images: 1,160
* Training images: 789
* Validation images: 139
* Test images: 232
* Object class: Ship

The model was trained as a single-class object detector:

```text
0 → ship
```

Dataset source:

https://huggingface.co/datasets/dronefreak/SSDD

---

## 🧠 Model

**Model:** YOLO11n

The model was fine-tuned for single-class SAR ship detection.

### Model Configuration

| Parameter         | Value            |
| ----------------- | ---------------- |
| Model             | YOLO11n          |
| Task              | Object Detection |
| Number of Classes | 1                |
| Class             | Ship             |
| Image Size        | 640 × 640        |
| Training Epochs   | 20               |
| Framework         | Ultralytics      |
| Hardware          | CPU              |

---

## 🔄 Project Workflow

```text
SSDD SAR Dataset
       ↓
Dataset Preparation
       ↓
YOLO11n Fine-Tuning
       ↓
Model Validation
       ↓
Test-Set Evaluation
       ↓
Ship Detection
       ↓
Prediction Visualization
       ↓
Streamlit Dashboard
```

---

## 📈 Model Performance

The final evaluation was performed on the **232-image SSDD test set containing 546 ship instances**.

| Metric         | Result |
| -------------- | -----: |
| Precision      |  95.9% |
| Recall         |  91.9% |
| mAP@50         |  97.4% |
| mAP@50-95      |  69.2% |
| Test Images    |    232 |
| Ship Instances |    546 |

### Inference Speed

Average inference time during test evaluation:

```text
82.7 ms per image
```

---

## 🔍 Prediction Results

The trained model was tested on all **232 test images**.

The prediction pipeline generates:

* Bounding boxes around detected ships
* Confidence scores
* Ship count per image
* Annotated prediction images

Prediction outputs are stored under:

```text
runs/detect/outputs/predictions/sar_ship_test
```

---

## 🌐 Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

### Dashboard Features

* SAR image upload
* Confidence threshold control
* Ship detection
* Bounding-box visualization
* Number of detected ships
* Average confidence
* Highest confidence
* Processing time
* Detection details table
* Bounding-box coordinates
* Downloadable detection result

### Dashboard Workflow

```text
Upload SAR Image
        ↓
Select Confidence Threshold
        ↓
Detect Ships
        ↓
YOLO11n Inference
        ↓
Bounding Boxes
        ↓
Detection Statistics
        ↓
Download Result
```

---

## 🖥️ Running the Dashboard

### 1. Clone the repository

```bash
git clone https://github.com/Abhi-nandan4238/AI-Based-SAR-Infrastructure-Detection.git
```

### 2. Enter the project directory

```bash
cd AI-Based-SAR-Infrastructure-Detection
```

### 3. Create virtual environment

```bash
python -m venv .venv
```

### 4. Activate virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Start Streamlit dashboard

```bash
python -m streamlit run app.py
```

---

## 📁 Project Structure

```text
AI-Based-SAR-Infrastructure-Detection
│
├── app/
│
├── data/
│   ├── annotations/
│   ├── processed/
│   │   └── sar/
│   ├── raw/
│   └── samples/
│
├── models/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sar_preprocessing.ipynb
│   ├── 03_dataset_preparation.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── outputs/
│   ├── maps/
│   ├── metrics/
│   └── predictions/
│
├── qgis/
│
├── src/
│   ├── dataset/
│   ├── geospatial/
│   ├── inference/
│   ├── models/
│   ├── preprocessing/
│   └── training/
│
├── runs/
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🛠️ Technologies Used

### Programming

* Python

### Deep Learning

* YOLO11n
* Ultralytics
* PyTorch

### Computer Vision

* OpenCV
* Pillow

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib

### Remote Sensing / Geospatial

* SAR
* Rasterio
* QGIS

### Deployment

* Streamlit

### Development

* VS Code
* Git
* GitHub

---

## 📌 Limitations

The current implementation is a research-oriented prototype focused on SAR ship detection using the SSDD dataset.

Important limitations include:

* The model is trained specifically for the ship class.
* Performance can vary on SAR imagery with different characteristics from the training dataset.
* The current model is not a multi-class SAR object detector.
* The current dashboard is designed for image-based inference rather than large-scale production deployment.
* Real-world Sentinel-1 data may require additional preprocessing and domain adaptation.

---

## 🔮 Future Improvements

Potential improvements include:

* Training on larger and more diverse SAR datasets.
* Multi-class SAR object detection.
* Sentinel-1 GRD preprocessing pipeline.
* SAR-specific data augmentation.
* Geospatial mapping of detected ships.
* QGIS integration.
* Large-scale batch inference.
* Model comparison with other object detection architectures.
* GPU-based inference optimization.
* Cloud deployment of the Streamlit application.

---

## 📚 Key Learning Outcomes

This project demonstrates practical experience with:

* SAR remote sensing concepts
* Object detection
* YOLO model fine-tuning
* Dataset preparation
* Deep learning model evaluation
* Precision and recall analysis
* mAP-based evaluation
* Computer vision inference
* Bounding-box localization
* Streamlit application development
* Python project structuring
* Git/GitHub workflow

---

## 👨‍💻 Author

**Abhinandan Patel**

GitHub:

https://github.com/Abhi-nandan4238

---

## 📄 Project Status

**Completed — Research/Portfolio Prototype**

The project currently includes:

* Dataset preparation
* YOLO11n training
* Test-set evaluation
* Automated ship detection
* Prediction visualization
* Streamlit dashboard
* Project documentation
