# SmartVoice AI Vision Stick

An advanced computer vision framework for vision assistance, featuring object detection (COCO-90) and real-time OCR for text and sign reading.

## Features
- **Real-time Object Detection**: Uses TFLite/LiteRT for high-performance detection across 90 object classes.
- **OCR Assistant**: Processes text and signs in the live feed using EasyOCR.
- **Web Dashboard**: Glassmorphic UI with live video stream, AI terminal logs, and system health status.
- **Responsive Design**: Works on Desktop, Tablets, and Mobile.

## Hardware Support
- **Processor**: Optimized for Raspberry Pi 4/5 (also compatible with Windows/Linux).
- **Camera**: Support for Raspberry Pi Camera Module V2 and USB Webcams.
- **Backend**: Uses OpenCV with DirectShow (Windows) and GStreamer (Linux) for high-frame-rate streaming.

## Setup Instructions

### 1. Requirements
- Python 3.10 - 3.13
- Webcam or USB Camera

### 2. Installation
Create a virtual environment and install the dependencies:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Running the App
Run the Flask server:
```bash
python app.py
```
Then open your browser at `http://127.0.0.1:5000`.

## Directory Structure
- `app.py`: Main Flask server and AI inference logic.
- `templates/`: HTML/CSS for the dashboard.
- `tflite_model/`: Contains detected.tflite and labelmap.
- `requirements.txt`: Python package list.
