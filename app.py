import os
import time
import threading
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify

try:
    import easyocr
    OCR_OK = True
except ImportError:
    OCR_OK = False

# --- AI Configuration & Model Setup ---
# Attempt to import TFLite
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_OK = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_OK = True
    except ImportError:
        try:
            import ai_edge_litert.interpreter as tflite
            TFLITE_OK = True
        except ImportError:
            TFLITE_OK = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "tflite_model", "detect.tflite")
LABELS_PATH = os.path.join(SCRIPT_DIR, "tflite_model", "labelmap.txt")

# Default detection thresholds (simplified for dashboard)
CONF_THRESHOLD = 0.50

class SmartDetector:
    def __init__(self, model_path, labels_path):
        self.labels = self.load_labels(labels_path)
        if TFLITE_OK and os.path.exists(model_path):
            self.interp = tflite.Interpreter(model_path=model_path)
            self.interp.allocate_tensors()
            inp = self.interp.get_input_details()[0]
            out = self.interp.get_output_details()
            self.input_idx = inp["index"]
            self.input_h = inp["shape"][1]
            self.input_w = inp["shape"][2]
            self.float_input = (inp["dtype"] == np.float32)
            self.out_boxes = out[0]["index"]
            self.out_cls = out[1]["index"]
            self.out_scores = out[2]["index"]
            self.out_count = out[3]["index"]
            self.active = True
            print(f"[AI] Model loaded: {self.input_w}x{self.input_h}")
        else:
            self.active = False
            print("[AI] Running in Dummy mode (No TFLite or Model found)")

    def load_labels(self, path):
        if not os.path.exists(path): return ["Default"]
        with open(path) as f:
            lines = [l.strip() for l in f.readlines()]
            return [l.split(maxsplit=1)[-1] if l[0].isdigit() else l for l in lines]

    def detect(self, frame):
        if not self.active:
            # Simulated detection for demo if no model
            return [("No Model Loaded", 0.99, 0.1, 0.1, 0.2, 0.2)]
        
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        if self.float_input: img = (img - 127.5) / 127.5

        self.interp.set_tensor(self.input_idx, img)
        self.interp.invoke()

        boxes = self.interp.get_tensor(self.out_boxes)[0]
        clsids = self.interp.get_tensor(self.out_cls)[0]
        scores = self.interp.get_tensor(self.out_scores)[0]
        count = int(self.interp.get_tensor(self.out_count)[0])

        results = []
        for i in range(min(count, len(scores))):
            score = float(scores[i])
            if score < CONF_THRESHOLD: continue
            cid = int(clsids[i]) + 1
            if cid < len(self.labels):
                label = self.labels[cid]
                y1, x1, y2, x2 = boxes[i]
                results.append((label, score, float(x1), float(y1), float(x2), float(y2)))
        return results

# --- Flask Server Logic ---
app = Flask(__name__)
detector = SmartDetector(MODEL_PATH, LABELS_PATH)
# Initialize camera with DirectShow for better Windows support
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
current_detections = []
ocr_detections = []
latest_frame = None
lock = threading.Lock()

def ocr_worker():
    global ocr_detections, latest_frame
    if not OCR_OK:
        return
    print("[OCR] Initializing EasyOCR reader...")
    reader = easyocr.Reader(['en'], gpu=False)
    print("[OCR] Reader Ready.")
    while True:
        frame_to_process = None
        with lock:
            if latest_frame is not None:
                frame_to_process = latest_frame.copy()
        
        if frame_to_process is not None:
            # We convert to RGB for OCR (or just let EasyOCR handle it, but BGR to RGB is safer if passing numpy array)
            rgb_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
            try:
                results = reader.readtext(rgb_frame)
                new_ocr_dets = []
                for (bbox, text, prob) in results:
                    if prob > 0.4:
                        # bounding box is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        xs = [p[0] for p in bbox]
                        ys = [p[1] for p in bbox]
                        h, w, _ = frame_to_process.shape
                        new_ocr_dets.append((f"TEXT: {text}", prob, min(xs)/w, min(ys)/h, max(xs)/w, max(ys)/h))
                
                with lock:
                    ocr_detections = new_ocr_dets
            except Exception as e:
                print(f"[OCR] Error: {e}")
        time.sleep(1.0) # wait before next read so we don't hog CPU

if OCR_OK:
    threading.Thread(target=ocr_worker, daemon=True).start()

def video_stream_gen():
    global current_detections, latest_frame
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Run AI every few frames to save CPU
        dets = detector.detect(frame)
        with lock:
            latest_frame = frame
            current_detections = dets + ocr_detections

        # Draw detections on frame for visual feed
        for label, score, x1, y1, x2, y2 in dets:
            h, w, _ = frame.shape
            cv2.rectangle(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {int(score*100)}%", (int(x1*w), int(y1*h)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def detections():
    with lock:
        return jsonify([{"label": d[0], "conf": round(d[1], 2), "time": time.strftime("%H:%M:%S")} for d in current_detections])

@app.route('/status')
def status():
    return jsonify({
        "camera": camera.isOpened(),
        "ai": detector.active,
        "model": "Phase 5 COCO-90",
        "ocr": OCR_OK,
        "fps": 15
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
