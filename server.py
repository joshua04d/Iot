#!/usr/bin/env python3
import os
import time
import threading
import platform
import requests

import cv2
from flask import Flask, Response, jsonify

from ultralytics import YOLO

# Optional fast JPEG encoder
try:
    from turbojpeg import TurboJPEG, TJPF_BGR, TJSAMP_420, TJFLAG_FASTDCT
    jpeg = TurboJPEG()
except Exception:
    jpeg = None

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "fire_smoke_yolo11s_20epochs.pt"   # your trained model
CAM_INDEX = 1                                   # USB2.0 PC CAMERA index (adjust if needed)
BACKEND = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY

IMG_MAX_WIDTH = 960
CONF = 0.25
TARGET_INF_FPS = 30
STREAM_FPS = 30
JPEG_QUALITY = 70
FRAME_SKIP = 2
USE_CUDA = os.environ.get("USE_CUDA", "1") == "1"

# ThingSpeak settings
CHANNEL_ID = "3131292"
FIELDS = [1, 2, 3]
TS_URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/fields/{{}}.json?results=1"

# ----------------------------
# Init
# ----------------------------
app = Flask(__name__)

print(f"[INFO] Loading YOLO model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

import torch
if not torch.cuda.is_available():
    raise RuntimeError("[ERROR] CUDA not available! This script requires a GPU.")
model.to("cuda")
GPU = True
print("[INFO] Using GPU (CUDA)")

try:
    model.fuse()
except Exception:
    pass

# ----------------------------
# Camera setup
# ----------------------------
cap = cv2.VideoCapture(CAM_INDEX, BACKEND)
if not cap.isOpened():
    raise RuntimeError(f"[ERROR] Could not open camera index {CAM_INDEX}")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Shared state
lock = threading.Lock()
latest_raw = None
latest_annot = None
stop_flag = False

# ----------------------------
# Helper functions
# ----------------------------
def resize_if_needed(frame):
    if IMG_MAX_WIDTH and frame.shape[1] > IMG_MAX_WIDTH:
        scale = IMG_MAX_WIDTH / frame.shape[1]
        frame = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    return frame

def encode_jpeg(img_bgr):
    if jpeg:
        return jpeg.encode(img_bgr, quality=JPEG_QUALITY)
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buf.tobytes() if ok else None

def mjpeg_generator(get_frame_fn):
    frame_interval = 1.0 / max(STREAM_FPS, 1)
    while True:
        frame = get_frame_fn()
        if frame is None:
            time.sleep(0.005)
            continue
        jpg = encode_jpeg(frame)
        if jpg is None:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
            jpg + b"\r\n"
        )
        time.sleep(frame_interval)

# ThingSpeak helper
def get_latest_field(field_num):
    try:
        r = requests.get(TS_URL.format(field_num), timeout=2).json()
        value = r['feeds'][0].get(f'field{field_num}')
        return float(value) if value else 0
    except:
        return 0

# ----------------------------
# Threads
# ----------------------------
def capture_loop():
    global latest_raw
    while not stop_flag:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.005)
            continue
        with lock:
            latest_raw = frame

def yolo_loop():
    global latest_annot
    prev_t = time.time()
    counter = 0
    min_interval = 1.0 / max(TARGET_INF_FPS, 1)

    while not stop_flag:
        now = time.time()
        sleep_left = min_interval - (now - prev_t)
        if sleep_left > 0:
            time.sleep(sleep_left)
        prev_t = time.time()

        with lock:
            frame = None if latest_raw is None else latest_raw.copy()
        if frame is None:
            time.sleep(0.005)
            continue

        counter += 1
        if counter % FRAME_SKIP != 0:
            continue

        t0 = time.time()
        frame = resize_if_needed(frame)

        results = model.predict(
            frame,
            conf=CONF,
            imgsz=IMG_MAX_WIDTH if IMG_MAX_WIDTH else None,
            half=True if GPU else False,
            verbose=False,
            max_det=50,
        )
        annotated = results[0].plot()

        inf_fps = 1.0 / max(time.time() - t0, 1e-6)
        cv2.putText(
            annotated, f"INF FPS: {inf_fps:.1f}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )

        with lock:
            latest_annot = annotated

def start_threads():
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=yolo_loop, daemon=True).start()

# ----------------------------
# Flask routes
# ----------------------------
@app.route("/")
def index():
    return f"""
<html>
<head>
<title>Fire/Smoke + Sensor Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body style='font-family:sans-serif'>
<h1>Fire/Smoke Detection</h1>
<p><a href='/video_feed'>Annotated Output</a> | <a href='/video_feed_raw'>Raw Input</a></p>
<img src='/video_feed' style='width:100%;max-width:1000px;border-radius:8px'/>

<h2>Live Sensor Data from ThingSpeak</h2>

<div style="display:flex;gap:20px;flex-wrap:wrap;">
  <div><canvas id="chart1" width="400" height="200"></canvas></div>
  <div><canvas id="chart2" width="400" height="200"></canvas></div>
  <div><canvas id="chart3" width="400" height="200"></canvas></div>
</div>

<script>
function createChart(ctx, label, color) {{
    return new Chart(ctx, {{
        type: 'line',
        data: {{ labels: [], datasets: [{{ label: label, data: [], borderColor: color, fill: false }}] }},
        options: {{ animation: false }}
    }});
}}

const chart1 = createChart(document.getElementById('chart1').getContext('2d'), 'Field 1', 'red');
const chart2 = createChart(document.getElementById('chart2').getContext('2d'), 'Field 2', 'blue');
const chart3 = createChart(document.getElementById('chart3').getContext('2d'), 'Field 3', 'green');

function fetchSensorData() {{
    fetch('/sensor_data')
    .then(res => res.json())
    .then(data => {{
        const now = new Date().toLocaleTimeString();
        [chart1, chart2, chart3].forEach((chart,i) => {{
            chart.data.labels.push(now);
            chart.data.datasets[0].data.push(data['field'+(i+1)]);
            if(chart.data.labels.length > 20){{
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
            }}
            chart.update();
        }});
    }});
}}

setInterval(fetchSensorData, 5000);
</script>
</body>
</html>
"""


@app.route("/sensor_data")
def sensor_data():
    return jsonify({
        'field1': get_latest_field(1),
        'field2': get_latest_field(2),
        'field3': get_latest_field(3)
    })

@app.route("/video_feed")
def video_feed():
    def get_annot():
        with lock:
            return None if latest_annot is None else latest_annot.copy()
    return Response(mjpeg_generator(get_annot),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_raw")
def video_feed_raw():
    def get_raw():
        with lock:
            return None if latest_raw is None else latest_raw.copy()
    return Response(mjpeg_generator(get_raw),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    start_threads()
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
