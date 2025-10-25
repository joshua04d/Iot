#!/usr/bin/env python3
import os
import time
import threading
import platform
import requests

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory
from ultralytics import YOLO

# Optional fast JPEG encoder
try:
    from turbojpeg import TurboJPEG
    jpeg = TurboJPEG()
except Exception:
    jpeg = None

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "fire_smoke_yolo11s_20epochs.pt"
CAM_INDEX = 2
BACKEND = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY

IMG_MAX_WIDTH = 960     # will auto-reduce on CPU (see below)
CONF = 0.25
TARGET_INF_FPS = 30
STREAM_FPS = 30
JPEG_QUALITY = 70
FRAME_SKIP = 2          # may auto-increase on CPU (see below)
USE_CUDA = os.environ.get("USE_CUDA", "1") == "1"  # set to 0 to force CPU

# ThingSpeak settings
CHANNEL_ID = "3131292"
TS_URL = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/fields/{{}}.json?results=1"

# ----------------------------
# Init
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")

# Serve /static/* from the web folder
app = Flask(__name__, static_folder=WEB_DIR, static_url_path="/static")

print(f"[INFO] Loading YOLO model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Decide device (GPU if available and allowed, else CPU)
GPU = False
try:
    import torch
    gpu_available = torch.cuda.is_available()
    if USE_CUDA and gpu_available:
        model.to("cuda")
        GPU = True
        # Optional small speedup for constant resolution inputs
        torch.backends.cudnn.benchmark = True
        print("[INFO] Using GPU (CUDA)")
    else:
        model.to("cpu")
        print("[INFO] Using CPU")
        # Optional: let torch use many threads
        try:
            threads = int(os.environ.get("CPU_THREADS", "0"))
            if threads > 0:
                torch.set_num_threads(threads)
                print(f"[INFO] torch.set_num_threads({threads})")
        except Exception:
            pass
except Exception as e:
    # If torch is not present for some reason, Ultralytics would fail earlier,
    # but keep a helpful message.
    print(f"[WARN] Torch check failed ({e}). Continuing; model will default to CPU.")

# Slight speed-up
try:
    model.fuse()
except Exception:
    pass

# Auto-tune for CPU to keep FPS reasonable
if not GPU:
    # If you want to force different values, set them via env vars before running.
    if IMG_MAX_WIDTH and IMG_MAX_WIDTH > 640:
        print(f"[INFO] CPU detected: reducing IMG_MAX_WIDTH {IMG_MAX_WIDTH} -> 640")
        IMG_MAX_WIDTH = 640
    if FRAME_SKIP < 2:
        print(f"[INFO] CPU detected: increasing FRAME_SKIP {FRAME_SKIP} -> 2")
        FRAME_SKIP = 2

# ----------------------------
# Camera setup
# ----------------------------
cap = cv2.VideoCapture(CAM_INDEX, BACKEND)
if not cap.isOpened():
    raise RuntimeError(f"[ERROR] Could not open camera index {CAM_INDEX}")

# Reduce buffering if supported
try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
except Exception:
    pass

# Shared state
lock = threading.Lock()
latest_raw = None
latest_annot = None
stop_flag = False

# ----------------------------
# Helpers
# ----------------------------
def resize_if_needed(frame):
    if IMG_MAX_WIDTH and frame.shape[1] > IMG_MAX_WIDTH:
        scale = IMG_MAX_WIDTH / frame.shape[1]
        frame = cv2.resize(
            frame,
            (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return frame

def encode_jpeg(img_bgr):
    if jpeg:
        try:
            return jpeg.encode(img_bgr, quality=JPEG_QUALITY)
        except Exception:
            pass
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
        value = r.get('feeds', [{}])[-1].get(f'field{field_num}', None)
        return float(value) if value not in (None, "") else 0.0
    except Exception:
        return 0.0

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
            latest_raw = frame  # drop older frames, keep freshest

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
        if FRAME_SKIP > 1 and (counter % FRAME_SKIP != 0):
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
    # Serve web/index.html
    return send_from_directory(WEB_DIR, "index.html")

@app.route("/favicon.ico")
def favicon():
    # Optional favicon support; returns empty if not present
    path = os.path.join(WEB_DIR, "favicon.ico")
    if os.path.exists(path):
        return send_from_directory(WEB_DIR, "favicon.ico")
    return ("", 204)

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