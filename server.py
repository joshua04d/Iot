#!/usr/bin/env python3
import os
import time
import threading
import platform

import cv2
import numpy as np
from flask import Flask, Response
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
CAM_INDEX = 1                                   # OBS / VTX capture index
BACKEND = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY

# Performance knobs
IMG_MAX_WIDTH = 960         # downscale input to this width (None to disable)
CONF = 0.25                 # detection confidence
TARGET_INF_FPS = 30         # max YOLO inference rate (drops frames to keep live)
STREAM_FPS = 30             # how fast to push MJPEG frames out
JPEG_QUALITY = 80           # 60-90 typical

# Use GPU if available
USE_CUDA = os.environ.get("USE_CUDA", "1") == "1"

# ----------------------------
# Init
# ----------------------------
app = Flask(__name__)

print(f"[INFO] Loading YOLO model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
try:
    if USE_CUDA:
        import torch
        if torch.cuda.is_available():
            model.to("cuda")
            GPU = True
            print("[INFO] Using GPU (CUDA)")
        else:
            GPU = False
            print("[INFO] CUDA not available, using CPU")
    else:
        GPU = False
        print("[INFO] Forcing CPU")
except Exception:
    GPU = False
    print("[WARN] Torch/CUDA not detected, using CPU")

# Slight speed-up on CPU
try:
    model.fuse()
except Exception:
    pass

cap = cv2.VideoCapture(CAM_INDEX, BACKEND)
if not cap.isOpened():
    raise RuntimeError(f"[ERROR] Could not open camera index {CAM_INDEX}")

# Reduce buffering where supported
try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, STREAM_FPS)
except Exception:
    pass

# Shared state
lock = threading.Lock()
latest_raw = None          # last captured raw frame (BGR)
latest_annot = None        # last annotated frame (BGR)
stop_flag = False

def resize_if_needed(frame):
    if IMG_MAX_WIDTH and frame.shape[1] > IMG_MAX_WIDTH:
        scale = IMG_MAX_WIDTH / frame.shape[1]
        new_w = int(frame.shape[1] * scale)
        new_h = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame

def capture_loop():
    global latest_raw
    while not stop_flag:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.005)
            continue
        with lock:
            latest_raw = frame  # overwrite (drop older) to stay live

def yolo_loop():
    global latest_annot
    prev_t = time.time()
    min_interval = 1.0 / max(TARGET_INF_FPS, 1)

    while not stop_flag:
        # pace inference
        now = time.time()
        sleep_left = min_interval - (now - prev_t)
        if sleep_left > 0:
            time.sleep(sleep_left)
        prev_t = time.time()

        # grab the newest raw frame
        with lock:
            frame = None if latest_raw is None else latest_raw.copy()

        if frame is None:
            time.sleep(0.005)
            continue

        t0 = time.time()
        frame = resize_if_needed(frame)

        # YOLO inference (GPU if available)
        results = model.predict(
            frame,
            conf=CONF,
            imgsz=IMG_MAX_WIDTH if IMG_MAX_WIDTH else None,
            half=True if GPU else False,
            verbose=False,
            max_det=50,
        )
        annotated = results[0].plot()

        # Overlay inference FPS
        inf_fps = 1.0 / max(time.time() - t0, 1e-6)
        cv2.putText(
            annotated, f"INF FPS: {inf_fps:.1f}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )

        with lock:
            latest_annot = annotated

def encode_jpeg(img_bgr):
    if jpeg:
        return jpeg.encode(img_bgr, quality=JPEG_QUALITY,
                           pixel_format=TJPF_BGR, jpeg_subsample=TJSAMP_420,
                           flags=TJFLAG_FASTDCT)
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

# Routes
@app.route("/")
def index():
    # simple landing page
    return (
        "<html><head><title>Fire/Smoke Detection</title></head>"
        "<body style='font-family:sans-serif'>"
        "<h1>Fire/Smoke Detection</h1>"
        "<p><a href='/video_feed'>Annotated Output</a> | "
        "<a href='/video_feed_raw'>Raw Input</a></p>"
        "<img src='/video_feed' style='width:100%;max-width:1000px;border-radius:8px'/>"
        "</body></html>"
    )

@app.route("/video_feed")
def video_feed():
    # Annotated YOLO output (the 'o/p' you want)
    def get_annot():
        with lock:
            return None if latest_annot is None else latest_annot.copy()
    return Response(mjpeg_generator(get_annot),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_raw")
def video_feed_raw():
    # Raw camera feed (for comparison/debug)
    def get_raw():
        with lock:
            return None if latest_raw is None else latest_raw.copy()
    return Response(mjpeg_generator(get_raw),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def start_threads():
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=yolo_loop, daemon=True).start()

if __name__ == "__main__":
    start_threads()
    # Run Flask without reloader so we don't duplicate threads/camera
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)