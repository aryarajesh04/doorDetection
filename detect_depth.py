#!/usr/bin/env python3
"""
Live door detection with spatial depth on Luxonis OAK camera.

BEFORE RUNNING — export your trained model to .blob:
  1. Export to OpenVINO:
       from ultralytics import YOLO
       model = YOLO("runs/door_detection/weights/best.pt")
       model.export(format="openvino", imgsz=640)
       # produces: runs/door_detection/weights/best_openvino_model/

  2. Convert to .blob (install once: pip install blobconverter):
       import blobconverter
       blob_path = blobconverter.from_openvino(
           xml="runs/door_detection/weights/best_openvino_model/best.xml",
           bin="runs/door_detection/weights/best_openvino_model/best.bin",
           shaves=6,
       )
       print(blob_path)

  3. Run this script:
       python detect_depth.py path/to/best.blob
"""

import sys
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

LABEL_MAP = ["closed", "open", "semi"]
LABEL_COLORS = {          # BGR
    "closed": (0,   0,   220),   # red
    "open":   (0,   220, 0  ),   # green
    "semi":   (0,   200, 255),   # yellow
}
CONFIDENCE_THRESHOLD = 0.5
PREVIEW_W, PREVIEW_H = 640, 640
DEPTH_LOWER_MM = 100    # 10 cm
DEPTH_UPPER_MM = 8000   # 8 m

# ── blob path from CLI arg ────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python detect_depth.py <path/to/best.blob>")
    sys.exit(1)

blob_path = Path(sys.argv[1])
if not blob_path.exists():
    raise FileNotFoundError(f"Blob not found: {blob_path}")

# ── build pipeline ────────────────────────────────────────────────────────────
pipeline = dai.Pipeline()

camRgb      = pipeline.create(dai.node.ColorCamera)
monoLeft    = pipeline.create(dai.node.MonoCamera)
monoRight   = pipeline.create(dai.node.MonoCamera)
stereo      = pipeline.create(dai.node.StereoDepth)
spatialNet  = pipeline.create(dai.node.YoloSpatialDetectionNetwork)

xoutRgb     = pipeline.create(dai.node.XLinkOut)
xoutDet     = pipeline.create(dai.node.XLinkOut)
xoutDepth   = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutDet.setStreamName("detections")
xoutDepth.setStreamName("depth")

# Color camera
camRgb.setPreviewSize(PREVIEW_W, PREVIEW_H)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(30)

# Mono cameras for stereo depth
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

# Stereo depth
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align depth to RGB perspective
stereo.setSubpixel(True)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

# YOLOv8 spatial detection network (anchor-free — no setAnchors needed)
spatialNet.setBlobPath(blob_path)
spatialNet.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
spatialNet.setNumClasses(len(LABEL_MAP))
spatialNet.setCoordinateSize(4)
spatialNet.setIouThreshold(0.5)
spatialNet.setDepthLowerThreshold(DEPTH_LOWER_MM)
spatialNet.setDepthUpperThreshold(DEPTH_UPPER_MM)
spatialNet.setBoundingBoxScaleFactor(0.5)
spatialNet.input.setBlocking(False)
spatialNet.setNumInferenceThreads(2)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
camRgb.preview.link(spatialNet.input)
spatialNet.passthrough.link(xoutRgb.input)
spatialNet.out.link(xoutDet.input)
stereo.depth.link(spatialNet.inputDepth)
spatialNet.passthroughDepth.link(xoutDepth.input)

# ── run ───────────────────────────────────────────────────────────────────────
with dai.Device(pipeline) as device:
    qRgb   = device.getOutputQueue("rgb",        maxSize=4, blocking=False)
    qDet   = device.getOutputQueue("detections", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue("depth",      maxSize=4, blocking=False)

    start_time = time.monotonic()
    counter = 0
    fps = 0.0
    WHITE = (255, 255, 255)

    while True:
        frame_msg = qRgb.get()
        det_msg   = qDet.get()
        depth_msg = qDepth.get()

        frame      = frame_msg.getCvFrame()
        depth_raw  = depth_msg.getFrame()   # values in mm
        detections = det_msg.detections

        # FPS
        counter += 1
        now = time.monotonic()
        if now - start_time >= 1.0:
            fps = counter / (now - start_time)
            counter = 0
            start_time = now

        # Colorise depth for visualisation
        d = depth_raw[::4]
        min_d = np.percentile(d[d != 0], 1) if not np.all(d == 0) else 0
        max_d = np.percentile(d, 99)
        depth_color = np.interp(depth_raw, (min_d, max_d), (0, 255)).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_color, cv2.COLORMAP_HOT)

        h, w = frame.shape[:2]

        for det in detections:
            # Draw ROI on depth frame
            roi = det.boundingBoxMapping.roi.denormalize(
                depth_color.shape[1], depth_color.shape[0]
            )
            cv2.rectangle(
                depth_color,
                (int(roi.topLeft().x), int(roi.topLeft().y)),
                (int(roi.bottomRight().x), int(roi.bottomRight().y)),
                WHITE, 1,
            )

            # Bounding box on RGB
            x1, x2 = int(det.xmin * w), int(det.xmax * w)
            y1, y2 = int(det.ymin * h), int(det.ymax * h)
            label = LABEL_MAP[det.label] if det.label < len(LABEL_MAP) else str(det.label)

            color = LABEL_COLORS.get(label, WHITE)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            for i, text in enumerate([
                f"{label}  {int(det.confidence * 100)}%",
                f"X: {int(det.spatialCoordinates.x)} mm",
                f"Y: {int(det.spatialCoordinates.y)} mm",
                f"Z: {int(det.spatialCoordinates.z)} mm",
            ]):
                cv2.putText(frame, text, (x1 + 6, y1 + 18 + i * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        cv2.putText(frame, f"FPS: {fps:.1f}", (4, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

        cv2.imshow("OAK — RGB + Detections", frame)
        cv2.imshow("OAK — Depth", depth_color)

        if cv2.waitKey(1) == ord("q"):
            break
