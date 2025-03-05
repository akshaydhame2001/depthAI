# code with depth on OAK-D and RGB object detection on external camera
import cv2
import numpy as np
import depthai as dai
import time
import threading
from ultralytics import YOLO

# Global variables
THREAD_STOP = False
RUNNING = False

def initialize_depth_pipeline():
    """Initializes the OAK-D depth pipeline."""
    pipeline = dai.Pipeline()
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    xout_depth = pipeline.create(dai.node.XLinkOut)
    
    xout_depth.setStreamName("depth")
    
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    stereo.depth.link(xout_depth.input)
    
    return pipeline

def get_depth_from_oak(detections, depth_frame):
    """Extracts depth information from OAK-D based on YOLO detections."""
    depth_values = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        depth = depth_frame[center_y, center_x] if depth_frame is not None else -1
        depth_values.append({
            "class": det['class'],
            "bbox": (x1, y1, x2, y2),
            "depth": depth
        })
    return depth_values

def yolo_inference_on_jetson(frame, model):
    """Runs YOLOv8 on the Jetson host to detect objects."""
    results = model(frame)
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        for box, cls in zip(boxes, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            detections.append({"class": cls, "bbox": (x1, y1, x2, y2)})
    return detections

def processing_thread(callback, model):
    """Thread that handles YOLO inference on Jetson and depth estimation on OAK-D."""
    global THREAD_STOP, RUNNING
    
    cap = cv2.VideoCapture(0)  # Use Jetson's camera
    pipeline = initialize_depth_pipeline()
    with dai.Device(pipeline) as device:
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        RUNNING = True
        
        while not THREAD_STOP:
            ret, frame = cap.read()
            if not ret:
                continue
            
            detections = yolo_inference_on_jetson(frame, model)
            inDepth = depthQueue.get()
            depth_frame = inDepth.getFrame() if inDepth is not None else None
            
            depth_results = get_depth_from_oak(detections, depth_frame)
            callback(depth_results, frame)
        
        cap.release()
        RUNNING = False

def start_jetson_yolo(callback):
    """Starts the Jetson-based YOLO and OAK-D depth thread."""
    global THREAD_STOP
    model = YOLO("yolov8n.pt")
    THREAD_STOP = False
    thread = threading.Thread(target=processing_thread, args=(callback, model))
    thread.start()
    return thread

def stop():
    """Stops the YOLO processing thread."""
    global THREAD_STOP
    THREAD_STOP = True
