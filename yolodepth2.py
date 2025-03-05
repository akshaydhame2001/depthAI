# code with OAK-D depth and RGB OAK-D with object detection processing on host Jetson
import cv2
import numpy as np
import depthai as dai
import time
import threading
from ultralytics import YOLO

# Global variables
THREAD_STOP = False
RUNNING = False

def initialize_oak_pipeline():
    """Initializes the OAK-D pipeline for RGB and depth."""
    pipeline = dai.Pipeline()
    
    # Create nodes
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_depth = pipeline.create(dai.node.XLinkOut)
    
    # Configure RGB camera
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setPreviewSize(1280, 720)
    xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)
    
    # Configure stereo depth
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    
    xout_depth.setStreamName("depth")
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

def yolo_inference_on_oak(frame, model):
    """Runs YOLOv8 on the RGB frames from OAK-D."""
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
    """Thread that handles YOLO inference on OAK-D RGB and depth estimation."""
    global THREAD_STOP, RUNNING
    
    pipeline = initialize_oak_pipeline()
    with dai.Device(pipeline) as device:
        rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        RUNNING = True
        
        while not THREAD_STOP:
            inRgb = rgbQueue.get()
            inDepth = depthQueue.get()
            
            if inRgb is None or inDepth is None:
                continue
            
            frame = inRgb.getCvFrame()
            depth_frame = inDepth.getFrame()
            
            detections = yolo_inference_on_oak(frame, model)
            depth_results = get_depth_from_oak(detections, depth_frame)
            callback(depth_results, frame)
        
        RUNNING = False

def start_oak_yolo(callback):
    """Starts the OAK-D based YOLO and depth thread."""
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
