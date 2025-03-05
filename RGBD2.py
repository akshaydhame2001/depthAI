# HostSpatialsCalc approach
import depthai as dai
import cv2
import numpy as np
import math

class HostSpatialsCalc:
    def __init__(self, device, delta=5):
        self.delta = delta
        self.depth_config = dai.StereoDepthConfig()
        self.device = device
    
    def setDeltaRoi(self, delta):
        self.delta = delta
    
    def calc_spatials(self, depthFrame, centroid):
        x, y = centroid
        roi = depthFrame[y-self.delta:y+self.delta, x-self.delta:x+self.delta]
        mean_depth = np.mean(roi[roi > 0]) if np.any(roi > 0) else 0
        return {"x": x, "y": y, "z": mean_depth}

def initialize_oak_pipeline():
    """Initializes the OAK-D pipeline for RGB and depth streaming."""
    pipeline = dai.Pipeline()
    
    # Create color camera node
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam_rgb.setInterleaved(False)
    
    # Create mono cameras for depth
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    
    # Linking
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    
    # Output streams
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_depth = pipeline.create(dai.node.XLinkOut)
    
    xout_rgb.setStreamName("rgb")
    xout_depth.setStreamName("depth")
    
    cam_rgb.video.link(xout_rgb.input)
    stereo.depth.link(xout_depth.input)
    
    return pipeline

def get_rgbd_image(device, hostSpatials, centroid=(300, 200)):
    """Fetches RGB and Depth images from OAK-D and calculates spatial coordinates."""
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    
    rgb_frame = None
    depth_frame = None
    spatials = {"x": 0, "y": 0, "z": 0}
    
    in_rgb = rgb_queue.get()
    in_depth = depth_queue.get()
    
    if in_rgb is not None:
        rgb_frame = in_rgb.getCvFrame()
    
    if in_depth is not None:
        depth_frame = in_depth.getFrame()
        depth_frame = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
        
        # Calculate spatial coordinates
        spatials = hostSpatials.calc_spatials(depth_frame, centroid)
    
    return rgb_frame, depth_colormap, spatials

if __name__ == "__main__":
    pipeline = initialize_oak_pipeline()
    with dai.Device(pipeline) as device:
        hostSpatials = HostSpatialsCalc(device)
        centroid = (300, 200)  # Default point for spatial calculation
        
        while True:
            rgb, depth, coords = get_rgbd_image(device, hostSpatials, centroid)
            if rgb is not None and depth is not None:
                cv2.putText(rgb, f"X: {coords['x'] / 1000:.2f}m", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(rgb, f"Y: {coords['y'] / 1000:.2f}m", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(rgb, f"Z: {coords['z'] / 1000:.2f}m", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                stacked = np.hstack((rgb, depth))
                cv2.imshow("OAK-D RGB & Depth with Coordinates", stacked)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    cv2.destroyAllWindows()