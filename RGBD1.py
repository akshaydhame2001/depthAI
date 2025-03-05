# Code for this repo(https://github.com/sieuwe1/Autonomous-Ai-drone-scripts/)
# to use OAK-D depth camera
import depthai as dai
import cv2
import numpy as np

def initialize_oak_pipeline():
    """Initializes the OAK-D pipeline for RGB and depth streaming."""
    pipeline = dai.Pipeline()
    
    # Create color camera node
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K) # (THE_12_MP)
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

def get_rgbd_image(device):
    """Fetches RGB and Depth images from OAK-D."""
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    
    rgb_frame = None
    depth_frame = None
    
    in_rgb = rgb_queue.get()
    in_depth = depth_queue.get()
    
    if in_rgb is not None:
        rgb_frame = in_rgb.getCvFrame()
    
    if in_depth is not None:
        depth_frame = in_depth.getFrame()
        depth_frame = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return rgb_frame, depth_frame

if __name__ == "__main__":
    pipeline = initialize_oak_pipeline()
    with dai.Device(pipeline) as device:
        while True:
            rgb, depth = get_rgbd_image(device)
            if rgb is not None and depth is not None:
                stacked = np.hstack((rgb, cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)))
                cv2.imshow("OAK-D RGB & Depth", stacked)
            
            if cv2.waitKey(1) == ord('q'):
                break
    cv2.destroyAllWindows()
