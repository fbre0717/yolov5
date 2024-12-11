import cv2
import numpy as np
import torch
import time
import serial
import pyrealsense2 as rs
import pathlib
from models.common import DetectMultiBackend, AutoShape
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from PIL import Image

class ModelLoader:
    def __init__(self, weights_path, device=None):
        self.weights = weights_path
        self.device = select_device(device) if device is None else device
        self.model = None
        
    def load_model(self):
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=None)
        self.model = AutoShape(self.model)
        self.model.eval()
        
    def warm_up(self):
        with torch.no_grad():
            dummy_input = torch.zeros((1, 3, 640, 480)).to(self.device)
            for _ in range(2):
                _ = self.model(dummy_input)

class RealSenseCamera:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.depth_scale = 0.001
        
    def setup(self):
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        self.pipeline.start(self.config)
        
    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()) * self.depth_scale
        
        return color_image, depth_image
    
    def stop(self):
        self.pipeline.stop()

class ObjectDetector:
    def __init__(self, model, category_names):
        self.model = model
        self.category_names = category_names
        self.FOV_HORIZONTAL = 87
        self.pixel_to_degree = self.FOV_HORIZONTAL / 640
        
    def detect(self, color_image, depth_image):
        results = self.model(color_image)
        detections = []
        
        for result in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2, confidence, class_id = result
            
            if int(class_id) in [i for i, _ in enumerate(self.category_names)]:
                detection = self._process_detection(result, depth_image)
                detections.append(detection)
                
        return detections
    
    def _process_detection(self, result, depth_image):
        x1, y1, x2, y2, confidence, class_id = result
        center_x = (x1 + x2) / 2
        image_center = 640 / 2
        
        object_depth = np.median(depth_image[int(y1):int(y2), int(x1):int(x2)])
        pixel_difference = center_x - image_center
        angle = pixel_difference * self.pixel_to_degree
        
        return {
            'bbox': (int(x1), int(y1), int(x2), int(y2)),
            'class_id': int(class_id),
            'class_name': self.model.names[int(class_id)],
            'depth': object_depth,
            'angle': angle,
            'confidence': confidence
        }

class Visualizer:
    @staticmethod
    def draw_detections(image, detections):
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            label = f"{detection['class_name']}: {detection['depth']:.2f}m, angle: {detection['angle']:.1f}"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (252, 119, 30), 2)
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (252, 119, 30), 2)
        
        return image

class RobotController:
    def __init__(self, port='/dev/ttyACM0', baudrate=57600):
        self.serial = serial.Serial(port=port, baudrate=baudrate)
        self.state = 0
        self.rotate_180 = False
        
    def process_detection(self, detection):
        if not self.rotate_180:
            return None
            
        angle = detection['angle']
        command = None
        
        if self.state == 1:
            if abs(angle) < 20:
                command = 'w'
                self.state = 2
        elif self.state == 2:
            if angle >= 25:
                command = 'sd'
                self.state = 1
            elif angle <= -25:
                command = 'sa'
                self.state = 1
                
        if command:
            self.send_command(command)
            print(f"angle: {angle}, current state: {self.state}, serial command: {command}")
            
        return command
    
    def send_command(self, command):
        self.serial.write(command.encode('utf-8'))
        
    def initialize_rotation(self):
        self.send_command('r')
        self.send_command('w')
        self.state = 2
        self.rotate_180 = True

def main():
    # Fix for WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath
    
    # Initialize components
    start_time = time.time()
    
    model_loader = ModelLoader('./runs/train/exp/weights/best.pt')
    model_loader.load_model()
    model_loader.warm_up()
    print(f"{time.time()-start_time:.3f}s Model loaded and warmed up")
    
    camera = RealSenseCamera()
    camera.setup()
    print(f"{time.time()-start_time:.3f}s Camera initialized")
    
    category_names = ['ai', 'awear', 'imr', 'gist']
    detector = ObjectDetector(model_loader.model, category_names)
    
    robot = RobotController()
    visualizer = Visualizer()
    
    print(f"{time.time()-start_time:.3f}s Starting detection loop")
    
    # Main loop
    with torch.no_grad():
        while True:
            color_image, depth_image = camera.get_frames()
            detections = detector.detect(color_image, depth_image)
            
            if detections:
                color_image = visualizer.draw_detections(color_image, detections)
                for detection in detections:
                    robot.process_detection(detection)
            
            cv2.imshow("Color Image", color_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            if not robot.rotate_180:
                print(f"{time.time()-start_time:.3f}s Please insert any key.")
                input()
                robot.initialize_rotation()
                print("Rotate 180 is True")
    
    # Cleanup
    cv2.destroyAllWindows()
    camera.stop()

if __name__ == "__main__":
    main()