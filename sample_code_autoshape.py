from models.common import DetectMultiBackend, AutoShape
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import torch
import numpy as np
from PIL import Image

import pathlib
import platform

# WindowsPath 문제 해결을 위한 패치
temp = pathlib.WindowsPath
pathlib.WindowsPath = pathlib.PosixPath

# Load the YOLOv5 custom model
weights = './runs/train/exp/weights/best.pt'
device = select_device()
model = DetectMultiBackend(weights, device=device, dnn=False, data=None).eval()
model = AutoShape(model)


# Load the YOLOv5 model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

import cv2
import pyrealsense2 as rs
import serial

py_serial = serial.Serial(  port = '/dev/ttyACM0',  baudrate=57600)


# Define COCO instance category names for label identification
COCO_INSTANCE_CATEGORY_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
GIST_INSTANCE_CATEGORY_NAMES = ['ai', 'awear', 'imr', 'gist']

# choose_labels = ['person']
choose_labels = GIST_INSTANCE_CATEGORY_NAMES
# choose_labels = COCO_INSTANCE_CATEGORY_NAMES
extract_idx = [i for i, label in enumerate(GIST_INSTANCE_CATEGORY_NAMES) if label in choose_labels]
# extract_idx = [i for i, label in enumerate(COCO_INSTANCE_CATEGORY_NAMES) if label in choose_labels]

# Set up the RealSense D455 camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
# Write your yolov5 depth scale here
depth_scale = 0.001


def preprocess_realsense_image(color_image, img_size=640):
    """
    Preprocess RealSense color image for YOLOv5 model input
    
    Args:
        color_image (np.ndarray): BGR color image from RealSense camera with shape (480, 640, 3)
        img_size (int): Target image size for model input
        
    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input
    """
    # Input validation
    if not isinstance(color_image, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if len(color_image.shape) != 3:
        raise ValueError("Input must have 3 dimensions (H, W, C)")
    if color_image.shape[2] != 3:
        raise ValueError("Input must have 3 channels")
        
    # BGR to RGB conversion (RealSense provides BGR format)
    img = color_image[..., ::-1].copy()  # BGR to RGB
    
    # Letterbox 리사이징
    img = letterbox(img, new_shape=img_size, auto=True)[0]
    
    # Convert
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)
    
    # Convert to torch, normalize
    img = torch.from_numpy(img).float()
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    
    # Add batch dimension
    if len(img.shape) == 3:
        img = img.unsqueeze(0)  # add batch dimension
    
    return img



# Main loopprint

while True:

    # Get the latest frame from the cameraprint
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    # # Convert the frames to numpy arrays
    # color_image = np.asanyarray(color_frame.get_data())
    # depth_image = np.asanyarray(depth_frame.get_data())

    # # Convert the color image to grayscale
    # gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # # Convert the depth image to meters
    # depth_image = depth_image * depth_scale

    # # Preprocess color image for YOLOv5
    # processed_image = preprocess_realsense_image(color_image)
    
    # # Move to appropriate device (GPU/CPU)
    # processed_image = processed_image.to(device)
    
    # # Detect objects using YOLOv5
    # results = model(processed_image)


    # Convert the frames to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Convert the color image to grayscale
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Convert the depth image to meters
    depth_image = depth_image * depth_scale

    # Detect objects using YOLOv5
    results = model(color_image)

    # Process the results
    for result in results.xyxy[0]:
        x1, y1, x2, y2, confidence, class_id = result

        if int(class_id) in extract_idx:
            # Calculate the distance to the object
            object_depth = np.median(depth_image[int(y1):int(y2), int(x1):int(x2)])

            # Get the object's class name
            class_name = model.names[int(class_id)]

            # Create label with class name and distance
            label = f"{class_name}: {object_depth:.2f}m"

            # Draw a rectangle around the object
            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (252, 119, 30), 2)

            # Draw the label
            cv2.putText(color_image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (252, 119, 30), 2)

            # Print the object's class and distance
            print(label, class_id)
            if object_depth < 0.5:
                com = 'b'
                py_serial.write(com.encode('utf-8'))
            else:
                com = 'a'
                py_serial.write(com.encode('utf-8'))
    # Show the image
    cv2.imshow("Color Image", color_image)

    # Break out of the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources

cv2.destroyAllWindows()
pipeline.stop()