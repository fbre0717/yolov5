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
model = DetectMultiBackend(weights, device=device, dnn=False, data=None)
model = AutoShape(model)

# Warm up
model.eval()
with torch.no_grad():
    dummy_input = torch.zeros((1, 3, 640, 480)).to(device)
    for _ in range(2):
        _ = model(dummy_input)
    print("warm up process end")


# Load the YOLOv5 model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

import cv2
import pyrealsense2 as rs
import serial
import time

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
print("Setting RealSense Camera...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
# Write your yolov5 depth scale here
depth_scale = 0.001
print("Setting RealSense Camera finished.")


# Rotate 180 degree




# Main loopprint
state = 0
rotate_180 = False
with torch.no_grad():
    while True:

        # Get the latest frame from the cameraprint
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

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
                # 객체의 중심 x좌표 계산
                center_x = (x1 + x2) / 2
                # 이미지의 중심 x좌표 (640x480 이미지이므로 중심은 320)
                image_center = 640 / 2  # 320

                # Calculate the distance to the object
                object_depth = np.median(depth_image[int(y1):int(y2), int(x1):int(x2)])

                # 픽셀 차이를 실제 거리로 변환
                # RealSense D455의 수평 FOV는 87도
                FOV_HORIZONTAL = 87  # degrees
                pixel_to_degree = FOV_HORIZONTAL / 640  # degree/pixel

                # 중심으로부터의 픽셀 차이
                pixel_difference = center_x - image_center

                # 각도 계산 (탄젠트 사용)
                # 음수: 왼쪽, 양수: 오른쪽
                angle = pixel_difference * pixel_to_degree

                # Get the object's class name
                class_name = model.names[int(class_id)]

                # Create label with class name, distance, and center position
                label = f"{class_name}: {object_depth:.2f}m, angle: {angle:.1f}"

                # Draw a rectangle around the object
                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (252, 119, 30), 2)

                # Draw the label
                cv2.putText(color_image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (252, 119, 30), 2)

                if rotate_180:
                    if state == 0:
                        if angle <= -15:
                            command = 'sa'
                            state = 1
                        elif angle >= 15:
                            command = 'sd'
                            state = 1
                        else:
                            # command = 'w'
                            state = 2
                            continue

                    elif state == 1:
                        if angle < 30 or angle > -30:
                            command = 'w'
                            state = 2
                        else:
                            continue

                    elif state == 2:
                        if angle >= 30:
                            command = 'sd'
                            state = 1
                        elif angle <= -30:
                            command = 'sa'
                            state = 1
                        else:
                            continue
                    print("current state :", state)
                    print("serial command :", command)

                    py_serial.write(command.encode('utf-8'))
                else:
                    print("Rotate 190 is False")
        # Show the image
        cv2.imshow("Color Image", color_image)

        # Break out of the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not rotate_180:
            start_key = input()
            right_command = 'r'
            py_serial.write(right_command.encode('utf-8'))
            print(right_command)
            run_command = 'w'
            py_serial.write(run_command.encode('utf-8'))
            print(run_command)
            rotate_180 = True
            print("Rotate 180 is True")


# Release resources

cv2.destroyAllWindows()
pipeline.stop()