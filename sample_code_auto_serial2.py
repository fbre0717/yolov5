from models.common import DetectMultiBackend, AutoShape
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import torch
import numpy as np
from PIL import Image
import time
start = time.time()

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
print(f"{time.time()-start:.3f}s Model loaded")

# Warm up
model.eval()
with torch.no_grad():
    dummy_input = torch.zeros((1, 3, 640, 480)).to(device)
    for _ in range(2):
        _ = model(dummy_input)
print(f"{time.time()-start:.3f}s Warm up process end")


import cv2
import pyrealsense2 as rs
import serial


py_serial = serial.Serial(  port = '/dev/ttyACM0',  baudrate=57600)


GIST_INSTANCE_CATEGORY_NAMES = ['ai', 'awear', 'imr', 'gist']
choose_labels = GIST_INSTANCE_CATEGORY_NAMES
extract_idx = [i for i, label in enumerate(GIST_INSTANCE_CATEGORY_NAMES) if label in choose_labels]

# Set up the RealSense D455 camera
print(f"{time.time()-start:.3f}s Setting RealSense Camera...")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
# Write your yolov5 depth scale here
depth_scale = 0.001
print(f"{time.time()-start:.3f}s Setting RealSense Camera finished.")



# Rotate 180 degree

print('rwrwrw')


# Main loopprint
state = 0
rotate_180 = False
print(f"{time.time()-start:.3f}s Start frist frame")
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
        # gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Convert the depth image to meters
        depth_image = depth_image * depth_scale

        # Detect objects using YOLOv5
        results = model(color_image)

        # Process the results
        for result in results.xyxy[0].cpu().numpy():
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
                    if state == 1:
                        if angle < 15 or angle > -15:
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
                    print("angle :", angle, "current state :", state, "serial command :", command)

                    py_serial.write(command.encode('utf-8'))
                else:
                    print("Rotate 190 is False")
        # Show the image
        cv2.imshow("Color Image", color_image)

        # Break out of the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not rotate_180:
            print(f"{time.time()-start:.3f}s Please insert any key.")
            start_key = input()
            right_command = 'r'
            py_serial.write(right_command.encode('utf-8'))
            print(right_command)
            run_command = 'w'
            py_serial.write(run_command.encode('utf-8'))
            print(run_command)
            state = 2

            rotate_180 = True
            print("Rotate 180 is True")


# Release resources

cv2.destroyAllWindows()
pipeline.stop()