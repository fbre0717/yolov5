from models.common import DetectMultiBackend
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

# 모델 로드
weights = './runs/train/exp/weights/best.pt'
device = select_device()
model_1 = DetectMultiBackend(weights, device=device, dnn=False, data=None).eval()
print("model loaded")

def preprocess_numpy_image(img, img_size=640):
    """
    Preprocess numpy array image for YOLOv5 model input
    
    Args:
        img (np.ndarray): Input image in numpy array format with shape (height, width, channels)
        img_size (int): Target image size for model input
        
    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input
    """
    # Input validation
    if not isinstance(img, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if len(img.shape) != 3:
        raise ValueError("Input must have 3 dimensions (H, W, C)")
    if img.shape[2] != 3:
        raise ValueError("Input must have 3 channels")
        
    # 원본 이미지 복사하여 수정
    img = img.copy()
    
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

def preprocess_image(image_path, img_size=640):
    # 이미지 로드 및 화질 조정
    image = Image.open(image_path)
    image.save('temp_image.jpg', quality=75)
    image = Image.open('temp_image.jpg')
    
    # PIL Image를 numpy array로 변환
    img = np.array(image)
    
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

# 이미지 경로
image_path = './data/images/bus.jpg'

# 이미지 전처리
processed_img = preprocess_image(image_path)
print("Processed image shape:", processed_img.shape)  # should be [1, 3, height, width]

# 이미지 전처리2
image = Image.open(image_path)
image.save('temp_image.jpg', quality=75)
image = Image.open('temp_image.jpg')
img = np.array(image)
processed_img2 = preprocess_numpy_image(img)
print("Processed image2 shape:", processed_img2.shape)  # should be [1, 3, height, width]


# 모델에 이미지 전달
with torch.no_grad():
    processed_img = processed_img.to(device)
    results = model_1(processed_img)
    print("jpg to custom model")

    processed_img2 = processed_img2.to(device)
    results1_2 = model_1(processed_img2)
    print("np image to custom model")

# Pre-trained 모델 테스트
# model_2 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
# results2 = model_2(processed_img)
# print("jpg to pretrained model")
