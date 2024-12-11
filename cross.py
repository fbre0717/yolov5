import os
import sys
import torch
import pathlib
import platform

# WindowsPath 문제 해결을 위한 패치
temp = pathlib.WindowsPath
pathlib.WindowsPath = pathlib.PosixPath

# YOLOv5 디렉토리를 시스템 경로에 추가
YOLOV5_DIR = '/home/jetson/Downloads/detection-robot/yolov5'
sys.path.append(YOLOV5_DIR)

# YOLOv5 관련 모듈 import
from models.common import DetectMultiBackend
from utils.general import check_img_size
from utils.torch_utils import select_device

def load_model(weights_path, device='0'):
    """
    크로스 플랫폼 호환성을 고려한 YOLOv5 모델 로딩 함수
    
    Args:
        weights_path (str): 모델 가중치 파일 경로
        device (str): 사용할 장치 ('0': GPU, 'cpu': CPU)
    """
    # GPU 설정
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    # 장치 선택
    device = select_device(device)
    
    try:
        # 모델 로드
        model = DetectMultiBackend(
            weights=weights_path,
            device=device,
            dnn=False,
            data=None,
        )
        
        # 입력 이미지 크기 설정
        imgsz = check_img_size((640, 640), s=model.stride)
        
        # 모델을 평가 모드로 설정
        model.eval()
        
        # 워밍업
        model.warmup(imgsz=(1, 3, *imgsz))
        
        print(f"Model loaded successfully on {device}")
        return model, device
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

if __name__ == "__main__":
    # 모델 가중치 파일 경로 설정
    trained_model = './runs/train/exp/weights/best.pt'
    
    # 모델 로드
    model, device = load_model(trained_model)
    
    if model is not None:
        print("Model is ready for inference")
    else:
        print("Failed to load model")