import serial
import time
import pathlib

from third_class import ModelLoader, RealSenseCamera, ObjectDetector, Visualizer

class RobotController:
    def __init__(self, port='/dev/ttyACM0', baudrate=57600):
        CATEGORY_NAMES = ['ai', 'awear', 'imr', 'gist']
        WEIGHTS_PATH = './runs/train/6_s_300epoch/weights/best.pt' # 0.443s
        MODEL_CONFIDENCE = 0.25 # Default NMS confidence threshold
        MODEL_IOU = 0.45 # NMS IoU threshold

        self.direction = None

        start_time = time.time()
        # self.initSerial(port, baudrate)
        self.initModelLoader(weights_path=WEIGHTS_PATH, confidence=MODEL_CONFIDENCE, iou=MODEL_IOU)
        print(f"{time.time()-start_time:.3f}s Model loaded and warmed up")
        self.initRealSenseCamera()
        print(f"{time.time()-start_time:.3f}s Camera initialized")
        self.initObjectDetector(CATEGORY_NAMES)
        self.initVisualizer()
        print(f"{time.time()-start_time:.3f}s Starting detection loop")

    def initSerial(self, port, baudrate):
        self.serial = serial.Serial(port=port, baudrate=baudrate)

    def initModelLoader(self, weights_path, confidence, iou):
        self.model_loader = ModelLoader(weights_path)
        self.model_loader.load_model()
        self.model_loader.model.conf = confidence
        self.model_loader.model.iou = iou
        self.model_loader.warm_up()

    def initRealSenseCamera(self):
        self.camera = RealSenseCamera()
        self.camera.setup()

    def initObjectDetector(self, category_names):
        self.detector = ObjectDetector(self.model_loader.model, category_names)

    def initVisualizer(self):
        self.visualizer = Visualizer()




    # Main Function
    def initializeDirecition(self):
        print("initialize direction")
        pass

    # Complete
    def setDirectionOfTarget(self, target):
        print("set direction of", target)
        color_image, depth_image = self.camera.get_frames()
        detections = self.detector.detect(color_image, depth_image)
        
        if self.isTargetDetected(target, detections):
            if self.isTargetRight(target, detections):
                self.setDirection('R')
            else:
                self.setDirection('L')


    def navigateToTarget(self, target):
        print("navigate to", target)
        pass

    def moveForwardBackward(self, step):
        print("move forward", step, "step, backward", step, "step")
        pass

    def searchTarget(self, target):
        print("search", target)
        pass


    # Util Function
    # Complete
    def isTargetDetected(self, target, detections):
        for detection in detections:
            if detection['class_name'].lower() == target.lower():
                return True
        print(f"Target {target} not detected")
        return False

    # Complete
    def isTargetRight(self, target, detections):
        # isTargetDetected이후에 실행되므로, detection['class_name']이 없는 경우는 없을 것이다.
        for detection in detections:
            if detection['class_name'].lower() == target.lower():
                return detection['angle'] > 0

    # Complete
    def setDirection(self, direction):
        print("self.direciton is", direction)
        self.direction = direction

    def isTargetAngleExceeded(self, target, detection):
        pass
    def sendCommand(self, command):
        if self.serial:
            self.serial.write(command.encode('utf-8'))
            print("send serial command", command)
        else:
            print("can't send serial command", command)

    def hasReceivedCommandComplete(self, command):
        if not self.serial:
            print("Serial connect failed")
            return True
            
        if self.serial.in_waiting:
            response = self.serial.readline().decode('utf-8').strip()
            print("Serial reponse is", response)
            return response == f"{command}_COMPLETE"
                
        return False

    def isWithinTargetRange(self, range):
        pass
    

def main():
    pathlib.WindowsPath = pathlib.PosixPath
    PORT = '/dev/ttyACM0'
    BAUDRATE = 57600
    RED = 'RED'
    GOAL = 'GIST'

    robot = RobotController(port=PORT, baudrate=BAUDRATE)
    robot.initializeDirecition()
    robot.setDirectionOfTarget(target='IMR')
    # robot.setDirectionOfTarget(target=RED)
    # robot.navigateToTarget(target=RED)
    # robot.moveForwardBackward(step=3)
    # robot.searchTarget(target=GOAL)
    # robot.navigateToTarget(target=GOAL)

if __name__ == "__main__":
    main()