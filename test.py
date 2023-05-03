import torch 
import cv2

img = cv2.imread("sh/4444444.jpg")
img.shape


model_yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model_yolov5.conf = 0.4

def detect_vehicle(img):
    results = model_yolov5(img)
    cars = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'] == 'car']
    if len(cars) == 0:
        return None
    biggest_car = cars.iloc[(cars['xmax'] - cars['xmin']).argmax()]
    return biggest_car


bb = detect_vehicle(img)
print("TTTTTTTTTTT " , bb)