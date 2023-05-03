import cv2
import numpy as np
import os
import torch
import time
from PIL import Image
import pandas as pd



from matplotlib import pyplot as plt
from util.post_processing import gen_3D_box,draw_3D_box,draw_2D_box
from net.bbox_3D_net import bbox_3D_net
from util.process_data import get_cam_data, get_dect2D_data

RADIAN_TO_DEGREE = 57.2958
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Yaw:
    def __init__(self):
        self.model_yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # self.model_yolov5.conf = 0.4
        self.model = bbox_3D_net((224,224,3))
        self.model.load_weights(r'model_saved/weights.h5')
        print(self.model.summary())


    def downsample_image(self , img):
        img = Image.fromarray(img)
        # Calculate new width based on desired height of 300 pixels
        width = int((300 / float(img.size[1])) * img.size[0])
        # Resize image while maintaining aspect ratio
        img = img.resize((width, 300))
        np_image = np.array(img)
        # Save the NumPy array as an image file
        return np_image

    def detect_main_car(self , img):
        results = self.model_yolov5(img)
        cars = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'] == 'car']
        vans = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'] == 'van']
        trucks = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'] == 'truck']
        cars = pd.concat([cars, vans, trucks], axis=0)
        cars.reset_index(drop=True, inplace=True)
        if len(cars) == 0:
            return None
        biggest_car = cars.iloc[(cars['xmax'] - cars['xmin']).argmax()]
        return biggest_car
    
    def predict_yaw(self , img):
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if (img.shape[0] < 280 or img.shape[1] < 280):
            img = self.downsample_image(img)

        bb = self.detect_main_car(img)
        if bb is  None:
            return None

        xmin, ymin, xmax, ymax = bb[['xmin', 'ymin', 'xmax', 'ymax']]
        
        patch = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        patch = cv2.resize(patch, (224, 224))
        patch = patch - np.array([[[103.939, 116.779, 123.68]]])
        patch = np.expand_dims(patch, 0)
        prediction = self.model.predict(patch)

        max_anc = np.argmax(prediction[2][0])
        anchors = prediction[1][0][max_anc]

        if anchors[1] > 0:
            angle_offset = np.arccos(anchors[0])
        else:
            angle_offset = -np.arccos(anchors[0])
        bin_num = prediction[2][0].shape[0]
        wedge = 2. * np.pi / bin_num
        theta_loc = angle_offset + max_anc * wedge

        theta = theta_loc + 0 # theta_ray
            # object's yaw angle
        yaw = (np.pi/2 - theta) * RADIAN_TO_DEGREE

        if yaw < 0 :
            yaw = 360 + yaw

        yaw += 90
        if yaw > 360:
            yaw -= 360
        return yaw 
    






if __name__ == "__main__":

    image_dir = 'sh/bb/'



    all_image = sorted(os.listdir(image_dir))


    yaw_obj = Yaw()




    # your code here


    for f in all_image:
        image_file = image_dir + f
        print(image_file)
        img = cv2.imread(image_file)

        start_time = time.time()
        yaww = yaw_obj.predict_yaw(img)
        end_time = time.time()

        run_time = end_time - start_time

    
        

        print("^^^^^^^^^^^^^^^^  ", yaww   , "   time= ", run_time)
    
    
