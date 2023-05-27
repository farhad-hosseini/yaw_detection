import cv2
import numpy as np
import os
import torch
import time
from PIL import Image
import pandas as pd

from keras.models import load_model


from net.bbox_3D_net import bbox_3D_net

RADIAN_TO_DEGREE = 57.2958
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Yaw:
    def __init__(self):
        self.model_yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # self.model_yolov5.conf = 0.4
        # self.model = bbox_3D_net((224,224,3))
        # self.model.load_weights(r'model_saved/weights.h5')
        self.model = load_model("model_saved/Yaw_model.h5")
        print("Keras model loaded !!")

        self.important_angles = [0 , 60 , 90  , 110 , 170 , 240 , 270 , 305 , 350]
        self.angle_names = ["left" , "front-left" , "front" , "front-right" , "right" , "back-right" , "back" , "back-left" , "left"]

        print(self.model.summary())


    def __change_initial_point_and_direction(self , yaw): # start from <-- and clockwise
        if 0 <= yaw <= 90 :
            return 90 - yaw
        
        return 450 - yaw
        


    def __downsample_image(self , img):
        img = Image.fromarray(img)
        # Calculate new width based on desired height of 300 pixels
        width = int((300 / float(img.size[1])) * img.size[0])
        # Resize image while maintaining aspect ratio
        img = img.resize((width, 300))
        np_image = np.array(img)
        # Save the NumPy array as an image file
        return np_image

    def __detect_main_car(self , img):
        results = self.model_yolov5(img)
        cars0 = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'] == 'car']
        vans = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'] == 'van']
        trucks = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'] == 'truck']
        cars = pd.concat([cars0, vans , trucks], axis=0)
        cars.reset_index(drop=True, inplace=True)
        if len(cars) == 0:
            return None
        biggest_car = cars.iloc[(cars['xmax'] - cars['xmin']).argmax()]

        if biggest_car['name'] == 'truck' or  biggest_car['name'] == 'van':
            center_bb_x1 = img.shape[1] * 0.25
            center_bb_x2 = img.shape[1] * 0.75
            center_bb_y1 = img.shape[0] * 0.25
            center_bb_y2 = img.shape[0] * 0.75

            car_center_x = (biggest_car['xmax'] - biggest_car['xmin']) / 2
            car_center_y = (biggest_car['ymax'] - biggest_car['ymin']) / 2

            if (center_bb_x1 <= car_center_x <= center_bb_x2)  and (center_bb_y1 <= car_center_y <= center_bb_y2) :
                biggest_car = biggest_car
            else:
                cars = cars0
                if len(cars) == 0:
                    return None
                biggest_car = cars.iloc[(cars['xmax'] - cars['xmin']).argmax()]
        return biggest_car
    


    
    
    def predict_yaw(self , img):
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if (img.shape[0] < 280 or img.shape[1] < 280):
            img = self.__downsample_image(img)

        bb = self.__detect_main_car(img)
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

        yaw = self.__change_initial_point_and_direction(yaw)
        return yaw 
    



    def __find_nearest(self , lst, num):
        if num is  None:
            return None

        nearest_num = lst[0]
        nearest_idx = 0
        min_diff = 99999999

        if lst[0] is not None:
            min_diff = abs(num - lst[0])
        
        
        for i in range(1, len(lst) ):
            if lst[i] is  None:
                continue
            diff = abs(num - lst[i])
            if diff <= min_diff:
                nearest_num = lst[i]
                nearest_idx = i
                min_diff = diff
                
        return (nearest_num, nearest_idx)
    



    def __rotate_180(self , angle):
        return (angle + 180) % 360
    



    def find_main_angle(self , angle , expectation_idx = 0): # expectation_idx is previous angle_idx

        nearest = self.__find_nearest(self.important_angles , angle)
        if nearest is None:
            return "---" , expectation_idx # nothing detected
        
        ang , idx = nearest
        angle2 = angle
        if 280 > abs(ang - self.important_angles[expectation_idx]) > 100: # dont mistake on 180 degree
            angle2 = self.__rotate_180(angle)
            ang , idx = self.__find_nearest(self.important_angles , angle2)
        if idx != expectation_idx + 1: 
            idx = expectation_idx
        return self.angle_names[idx] , idx , angle2
    




    
    def light_frame_selection(self , shots , main_angles , yaws , num_of_frame = 100):
        # delete non-car
        selected_frames = []
        selected_yaws = []
        selected_main_angles = []

        for i in range(len(main_angles) - 1 , -1 , -1):
            if main_angles[i] == "---" :
                del main_angles[i]
                del yaws[i]
                shots = np.delete(shots, i)


        for i in range(0 , len(self.important_angles)):
            if i != 0 and i != len(self.important_angles) - 1 : 
                angle_range = (self.important_angles[i + 1] -  self.important_angles[i - 1]) / 2 
            if i == 0 :
                angle_range = (self.important_angles[i + 1] +  self.important_angles[0]) / 2
            if i == len(self.important_angles) - 1 :
                angle_range = 360 - ((self.important_angles[i] +  self.important_angles[i - 1]) / 2)

            num_frame_in_one_mainAngle = angle_range * num_of_frame / 360
            # print(angle_names[i] , main_angles[i])

            indices = np.where(np.array(main_angles) == self.angle_names[i])[0]
            # print(indices)
            if i == 0 :
                for ii in range(len(indices) - 1, -1, -1):
                    if indices[ii] > len(main_angles) / 2 :
                        indices = np.delete(indices, ii)

            if i == len(self.important_angles) - 1 :
                for ii in range(len(indices) - 1, -1, -1):
                    if indices[ii] <= len(main_angles) / 2 :
                        indices = np.delete(indices, ii)

            shot1 = [shots[i] for i in indices]
            main_angles1 = [main_angles[i] for i in indices]
            yaws1 = [yaws[i] for i in indices]

            if (i % 2 == 0):
                jump = int(int(len(yaws1) / num_frame_in_one_mainAngle))
            else:
                jump = int(np.ceil(len(yaws1) / num_frame_in_one_mainAngle))

            selected_frames += shot1[::jump]
            selected_yaws += yaws1[::jump]
            selected_main_angles += main_angles1[::jump]




        if len(selected_frames) > num_of_frame:
            extra = len(selected_frames) - num_of_frame
            distance = len(selected_frames) // (extra + 1)
            start_index = len(selected_frames) - distance - 1 

            for _ in range(extra):
                del selected_frames[start_index]
                del selected_yaws[start_index]
                del selected_main_angles[start_index]
                start_index -= distance
                

        return selected_frames , selected_yaws , selected_main_angles







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
    
    
