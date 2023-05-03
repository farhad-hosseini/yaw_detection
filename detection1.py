import cv2
import numpy as np
import os
import torch
from matplotlib import pyplot as plt
from util.post_processing import gen_3D_box,draw_3D_box,draw_2D_box
from net.bbox_3D_net import bbox_3D_net
from util.process_data import get_cam_data, get_dect2D_data

RADIAN_TO_DEGREE = 57.2958



model_yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model_yolov5.conf = 0.4

def detect_vehicle(img):
    results = model_yolov5(img)
    cars = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'] == 'car']
    if len(cars) == 0:
        return None
    biggest_car = cars.iloc[(cars['xmax'] - cars['xmin']).argmax()]
    return biggest_car




os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Construct the network
model = bbox_3D_net((224,224,3))

model.load_weights(r'model_saved/weights.h5')

image_dir = 'sh/bb/'
calib_file = 'F:/dataset/kitti/testing/calib.txt'
box2d_dir = 'F:/dataset/kitti/testing/label_2/'

classes = ['Car','Van','Truck','Pedestrian','Person_sitting','Cyclist','Tram']
cls_to_ind = {cls:i for i,cls in enumerate(classes)}

dims_avg = np.loadtxt(r'dataset/voc_dims.txt',delimiter=',')


# print(model.summary())

all_image = sorted(os.listdir(image_dir))
# np.random.shuffle(all_image)

# cam_to_img = get_cam_data(calib_file)
# fx = cam_to_img[0][0]
# u0 = cam_to_img[0][2]
# v0 = cam_to_img[1][2]

for f in all_image:
    image_file = image_dir + f
    # box2d_file = box2d_dir + f.replace('png', 'txt')
    print(image_file)
    img = cv2.imread(image_file)

    bb = detect_vehicle(img)
    print("TTTTTTTTTTT " , bb)
    xmin, ymin, xmax, ymax = bb[['xmin', 'ymin', 'xmax', 'ymax']]
    
    patch = img[int(ymin):int(ymax), int(xmin):int(xmax)]
    patch = cv2.resize(patch, (224, 224))
    patch = patch - np.array([[[103.939, 116.779, 123.68]]])
    patch = np.expand_dims(patch, 0)
    prediction = model.predict(patch)

        # compute dims
        # dims = dims_avg[cls_to_ind[cls]] + prediction[0][0]

        # Transform regressed angle
    box2d_center_x = (xmin + xmax) / 2.0
        # Transfer arctan() from (-pi/2,pi/2) to (0,pi)
        # theta_ray = np.arctan(fx /(box2d_center_x - u0))
        # if theta_ray<0:
        #     theta_ray = theta_ray+np.pi

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
    yaw = np.pi/2 - theta
    plt.imshow(img)

    print("^^^^^^^^^^^^^^^^  ", yaw * RADIAN_TO_DEGREE )
    #     points2D = gen_3D_box(yaw, dims, cam_to_img, box_2D)
    #     draw_3D_box(img, points2D)

    # for cls,box in box2d_reserved:
    #     draw_2D_box(img,box)

    # cv2.imshow(f, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('output/'+ f.replace('png','jpg'), img)
