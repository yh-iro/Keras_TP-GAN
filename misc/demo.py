# -*- coding: utf-8 -*-
"""
This program provides UI to input keypoints of profile image
 and generate frontal image using TP-GAN

Add Keras_TP-GAN directory to PYTHONPATH

"""

import numpy as np
import pickle
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import os
from keras_tpgan.tpgan import TPGAN

IMG_PATH = 'INPUT/PROFILE_IMG.jpg'
GENERATOR_WEIGHTS_FILE = 'SOME/WEIGHTS_FILE.hdf5'

EYE_H = 40; EYE_W = 40;
NOSE_H = 32; NOSE_W = 40;
MOUTH_H = 32; MOUTH_W = 48;

#nose = 30

eye_y = 40/128
mouth_y = 88/128
img_size = 128

i_right_eye = [37, 38, 40, 41]
i_left_eye = [43, 44, 46, 47]
i_mouth = [48, 54]


img = cv2.imread(IMG_PATH)    

def mouse_key_point_event(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(show_img, (x, y), 3, (255, 0, 0), -1)
        key_points.append((x, y))
        if len(key_points) == 1:
            print('input reye')
        elif len(key_points) == 2:
            print('input nose_tip')
        elif len(key_points) == 3:
            print('input mouth')




cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("img", mouse_key_point_event)


key_points = []
print('input leye')
show_img = np.copy(img)
while len(key_points)<4:

    
    cv2.imshow("img", show_img)
    key = cv2.waitKeyEx(10)

  

    if key == ord('q'):
        break
    
cv2.destroyAllWindows()


reye, leye, nose_tip, mouth = key_points    
reye = np.array(reye)
leye = np.array(leye)
nose_tip = np.array(nose_tip)
mouth = np.array(mouth)


vec_mouth2reye = reye - mouth
vec_mouth2leye = leye - mouth
# angle reye2mouth against leye2mouth
phi = np.arccos(vec_mouth2reye.dot(vec_mouth2leye) / (np.linalg.norm(vec_mouth2reye) * np.linalg.norm(vec_mouth2leye)))/np.pi * 180

if phi < 15: # consider the profile image is 90 deg.

    # in case of 90 deg. set invisible eye with copy of visible eye.
    eye_center = (reye + leye) / 2
    if nose_tip[0] > eye_center[0]:
        leye = reye
    else:
        reye = leye

 # calc angle eyes against horizontal as theta
if np.array_equal(reye, leye) or phi < 38: # in case of 90 deg. avoid rotation
    theta = 0
else: 
    vec_leye2reye = reye - leye
    if vec_leye2reye[0] < 0:
        vec_leye2reye = -vec_leye2reye
    theta = np.arctan(vec_leye2reye[1]/vec_leye2reye[0])/np.pi*180

imgcenter = (img.shape[1]/2, img.shape[0]/2)
rotmat = cv2.getRotationMatrix2D(imgcenter, theta, 1)
rot_img = cv2.warpAffine(img, rotmat, (img.shape[1], img.shape[0])) 

crop_size = int((mouth[1] - reye[1])/(mouth_y - eye_y))

crop_up = int(reye[1] - crop_size * eye_y)
if crop_up < 0:
    crop_up = 0
    
crop_down = crop_up + crop_size
if crop_down > rot_img.shape[0]:
    crop_down = rot_img.shape[0]
    
crop_left = int((reye[0] + leye[0]) / 2 - crop_size / 2)
if crop_left < 0:
    crop_left = 0
    
crop_right = crop_left + crop_size
if crop_right > rot_img.shape[1]:
    crop_right = rot_img.shape[1]

crop_img = rot_img[crop_up:crop_down, crop_left:crop_right]

crop_img = cv2.resize(crop_img, (img_size, img_size))


global left_up, right_down
left_up = None; right_down = None

def mouse_key_rect_event(event, x, y, flags, param):
    global left_up, right_down
    
    if event == cv2.EVENT_LBUTTONUP:
        key_rects.append((left_up, right_down))
        if len(key_rects) == 1:
            print('input reye')
        elif len(key_rects) == 2:
            print('input nose_tip')
        elif len(key_rects) == 3:
            print('input mouth')

    elif event == cv2.EVENT_MOUSEMOVE:    
        left_up = (int(x-w/2), int(y-h/2))
        right_down = (int(x+w/2), int(y+h/2))


cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("img", mouse_key_rect_event)
key_rects = []
print('input leye')
tpgan= None
while True:
    
    if len(key_rects) < 2:
        w = EYE_W
        h = EYE_H
    elif len(key_rects) == 2:
        w = NOSE_W
        h = NOSE_H
    else:
        w = MOUTH_W
        h = MOUTH_H

    
    show_img = np.copy(crop_img)
    
    if left_up is not None and right_down is not None:
        cv2.rectangle(show_img, left_up, right_down, (255,255,255), 1)
    
    cv2.imshow("img", show_img)
    key = cv2.waitKeyEx(10)

    if key == ord('s'):
        cv2.imwrite("crop_img.jpg", crop_img)
        if len(key_rects) == 4:
            leye_img = crop_img[key_rects[0][0][1]:key_rects[0][1][1], key_rects[0][0][0]:key_rects[0][1][0]]
            reye_img = crop_img[key_rects[1][0][1]:key_rects[1][1][1], key_rects[1][0][0]:key_rects[1][1][0]]
            nose_img = crop_img[key_rects[2][0][1]:key_rects[2][1][1], key_rects[2][0][0]:key_rects[2][1][0]]
            mouth_img = crop_img[key_rects[3][0][1]:key_rects[3][1][1], key_rects[3][0][0]:key_rects[3][1][0]]
            cv2.imwrite("leye.jpg", leye_img)
            cv2.imwrite("reye.jpg", reye_img)
            cv2.imwrite("nose.jpg", nose_img)
            cv2.imwrite("mouth.jpg", mouth_img)
            
    if key == ord('p'):
        global pred_faces, pred_faces64, pred_faces32, pred_leyes, pred_reyes, pred_noses, pred_mouthes
        x_z = np.random.normal(scale=0.02, size=(1, 100))
        x_face = (cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB).astype(np.float)/255)[np.newaxis,:]
        x_leye = (cv2.cvtColor(leye_img, cv2.COLOR_BGR2RGB).astype(np.float)/255)[np.newaxis,:]
        x_reye = (cv2.cvtColor(reye_img, cv2.COLOR_BGR2RGB).astype(np.float)/255)[np.newaxis,:]
        x_nose = (cv2.cvtColor(nose_img, cv2.COLOR_BGR2RGB).astype(np.float)/255)[np.newaxis,:]
        x_mouth = (cv2.cvtColor(mouth_img, cv2.COLOR_BGR2RGB).astype(np.float)/255)[np.newaxis,:]
        
        if tpgan is None:
            
            tpgan = TPGAN(generator_weights=GENERATOR_WEIGHTS_FILE)
        [pred_faces, pred_faces64, pred_faces32, pred_leyes, pred_reyes, pred_noses, pred_mouthes]\
        =tpgan.generate([x_face, x_leye, x_reye, x_nose, x_mouth, x_z])
        break

    if key == ord('q'):
        break
    
cv2.destroyAllWindows()   
