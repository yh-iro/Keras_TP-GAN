# -*- coding: utf-8 -*-
"""
Add Keras_TP-GAN directory to PYTHONPATH
"""

import numpy as np
import pickle
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import os
from keras_tpgan.multipie_gen import Datagen

datagen = Datagen(dataset_dir='MULTI-PIE/DATASET_DIR/', landmarks_dict_file='../landmarks.pkl')
files = datagen.train_list
i_file = 6
i_num = 0
show_crop = False

#nose = 30

eye_y = 40/128
mouth_y = 88/128
img_size = 128

i_right_eye = [37, 38, 40, 41]
i_left_eye = [43, 44, 46, 47]
i_mouth = [48, 54]


def load(i):
    global img, crop_img, preds, crop_preds
    print(files[i])        
    img = cv2.imread(os.path.join(datagen.dataset_dir, files[i] + '.jpg'))    
    
    preds = datagen.load_landmarks(files[i])


    reye, leye, nose, mouth = get_keypoint(preds)    


    vec_mouth2reye = reye - mouth
    vec_mouth2leye = leye - mouth
    # angle reye2mouth against leye2mouth
    phi = np.arccos(vec_mouth2reye.dot(vec_mouth2leye) / (np.linalg.norm(vec_mouth2reye) * np.linalg.norm(vec_mouth2leye)))/np.pi * 180
    
    if phi < 15: # consider the profile image is 90 deg.

        # in case of 90 deg. set invisible eye with copy of visible eye.
        eye_center = (reye + leye) / 2
        if nose[0] > eye_center[0]:
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
    rot_preds = np.transpose(rotmat[:, :2].dot(np.transpose(preds)) + np.repeat(rotmat[:, 2].reshape((2,1)), preds.shape[0], axis=1))
    
    crop_size = int((mouth[1] - reye[1])/(mouth_y - eye_y))
    crop_up = int(reye[1] - crop_size * eye_y)
    crop_left = int((reye[0] + leye[0]) / 2 - crop_size / 2)
    
    crop_img = rot_img[crop_up:crop_up+crop_size, crop_left:crop_left+crop_size]
    crop_preds = rot_preds - np.array([crop_left, crop_up])
    
    crop_img = cv2.resize(crop_img, (img_size, img_size))
    crop_preds *= img_size / crop_size
    
def get_keypoint(points):
    reye = np.average(np.array((points[37], points[38], points[40], points[41])), axis=0)
    leye = np.average(np.array((points[43], points[44], points[46], points[47])), axis=0)
    mouth = np.average(np.array((points[48], points[54])), axis=0)
    nose_tip = points[30]
    
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
    
    return reye, leye, nose_tip, mouth


load(i_file)
while True:
    
    if show_crop:
        show_img = np.copy(crop_img)
        show_preds = crop_preds
    else:
        show_img = np.copy(img)
        show_preds = preds 

    EYE_H = 40; EYE_W = 40;
    NOSE_H = 32; NOSE_W = 40;
    MOUTH_H = 32; MOUTH_W = 48;

    leye_points = show_preds[36:42]
    leye_center = (np.max(leye_points, axis=0) + np.min(leye_points, axis=0)) / 2
    leye_left = int(leye_center[0] - EYE_W / 2)
    leye_up = int(leye_center[1] - EYE_H / 2)
    leye_img = show_img[leye_up:leye_up + EYE_H, leye_left:leye_left + EYE_W].copy()
    
    reye_points = show_preds[42:48]
    reye_center = (np.max(reye_points, axis=0) + np.min(reye_points, axis=0)) / 2
    reye_left = int(reye_center[0] - EYE_W / 2)
    reye_up = int(reye_center[1] - EYE_H / 2)
    reye_img = show_img[reye_up:reye_up + EYE_H, reye_left:reye_left + EYE_W].copy()
    
    nose_points = show_preds[31:36]
    nose_center = (np.max(nose_points, axis=0) + np.min(nose_points, axis=0)) / 2
    nose_left = int(nose_center[0] - NOSE_W / 2)
    nose_up = int(nose_center[1] - 10 - NOSE_H / 2)
    nose_img = show_img[nose_up:nose_up + NOSE_H, nose_left:nose_left + NOSE_W].copy()
    
    plt.imshow(cv2.cvtColor(nose_img, cv2.COLOR_RGB2BGR))
    plt.show()
    
    mouth_points = show_preds[48:60]
    mouth_center = (np.max(mouth_points, axis=0) + np.min(mouth_points, axis=0)) / 2
    mouth_left = int(mouth_center[0] - MOUTH_W / 2)
    mouth_up = int(mouth_center[1] - MOUTH_H / 2)
    mouth_img = show_img[mouth_up:mouth_up + MOUTH_H, mouth_left:mouth_left + MOUTH_W].copy()

        
    for pred in show_preds:
        #cv2.circle(show_img, (int(pred[0]), int(pred[1])), 2, (255, 255, 255), -1)
        pass

    #cv2.circle(show_img, (int(show_preds[i_num][0]), int(show_preds[i_num][1])), 3, (0, 255, 255), -1)
    
    reye, leye, nose, mouth = get_keypoint(show_preds)
    
    cv2.circle(show_img, (int(reye[0]), int(reye[1])), 3, (255, 0, 0), -1)
    cv2.circle(show_img, (int(leye[0]), int(leye[1])), 3, (255, 0, 0), -1)
    cv2.circle(show_img, (int(nose[0]), int(nose[1])), 3, (255, 0, 0), -1)
    cv2.circle(show_img, (int(mouth[0]), int(mouth[1])), 3, (255, 0, 0), -1)
    
    
    
    
    cv2.imshow("", show_img)
    key = cv2.waitKeyEx(10)
  
  
    if key == 81 or key == 2490368:
        i_file -= 1
        if i_file < 0:
            i_file = 0
        load(i_file)
        print(i_file)                
    elif key == 83 or key == 2621440:
        i_file += 1
        if i_file >= len(files):
            i_file = len(files)-1

        load(i_file)
        print(i_file)
    if key == 82 or key == 2555904:
        i_num += 1
        if i_num >= preds.shape[0]:
           i_num = preds.shape[0] - 1 
        print(i_num)
    if key == 84 or key == 2424832:
        i_num -= 1
        if i_num < 0:
            i_num = 0
        print(i_num)
    if key == ord('c'):
        show_crop = not show_crop
    if key == ord('s'):
        cv2.imwrite("{}.png".format(i_file), show_img)
        
    if key == ord('q'):
        break
        
    
    
    
cv2.destroyAllWindows()
