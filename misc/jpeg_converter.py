# -*- coding: utf-8 -*-
"""
this program convert image files of MULTI-PIE to jpeg with same directory structures.
"""
import os
import glob
import numpy as np
import cv2


src_dataset_dir = 'MULTIPIE/DATASET_DIR' # eg. 'MULTIPIE_COMBINED/Multi-Pie/data'
dest_dataset_dir = 'SOME/DEST/DIR'
landmark_dir = 'LANDMARK/DIR/CREATED/BY/extract_from_multipie.py/'
ANGLE_DIR = {
        -90: "11_0",
        -75: "12_0",
        -60: "09_0",
        -45: "08_0",
        -30: "13_0",
        -15: "14_0",
        0: "05_1",
        15: "05_0",
        30: "04_1",
        45: "19_0",
        60: "20_0",
        75: "01_0",
        90: "24_0",
        }
        
curdir = os.getcwd()
try:
    os.chdir(landmark_dir)
    
    sessions = os.listdir('.')
    for session in sessions:
        print(session)
        
        multiview_dir = os.path.join(session, "multiview")
        
        subjects = os.listdir(multiview_dir)
        for subject in subjects:
            print("  " + subject)
            
            subject_dir = os.path.join(multiview_dir, subject)
            rec_nums = os.listdir(subject_dir)
            
            for rec_num in rec_nums:
                print("    " + rec_num)
                
                rec_num_dir = os.path.join(subject_dir, rec_num)
                cam_labels = os.listdir(rec_num_dir)
                for cam_label in cam_labels:
                    
                    cam_label_dir = os.path.join(rec_num_dir, cam_label)
                    
                    dest_cam_label_dir = os.path.join(dest_dataset_dir, cam_label_dir)
                    os.makedirs(dest_cam_label_dir, exist_ok=True)
                    
                    landmarks = glob.glob(os.path.join(cam_label_dir, '*.pkl'))
                    for landmark in landmarks:
                        
                        data_path = landmark[:-4]
                        
                        src_img_path = os.path.join(src_dataset_dir, data_path + '.png')
                        dest_img_path = os.path.join(dest_dataset_dir, data_path + '.jpg')
                        cv2.imwrite(dest_img_path, cv2.imread(src_img_path))
    os.chdir(curdir)                    
except:
    os.chdir(curdir)