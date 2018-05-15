# -*- coding: utf-8 -*-
"""
this program uploads all jpeg file to Google Clound Storage
"""
import os
import glob
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'YOUR/GOOGLE_APPLICATION_CREDENTIALS.json'
storage_client = storage.Client()
bucket = storage_client.get_bucket("BUCKET_NAME")

src_dataset_dir = 'MULTI-PIE/DATASET/DIR/'
dest_dataset_dir = 'DEST/DIR/ON/BUCKET/'

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
    os.chdir(src_dataset_dir)
    
    sessions = os.listdir('.')
    for session in sessions:
        print(session)
        
        if os.path.isfile(session):
            continue
        
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
                    
                    landmarks = glob.glob(os.path.join(cam_label_dir, '*.jpg'))
                    for landmark in landmarks:
                        
                        data_path = landmark[:-4]
                        
                        img_path = data_path + '.jpg'
                        blob = bucket.blob(os.path.join(dest_dataset_dir, img_path).replace('\\', '/'))
                        blob.upload_from_filename(img_path)
    os.chdir(curdir)                    
except:
    os.chdir(curdir)
    
               
