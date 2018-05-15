# -*- coding: utf-8 -*-
"""
this program extract face landmarks

git clone https://github.com/1adrianb/face-alignment.git
and add above project dir to PYTHONPATH
"""

import face_alignment
import cv2
import os
import pickle

ANGLE_DIR = {
        "-90": "11_0",
        "-75": "12_0",
        "-60": "09_0",
        "-45": "08_0",
        "-30": "13_0",
        "-15": "14_0",
        "0": "05_1",
        "15": "05_0",
        "30": "04_1",
        "45": "19_0",
        "60": "20_0",
        "75": "01_0",
        "90": "24_0",
        }

MULTIPIE_DATA_DIR = 'MULTIPIE_COMBINED/Multi-Pie/data'
OUT_DIR = 'Multi-Pie_build/feat'

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2halfD, enable_cuda=True, flip_input=False)

feat_dict = {}

os.chdir(MULTIPIE_DATA_DIR)
sessions = os.listdir('.')
for session in sessions:
    print(session)
    multiview_dir = os.path.join(session, "multiview")
    
    subjects = os.listdir(multiview_dir)
    
    for subject in subjects:
        print("  " + subject)
        subject_dir = os.path.join(multiview_dir, subject)
        #rec_nums = os.listdir(subject_dir)
        rec_nums = ["01"]
        
        for rec_num in rec_nums:
            print("    " + rec_num)
            rec_num_dir = os.path.join(subject_dir, rec_num)
            
            for cam_label in ANGLE_DIR.values():
                
                cam_label_dir = os.path.join(rec_num_dir, cam_label)
                out_cam_label_dir = os.path.join(OUT_DIR, cam_label_dir)
                os.makedirs(out_cam_label_dir, exist_ok=True)
                
                images = os.listdir(cam_label_dir)
                for image in images:
                    
                    out_feat = os.path.join(out_cam_label_dir, os.path.splitext(image)[0] + '.pkl')
                    if os.path.exists(out_feat) and os.path.getsize(out_feat) != 0:
                        continue
                    
                    image_path = os.path.join(cam_label_dir, image)
                    img = cv2.imread(image_path)
                    landmarks = fa.get_landmarks(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if landmarks is None:
                        print("No face detected: {}".format(image_path))
                        continue
                    
                    with open(out_feat, 'wb') as f:        
                        pickle.dump(landmarks[0], f)                    
                    

'''
with open(OUT_FEAT_FILE, 'rb') as f:        
    load_dict = pickle.load(f)
'''