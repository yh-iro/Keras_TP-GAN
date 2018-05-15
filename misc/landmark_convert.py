# -*- coding: utf-8 -*-
"""
store all pikled landmark info (for each image) in LANDMARKS_DIR into one dict 
object and save the dict as pikle
"""
import os
import glob
import pickle
import numpy as np

landmark_dir = 'LANDMARKS/DIR/'
out_dict = {}
curdir = os.getcwd()
try:
    os.chdir(landmark_dir)
    
    sessions = os.listdir('.')
    for session in sessions:
        print(session)
        out_dict[session] = {}
        
        multiview_dir = os.path.join(session, "multiview")
        out_dict[session]['multiview'] = {}
        
        subjects = os.listdir(multiview_dir)
        for subject in subjects:
            print("  " + subject)
            out_dict[session]['multiview'][subject] = {}
            
            subject_dir = os.path.join(multiview_dir, subject)
            rec_nums = os.listdir(subject_dir)
            
            for rec_num in rec_nums:
                print("    " + rec_num)
                out_dict[session]['multiview'][subject][rec_num] = {}
                
                rec_num_dir = os.path.join(subject_dir, rec_num)
                cam_labels = os.listdir(rec_num_dir)
                
                for cam_label in cam_labels:
                    out_dict[session]['multiview'][subject][rec_num][cam_label] = {}
                    
                    cam_label_dir = os.path.join(rec_num_dir, cam_label)
                    
                    
                    landmarks = glob.glob(os.path.join(cam_label_dir, '*.pkl'))
                    for landmark in landmarks:
                        
                        data_path = os.path.basename(landmark)[:-4]
                        
                        with open(landmark, 'rb') as f:
                            landmark_mat = pickle.load(f)
                    
                        out_dict[session]['multiview'][subject][rec_num][cam_label][data_path] = landmark_mat.astype(np.uint16)
    os.chdir(curdir)                    
except:
    os.chdir(curdir)
with open('landmarks.pkl', 'wb') as f:
    pickle.dump(out_dict, f)


               

    
        