# -*- coding: utf-8 -*-
"""
@author: yhiro
"""

import numpy as np
import cv2
import os
import random
from skimage.transform import resize
from skimage.color import rgb2gray
import pickle
import tensorflow as tf
from keras.utils import np_utils
from PIL import Image
import threading
from concurrent.futures import ThreadPoolExecutor

# image size to provide to TP-GAN
IMG_H, IMG_W = 128, 128
# subjects of MULTI-PIE
NUM_SUBJECTS = 346

# dictionary to map capture angle to MULTI-PIE dir name
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
# dictionary to map capture MULTI-PIE dir name to angle 
DIR_ANGLE = {}
for angle in ANGLE_DIR.keys():
    DIR_ANGLE[ANGLE_DIR[angle]] = angle

# size of cropped part image
EYE_H, EYE_W = 40, 40
NOSE_H, NOSE_W = 32, 40
MOUTH_H, MOUTH_W = 32, 48

# average part position of angle 0 deg images
LEYE_Y, LEYE_X = 40, 42
REYE_Y, REYE_X = 40, 86
NOSE_Y, NOSE_X = 71, 64
MOUTH_Y, MOUTH_X = 87, 64

class Datagen():
    """
    this class provides data generator of MULTI-PIE dataset.
    """
    
    def __init__(self, dataset_dir='MULTI-PIE/DIR/', landmarks_dict_file='../landmarks.pkl',
                 datalist_dir='datalist/', mirror_to_one_side=True,
                 min_angle=-60, max_angle=60, include_frontal=False,
                 face_trembling_range=0, valid_count=4, workers=16):
        """
        Initializer
        
        Args:
            dataset_dir (str): jpg converted MULTI-PIE data dir; parent dir of session dirs.
                               this directory can be created by misc/jpeg_converter.py
            landmarks_dict_file (str): pikled dict file. the structure of the dict is same as MULTI-PIE directories
                                       this dict file can be created by misc/landmark_convert.py
            datalist_dir (str): output dir for datalist file. datalist stores the list of train and valid image files.
            min_angle (str): min pose angle. must be one of [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
            max angle (str): max pose angle. must be one of [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
            include_frontal (bool): if False, return data doesn't include frontal (0 deg) image.
            face_trembling_range (int): random noise range (-val to + val) for face cropping position.
            valid_count (int): data count for validation dataset
            workers (int): worker count for multi-threading.
        """
        
        self.dataset_dir = dataset_dir
        if not tf.gfile.Exists(landmarks_dict_file):
            raise Exception('landmarks dict file doesnt exsit. target file: {}'.format(landmarks_dict_file))
        with Open(landmarks_dict_file, 'rb') as f:
            self.landmarks_dict = pickle.load(f)
        
        self.datalist_file = os.path.join(datalist_dir, 'datalist_{}_{}_front_{}.pkl'.format(min_angle, max_angle, include_frontal))
        if tf.gfile.Exists(self.datalist_file):
            with Open(self.datalist_file, 'rb') as f:
                self.datalist = pickle.load(f)
        else:
            self.datalist = self.create_datalist(min_angle, max_angle, include_frontal)
            tf.gfile.MakeDirs(datalist_dir)
            with Open(self.datalist_file, 'wb') as f:
                pickle.dump(self.datalist, f)
            
                
        self.train_list = self.datalist[valid_count:]
        self.train_cursor = random.randint(0, len(self.train_list)-1)
        self.valid_list = self.datalist[:valid_count]
        self.valid_cursor = 0
        self.mirror_to_one_side = mirror_to_one_side
        self.face_trembling_range = face_trembling_range
        self.workers = workers
        if self.workers > 1:
            self.thread_pool_executor = ThreadPoolExecutor(max_workers=workers)
        self.lock = threading.Lock()
        
    def __del__(self):
        if self.workers > 1:
            try:
                self.thread_pool_executor.shutdown()
            except:
                pass
                    
    def batch_data(self, datalist, cursor, batch_size = 16):
        """
        create mini-batch from datalist and cursor index
        
        Args:
            datalist (list): list of data file path
            cursor (int): current index cursor on the datalist
            batch_size (int): batch size of return data
            
        Returns:
            tuple of list of mini-batch data file path and updated cursor index
        """
            
        ret_list = []
        for i in range(batch_size):
            ret_list.append(datalist[(cursor + i)%len(datalist)])
            
        ret_cursor = (cursor + batch_size) % len(datalist)
            
        return ret_list, ret_cursor
    
    def create_datalist(self, min_angle, max_angle, include_frontal=False, shuffle=True):
        """
        create datalist; list of target image file path which saticefies arg params.
        this function also save created datalist and load datalist if already exists.
        
        Args:
            min_angle (str): min pose angle. must be one of [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
            max angle (str): max pose angle. must be one of [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
            include_frontal (bool): if False, return data doesn't include frontal (0 deg) image.
            shuffle (bool): if True, shuffle order of return list
            
        Returns:
            created or loaded datalist.
        """
        
        datalist = []
        
        cam_labels = []
        for angle in range(min_angle, max_angle+1, 15):
            if include_frontal or angle != 0:
                cam_labels.append(ANGLE_DIR[angle])
        
        curdir = os.getcwd()
        
        try:
            sessions = self.landmarks_dict.keys()
            for session in sessions:
                print(session)
                
                subjects = self.landmarks_dict[session]['multiview'].keys()
                for subject in subjects:
                    print("  " + subject)
                    rec_nums = self.landmarks_dict[session]['multiview'][subject].keys()
                    
                    for rec_num in rec_nums:
                        print("    " + rec_num)
                        
                        for cam_label in cam_labels:
                            
                            landmarks = self.landmarks_dict[session]['multiview'][subject][rec_num][cam_label].keys()
                            for landmark in landmarks:
                                
                                data_path = os.path.join(session, 'multiview', subject, rec_num, cam_label, landmark)
                                datalist.append(data_path)
            os.chdir(curdir)                    
        except:
            os.chdir(curdir)
                            
        if shuffle:
            random.shuffle(datalist)
        return datalist

    def load_landmarks(self, data_path):
        
        try:
            session, multiview, subject, rec_num, cam_label, landmark = data_path.split(os.sep)
        except Exception as e:
            print(e)
            print(data_path)
        
        return self.landmarks_dict[session][multiview][subject][rec_num][cam_label][landmark]
    
    def crop(self, image, landmarks, angle, size=128):
        """
        crop resized face and each part from target image.
        
        Args:
            image (np.array): target image
            landmarks (np.array): landmarks positions in the target image
            angle (int): camera angle of the target image
            size (int): cropping size
            
        Returns:
            tuple of cropped face and each part images
        """
        eye_y = 40/128
        mouth_y = 88/128
        
        reye = np.average(np.array((landmarks[37], landmarks[38], landmarks[40], landmarks[41])), axis=0)
        leye = np.average(np.array((landmarks[43], landmarks[44], landmarks[46], landmarks[47])), axis=0)
        mouth = np.average(np.array((landmarks[48], landmarks[54])), axis=0)
        nose_tip = landmarks[30]
        
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
        
        imgcenter = (image.shape[1]/2, image.shape[0]/2)
        rotmat = cv2.getRotationMatrix2D(imgcenter, theta, 1)
        rot_img = cv2.warpAffine(image, rotmat, (image.shape[1], image.shape[0])) 
        rot_landmarks = np.transpose(rotmat[:, :2].dot(np.transpose(landmarks)) + np.repeat(rotmat[:, 2].reshape((2,1)), landmarks.shape[0], axis=1))
        
        rot_reye = np.average(np.array((rot_landmarks[37], rot_landmarks[38], rot_landmarks[40], rot_landmarks[41])), axis=0)
        rot_leye = np.average(np.array((rot_landmarks[43], rot_landmarks[44], rot_landmarks[46], rot_landmarks[47])), axis=0)
        rot_mouth = np.average(np.array((rot_landmarks[48], rot_landmarks[54])), axis=0)
        
        crop_size = int((rot_mouth[1] - rot_reye[1])/(mouth_y - eye_y) + 0.5)
        crop_up = int(rot_reye[1] - crop_size * eye_y + 0.5)
        crop_left = int((rot_reye[0] + rot_leye[0]) / 2 - crop_size / 2 + 0.5)
        
        crop_img = rot_img[crop_up:crop_up+crop_size, crop_left:crop_left+crop_size]
        crop_landmarks = rot_landmarks - np.array([crop_left, crop_up])
        
        crop_img = cv2.resize(crop_img, (size, size))
        crop_landmarks *= size / crop_size
    
        leye_points = crop_landmarks[42:48]
        leye_center = (np.max(leye_points, axis=0) + np.min(leye_points, axis=0)) / 2
        leye_left = int(leye_center[0] - EYE_W / 2 + 0.5)
        leye_up = int(leye_center[1] - EYE_H / 2 + 0.5)
        leye_img = crop_img[leye_up:leye_up + EYE_H, leye_left:leye_left + EYE_W]
        
        reye_points = crop_landmarks[36:42]
        reye_center = (np.max(reye_points, axis=0) + np.min(reye_points, axis=0)) / 2
        reye_left = int(reye_center[0] - EYE_W / 2 + 0.5)
        reye_up = int(reye_center[1] - EYE_H / 2 + 0.5)
        reye_img = crop_img[reye_up:reye_up + EYE_H, reye_left:reye_left + EYE_W]
        
        nose_points = crop_landmarks[31:36]
        nose_center = (np.max(nose_points, axis=0) + np.min(nose_points, axis=0)) / 2
        nose_left = int(nose_center[0] - NOSE_W / 2 + 0.5)
        nose_up = int(nose_center[1] - 10 - NOSE_H / 2 + 0.5)
        nose_img = crop_img[nose_up:nose_up + NOSE_H, nose_left:nose_left + NOSE_W]
        
        mouth_points = crop_landmarks[48:60]
        mouth_center = (np.max(mouth_points, axis=0) + np.min(mouth_points, axis=0)) / 2
        mouth_left = int(mouth_center[0] - MOUTH_W / 2 + 0.5)
        mouth_up = int(mouth_center[1] - MOUTH_H / 2 + 0.5)
        mouth_img = crop_img[mouth_up:mouth_up + MOUTH_H, mouth_left:mouth_left + MOUTH_W]
        
        if self.face_trembling_range != 0:
            offset_x = random.randint(-self.face_trembling_range, self.face_trembling_range)
            offset_y = random.randint(-self.face_trembling_range, self.face_trembling_range)
            crop_img = rot_img[offset_y+crop_up:offset_y+crop_up+crop_size, offset_x+crop_left:offset_x+crop_left+crop_size]
            crop_img = cv2.resize(crop_img, (size, size))
        
        #print("angle:" + str(angle))
        #print("phi:" + str(phi))
        if leye_img.shape[:2] != (EYE_H, EYE_W) or reye_img.shape[:2] != (EYE_H, EYE_W) or nose_img.shape[:2] != (NOSE_H, NOSE_W) or mouth_img.shape[:2] != (MOUTH_H, MOUTH_W):
            raise Exception('Error while croping image. angle:{}, phi:{}'.format(angle, phi))
    
        
        return crop_img, leye_img, reye_img, nose_img, mouth_img
    
    def imread(self, path, normalize=True):
        
        with Open(path, 'rb') as f:
            image = Image.open(f)
            imarray = np.asarray(image)
        
        if normalize:
            return imarray.astype(np.float32) / np.iinfo(imarray.dtype).max
        else:
            imarray
            
        
         
    def get_generator(self, batch_size = 16, setting = 'train'):
        """
        data geneartor for training generator model.
        
        Args:
            batch_size (int): Number of images per batch
            setting (str): str of desired dataset type; 'train'/'valid'
        """ 
        
        def get_next():
             
            with self.lock:
                if setting == 'train':
                    datalist, self.train_cursor = self.batch_data(self.train_list, self.train_cursor, batch_size = batch_size)
                else:
                    datalist, self.valid_cursor = self.batch_data(self.valid_list, self.valid_cursor, batch_size = batch_size)
            
            first_time = True
            for i, x_data_path in enumerate(datalist):
                x_image_path = os.path.join(self.dataset_dir, x_data_path + '.jpg')
                x_image = self.imread(x_image_path, normalize=True)
                
                x_landmarks = self.load_landmarks(x_data_path)
                
                angle = DIR_ANGLE[x_data_path[-21:-17]]
                try:
                    x_face, x_leye, x_reye, x_nose, x_mouth = self.crop(x_image, x_landmarks, angle=angle)
                except (Exception, cv2.error) as e:
                    print(e)
                    print(x_data_path)
                    continue
                
                if self.mirror_to_one_side and angle < 0:
                    x_face = x_face[:,::-1,:]
                    buff = x_leye[:,::-1,:]
                    x_leye = x_reye[:,::-1,:]
                    x_reye = buff
                    x_nose = x_nose[:,::-1,:]
                    x_mouth = x_mouth[:,::-1,:]
                
                y_data_path = x_data_path[:-21] + '05_1' + os.sep + x_image_path[-20:-10] + '051_06'
                y_image_path = os.path.join(self.dataset_dir, y_data_path + '.jpg')
                y_image = self.imread(y_image_path, normalize=True)
                
                y_landmarks = self.load_landmarks(y_data_path)
                
                try:
                    y_face, y_leye, y_reye, y_nose, y_mouth = self.crop(y_image, y_landmarks, angle=0)
                except (Exception, cv2.error) as e:
                    print(e)
                    print(y_data_path)
                    continue
                
                if self.mirror_to_one_side and angle < 0:
                    y_face = y_face[:,::-1,:]
                    buff = y_leye[:,::-1,:]
                    y_leye = y_reye[:,::-1,:]
                    y_reye = buff                    
                    y_nose = y_nose[:,::-1,:]
                    y_mouth = y_mouth[:,::-1,:]
                y_face64 = resize(y_face, (64, 64), mode='constant')
                y_face32 = resize(y_face64, (32, 32),  mode='constant')
                
                # to adjust subject id starts from 0. (original multi pie subject id starts from 1)
                y_subject_id = int(x_data_path[-28:-25]) - 1
                y_subject_id = np_utils.to_categorical(y_subject_id, NUM_SUBJECTS)
                
                y_face_gray = rgb2gray(y_face)[:, :, np.newaxis]
                    
                if first_time:
                    first_time = False
                    
                    x_faces = x_face[np.newaxis,:]
                    x_leyes = x_leye[np.newaxis,:]
                    x_reyes = x_reye[np.newaxis,:]
                    x_noses = x_nose[np.newaxis,:]
                    x_mouthes = x_mouth[np.newaxis,:]
                    y_faces = y_face[np.newaxis,:]
                    y_face_grays = y_face_gray[np.newaxis,:]
                    y_faces64 = y_face64[np.newaxis,:]
                    y_faces32 = y_face32[np.newaxis,:]
                    y_subject_ids = y_subject_id[np.newaxis,:]
                    y_leyes = y_leye[np.newaxis,:]
                    y_reyes = y_reye[np.newaxis,:]
                    y_noses = y_nose[np.newaxis,:]
                    y_mouthes = y_mouth[np.newaxis,:]
                else:
                    if x_leyes.shape[1:] != x_leye.shape:
                        print(x_leyes.shape)
                        print(x_leye.shape)
                    if x_reyes.shape[1:] != x_reye.shape:
                        print(x_reyes.shape)
                        print(x_reye.shape)
                    if x_noses.shape[1:] != x_nose.shape:
                        print(x_noses.shape)
                        print(x_nose.shape)
                    if x_mouthes.shape[1:] != x_mouth.shape:
                        print(x_mouthes.shape)
                        print(x_mouth.shape)
                        
                    x_faces = np.concatenate((x_faces, x_face[np.newaxis,:]), axis=0)
                    x_leyes = np.concatenate((x_leyes, x_leye[np.newaxis,:]), axis=0)
                    x_reyes = np.concatenate((x_reyes, x_reye[np.newaxis,:]), axis=0)
                    x_noses = np.concatenate((x_noses, x_nose[np.newaxis,:]), axis=0)
                    x_mouthes = np.concatenate((x_mouthes, x_mouth[np.newaxis,:]), axis=0)
                    y_faces = np.concatenate((y_faces, y_face[np.newaxis,:]), axis=0)
                    y_face_grays = np.concatenate((y_face_grays, y_face_gray[np.newaxis,:]), axis=0)
                    y_faces64 = np.concatenate((y_faces64, y_face64[np.newaxis,:]), axis=0) 
                    y_faces32 = np.concatenate((y_faces32, y_face32[np.newaxis,:]), axis=0)
                    y_subject_ids = np.concatenate((y_subject_ids, y_subject_id[np.newaxis,:]), axis=0)
                    y_leyes = np.concatenate((y_leyes, y_leye[np.newaxis,:]), axis=0)
                    y_reyes = np.concatenate((y_reyes, y_reye[np.newaxis,:]), axis=0)
                    y_noses = np.concatenate((y_noses, y_nose[np.newaxis,:]), axis=0)
                    y_mouthes = np.concatenate((y_mouthes, y_mouth[np.newaxis,:]), axis=0)         
                
            x_z = np.random.normal(scale=0.02, size=(x_faces.shape[0], 100))
            
            return [x_faces, x_leyes, x_reyes, x_noses, x_mouthes, x_z], [y_faces, y_faces, y_faces, y_faces, y_faces, y_faces64, y_faces64, y_faces32, y_faces32, y_subject_ids, y_leyes, y_reyes, y_noses, y_mouthes]        
        

        if self.workers > 1:
            # use easy thread implementing
            # it is especially effective when getting data from google cloud storage
            data_pool = []
            while True:              
                if len(data_pool) > 0:
                    next_data = data_pool.pop(0)
                else:
                    next_data = get_next()
                
                while self.thread_pool_executor._work_queue.qsize() == 0 and len(data_pool) < self.workers:
                    self.thread_pool_executor.submit(fn=lambda : data_pool.append(get_next()))
                    
                yield next_data
        else:
            # dont use thread
            while True:
                next_data = get_next()
                
                yield next_data

            
    def get_class_generator(self, batch_size = 16, setting = 'train'):
        """
        data geneartor for fine tuning lcnn model with MULTI-PIE.
        
        Args:
            batch_size (int): Number of images per batch
            setting (str): str of desired dataset type; 'train'/'valid'
        """ 
        
        def get_next():
            
            with self.lock:
                if setting == 'train':
                    datalist, self.train_cursor = self.batch_data(self.train_list, self.train_cursor, batch_size = batch_size)
                else:
                    datalist, self.valid_cursor = self.batch_data(self.valid_list, self.valid_cursor, batch_size = batch_size)
            
            first_time = True
            for i, x_data_path in enumerate(datalist):
                x_image_path = os.path.join(self.dataset_dir, x_data_path + '.jpg')
                x_image = self.imread(x_image_path, normalize=True)
                
                x_landmarks = self.load_landmarks(x_data_path)
                
                angle = DIR_ANGLE[x_data_path[-21:-17]]
                try:
                    x_face = self.crop(x_image, x_landmarks, angle=angle)[0]
                except (Exception, cv2.error) as e:
                    print(e)
                    print(x_data_path)
                    continue
                
                x_face = x_face[:,:,np.newaxis]
                if self.mirror_to_one_side and angle < 0:
                    x_face = x_face[:,::-1,:]
                
                # to adjust subject id starts from 0. (original multi pie subject id starts from 1)
                y_subject_id = int(x_data_path[-28:-25]) - 1
                y_subject_id = np_utils.to_categorical(y_subject_id, NUM_SUBJECTS)
                                            
                if first_time:
                    first_time = False
                    
                    x_faces = x_face[np.newaxis,:]
                    y_subject_ids = y_subject_id[np.newaxis,:]
                else:
                    x_faces = np.concatenate((x_faces, x_face[np.newaxis,:]), axis=0)
                    y_subject_ids = np.concatenate((y_subject_ids, y_subject_id[np.newaxis,:]), axis=0)

            return x_faces, y_subject_ids
        
        if self.workers > 1:
            # use easy thread implementing
            # it is very effective when getting data from google cloud storage
            data_pool = []
            while True:              
                if len(data_pool) > 0:
                    next_data = data_pool.pop(0)
                else:
                    next_data = get_next()
                
                while self.thread_pool_executor._work_queue.qsize() == 0 and len(data_pool) < self.workers:
                    self.thread_pool_executor.submit(fn=lambda : data_pool.append(get_next()))
                    
                yield next_data
        else:
            # dont use thread
            while True:
                next_data = get_next()
                
                yield next_data

            
    
    def get_discriminator_generator(self, generator, batch_size=16, gt_shape=(4, 4), setting = 'train'):
        """
        data geneartor for training discriminator model.
        
        Args:
            generator (Model): generator model
            batch_size (int): Number of images per batch
            gt_shape (tuple): shape of return y 
            setting (str): str of desired dataset type; 'train'/'valid'
        """ 
        
        def get_next():

            with self.lock:
                if setting == 'train':
                    datalist, self.train_cursor = self.batch_data(self.train_list, self.train_cursor, batch_size = batch_size//2)
                else:
                    datalist, self.valid_cursor = self.batch_data(self.valid_list, self.valid_cursor, batch_size = batch_size//2)
                
            first_time = True
            for data_path_for_fake in datalist:
                # append generated image
                # generate(profile_face)
                profile_image_path = os.path.join(self.dataset_dir, data_path_for_fake + '.jpg')
                profile_image = self.imread(profile_image_path, normalize=True)
                
                profile_landmarks = self.load_landmarks(data_path_for_fake)
                
                angle = DIR_ANGLE[data_path_for_fake[-21:-17]]
                try:
                    profile_face, profile_leye, profile_reye, profile_nose, profile_mouth = self.crop(profile_image, profile_landmarks, angle=angle)
                except (Exception, cv2.error) as e:
                    print(e)
                    print(data_path_for_fake)
                    continue
                
                if self.mirror_to_one_side and angle < 0:
                    profile_face = profile_face[:,::-1,:]
                    buff = profile_leye[:,::-1,:]
                    profile_leye = profile_reye[:,::-1,:]
                    profile_reye = buff
                    profile_nose = profile_nose[:,::-1,:]
                    profile_mouth = profile_mouth[:,::-1,:]
                    
                if first_time:
                    first_time = False
                    
                    profile_faces = profile_face[np.newaxis,:]
                    profile_leyes = profile_leye[np.newaxis,:]
                    profile_reyes = profile_reye[np.newaxis,:]
                    profile_noses = profile_nose[np.newaxis,:]
                    profile_mouthes = profile_mouth[np.newaxis,:]
                else:
                    profile_faces = np.concatenate((profile_faces, profile_face[np.newaxis,:]), axis=0)
                    profile_leyes = np.concatenate((profile_leyes, profile_leye[np.newaxis,:]), axis=0)
                    profile_reyes = np.concatenate((profile_reyes, profile_reye[np.newaxis,:]), axis=0)
                    profile_noses = np.concatenate((profile_noses, profile_nose[np.newaxis,:]), axis=0)
                    profile_mouthes = np.concatenate((profile_mouthes, profile_mouth[np.newaxis,:]), axis=0)
                    
            x_fake_inputs = [profile_faces, profile_leyes, profile_reyes, profile_noses, profile_mouthes, np.random.normal(scale=0.02, size=(profile_faces.shape[0], 100))]
            
            
            first_time = True
            for data_path_for_real in datalist:
                # append true image
                # y_face
                
                image_path_for_real = os.path.join(self.dataset_dir, data_path_for_real + '.jpg')
                
                front_data_path = data_path_for_real[:-21] + '05_1' + os.sep + image_path_for_real[-20:-10] + '051_06'
                front_image_path = os.path.join(self.dataset_dir, front_data_path + '.jpg')
                front_image = self.imread(front_image_path, normalize=True)
                
                front_landmarks = self.load_landmarks(front_data_path)
                
                try:
                    front_face = self.crop(front_image, front_landmarks, angle=0)[0]
                except (Exception, cv2.error) as e:
                    print(e)
                    print(front_data_path)
                    continue
                
                if self.mirror_to_one_side and angle < 0:
                    front_face = front_face[:,::-1,:]
                    
                if first_time:
                    first_time = False
                    
                    x_real = front_face[np.newaxis,:]
                else:
                    x_real = np.concatenate((x_real, front_face[np.newaxis,:]), axis=0)
            
            return x_fake_inputs, x_real
        
        def make_batch(x_fake_inputs, x_real):
            x_fake = generator.predict(x_fake_inputs)[0]
            y_fake = np.zeros(shape=(x_fake.shape[0], *gt_shape, 1))
            y_real = np.ones(shape=(x_real.shape[0], *gt_shape, 1))
            
            return np.concatenate([x_fake, x_real]), np.concatenate([y_fake, y_real])
            
        if self.workers > 1:
            # use easy thread implementing
            # it is especially effective when getting data from google cloud storage
            data_pool = []
            while True:              
                if len(data_pool) > 0:
                    next_data = data_pool.pop(0)
                else:
                    next_data = get_next()
                
                while self.thread_pool_executor._work_queue.qsize() == 0 and len(data_pool) < self.workers:
                    self.thread_pool_executor.submit(fn=lambda : data_pool.append(get_next()))
                    
                yield make_batch(*next_data)
        else:
            # dont use thread
            while True:
                next_data = get_next()
                
                yield make_batch(*next_data)
            
def Open(name, mode='r'):
    """
    because, in my environment, sometimes gfile.Open is very slow when target file is localpath(not gs://),
    so, use normal open if the target path is localpath.
    """
    if len(name) >= 5 and name[:5] == 'gs://':
        return tf.gfile.Open(name, mode)
    else:
        return open(name, mode)
            
    