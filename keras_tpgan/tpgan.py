# -*- coding: utf-8 -*-
"""
Keras (with Tensorflow) re-implementation of TP-GAN.
Main part of this code is implemented reffering to author's pure Tensorflow implementation.
https://github.com/HRLTY/TP-GAN

original paper
Huang, R., Zhang, S., Li, T., & He, R. (2017). Beyond face rotation: Global and local perception gan for photorealistic and identity preserving frontal view synthesis. arXiv preprint arXiv:1704.04086.
https://arxiv.org/abs/1704.04086

@author: yhiro
"""

from keras import backend as K
from keras.layers import Input, Add, Maximum, Dense, Activation, BatchNormalization, Conv2D, Conv2DTranspose, Reshape, Flatten, Concatenate, Lambda, MaxPooling2D, ZeroPadding2D, Dropout, AveragePooling2D, Average
from keras.optimizers import SGD, Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import TensorBoard, Callback
from keras.models import Model, Sequential
from keras import regularizers
from keras.initializers import Constant, RandomNormal, TruncatedNormal, Zeros
from keras import losses
from keras.utils import multi_gpu_model
import tensorflow as tf
import os
import numpy as np
import re
import cv2
from PIL import Image

from keras_tpgan import multipie_gen

tf.logging.set_verbosity(tf.logging.WARN)

class TPGAN():

    def __init__(self, lcnn_extractor_weights='FINE_TUNED/LCNN/EXTRACTOR/WEIGHTS.hdf5',
                 base_filters=64, gpus=1,
                 generator_weights=None,
                 classifier_weights=None,
                 discriminator_weights=None):
        """ 
        initialize TP-GAN network with given weights file. if weights file is None, the weights are initialized by default initializer.
        
        Args:
            lcnn_extractor_weights (str): Light-CNN weights file which is trained with celeb-1M and fine-tuned with MULTI-PIE.
            base_filters (int): base filters count of TP-GAN. default 64.
            gpus (int): number of gpus to use.
            generator_weights (str): trained generator weights file path. it is used to resume training. not required when train from scratch.
            classifier_weights (str): trained classifier weights file path. it is used to resume training. not required when training from scratch.
            discriminator_weights (str): trained discriminator weights file path. it is used to resume training. not required when training from scratch.
        """

        
        self.gpus = gpus
        self.base_filters = base_filters
        self.generator_weights = generator_weights
        self.discriminator_weights = discriminator_weights
        self.classifier_weights = classifier_weights
        

        self._discriminator = None
        self._generator = None
        self._classifier = None
        self._parts_rotator = None
        
        self.generator_train_model = None
        self.discriminator_train_model = None
        
        self.gen_current_epochs = self.current_epoch_from_weights_file(self.generator_weights)
        self.disc_current_epochs = self.current_epoch_from_weights_file(self.discriminator_weights)
        self.class_epochs = self.current_epoch_from_weights_file(self.classifier_weights)
        
        self.lcnn = LightCNN(extractor_type='29v2', extractor_weights=lcnn_extractor_weights)     
            
    def current_epoch_from_weights_file(self, weights_file):

        if weights_file is not None:
            try:
                ret_epochs = int(re.match(r'.*epoch([0-9]+).*.hdf5', weights_file).groups()[0])
            except:
                ret_epochs = 0
        else:
            ret_epochs = 0
            
        return ret_epochs
                
    def discriminator(self):
        """ 
        getter of singleton discriminator
        """
        
        if self._discriminator is None:
            self._discriminator = self.build_discriminator(base_filters=self.base_filters)
        
            if self.discriminator_weights is not None:
                self._discriminator.load_weights(self.discriminator_weights)
            
        return self._discriminator

    def generator(self):
        """ 
        getter of singleton generator
        """
        
        if self._generator is None:
            self._generator = self.build_generator(base_filters=self.base_filters)
        
            if self.generator_weights is not None:
                self._generator.load_weights(self.generator_weights)
            
        return self._generator
    
    def classifier(self):
        """ 
        getter of singleton classifier
        """
        
        if self._classifier is None:
            self._classifier = self.build_classifier()
        
            if self.classifier_weights is not None:
                self._classifier.load_weights(self.classifier_weights)
            
        return self._classifier
    
    def parts_rotator(self):
        """ 
        getter of singleton part rotator for each part; left eye, right eye, nose, and mouth.
        """
        
        if self._parts_rotator is None:
            self._parts_rotator = self.build_parts_rotator(base_filters=self.base_filters)
                        
        return self._parts_rotator
    
    def _add_activation(self, X, func='relu'):
        """
        private func to add activation layer
        """

        if func is None:
            return X
        elif func == 'relu':
            return Activation('relu')(X)
        elif func == 'lrelu':
            return LeakyReLU()(X)
        else:
            raise Exception('Undefined function for activation: ' + func)

    def _res_block(self, X, kernel_size, batch_norm=False, activation=None, name=None):
        """
        private func to add residual block
        """
        
        X_shortcut = X
        
        if batch_norm:
            X = Conv2D(X.shape[-1].value, kernel_size=kernel_size, strides=(1, 1), padding='same', name=name+'_c1_0', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(X)
            X = BatchNormalization(epsilon=1e-5, name=name+'_c1_0_bn')(X)
        else:
            X = Conv2D(X.shape[-1].value, kernel_size=kernel_size, strides=(1, 1), padding='same', name=name+'_c1_1', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(X)
        self._add_activation(X, activation)
        
        if batch_norm:
            X = Conv2D(X.shape[-1].value, kernel_size=kernel_size, strides=(1, 1), padding='same', use_bias=False, name=name+'_c2_0', kernel_initializer=TruncatedNormal(stddev=0.02))(X)
            X = BatchNormalization(epsilon=1e-5, name=name+'_c2_0_bn')(X)
        else:
            X = Conv2D(X.shape[-1].value, kernel_size=kernel_size, strides=(1, 1), padding='same', name=name+'_c2_1', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(X)

        self._add_activation(X, activation)
            
        X = Add()([X_shortcut, X])
        
        return X
    
    def build_generator(self, name="generator", base_filters=64):
        """
        build generator model.
        """
        
        def combine_parts(size_hw, leye, reye, nose, mouth):

            img_h, img_w = size_hw
            
            leye_img = ZeroPadding2D(padding=((int(multipie_gen.LEYE_Y - multipie_gen.EYE_H/2), img_h - int(multipie_gen.LEYE_Y + multipie_gen.EYE_H/2)),
            (int(multipie_gen.LEYE_X - multipie_gen.EYE_W/2), img_w - int(multipie_gen.LEYE_X + multipie_gen.EYE_W/2))))(leye)
            
            reye_img = ZeroPadding2D(padding=((int(multipie_gen.REYE_Y - multipie_gen.EYE_H/2), img_h - int(multipie_gen.REYE_Y + multipie_gen.EYE_H/2)),
            (int(multipie_gen.REYE_X - multipie_gen.EYE_W/2), img_w - int(multipie_gen.REYE_X + multipie_gen.EYE_W/2))))(reye)
    
            nose_img = ZeroPadding2D(padding=((int(multipie_gen.NOSE_Y - multipie_gen.NOSE_H/2), img_h - int(multipie_gen.NOSE_Y + multipie_gen.NOSE_H/2)),
            (int(multipie_gen.NOSE_X - multipie_gen.NOSE_W/2), img_w - int(multipie_gen.NOSE_X + multipie_gen.NOSE_W/2))))(nose)
    
            mouth_img = ZeroPadding2D(padding=((int(multipie_gen.MOUTH_Y - multipie_gen.MOUTH_H/2), img_h - int(multipie_gen.MOUTH_Y + multipie_gen.MOUTH_H/2)),
            (int(multipie_gen.MOUTH_X - multipie_gen.MOUTH_W/2), img_w - int(multipie_gen.MOUTH_X + multipie_gen.MOUTH_W/2))))(mouth)
    
            return Maximum()([leye_img, reye_img, nose_img, mouth_img])
        
        full_name = name
        # shorten name
        name = name[0]
    
        in_img = Input(shape=(multipie_gen.IMG_H, multipie_gen.IMG_W, 3))
        mc_in_img128 = Concatenate()([in_img, Lambda(lambda x: x[:,:,::-1,:])(in_img)])
        mc_in_img64 = Lambda(lambda x: tf.image.resize_bilinear(x, [multipie_gen.IMG_H//2, multipie_gen.IMG_W//2]))(mc_in_img128)
        mc_in_img32 = Lambda(lambda x: tf.image.resize_bilinear(x, [multipie_gen.IMG_H//4, multipie_gen.IMG_W//4]))(mc_in_img64)
        
        c128 = Conv2D(base_filters, (7, 7), padding='same', strides=(1, 1), name=name+'_c128', use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(mc_in_img128)
        c128 = self._add_activation(c128, 'lrelu')
        c128r = self._res_block(c128, (7, 7), batch_norm=True, activation='lrelu', name=name+'_c128_r')
        
        c64 = Conv2D(base_filters, (5, 5), padding='same', strides=(2, 2), name=name+'_c64', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(c128r)
        c64 = BatchNormalization(epsilon=1e-5, name=name+'_c64_bn')(c64)
        c64 = self._add_activation(c64, 'lrelu')
        c64r = self._res_block(c64, (5, 5), batch_norm=True, activation='lrelu', name=name+'_c64_r')
        
        c32 = Conv2D(base_filters*2, (3, 3), padding='same', strides=(2, 2), name=name+'_c32', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(c64r)
        c32 = BatchNormalization(epsilon=1e-5, name=name+'_c32_bn')(c32)
        c32 = self._add_activation(c32, 'lrelu')
        c32r = self._res_block(c32, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c32_r')
        
        c16 = Conv2D(base_filters*4, (3, 3), padding='same', strides=(2, 2), name=name+'_c16', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(c32r)
        c16 = BatchNormalization(epsilon=1e-5, name=name+'_c16_bn')(c16)
        c16 = self._add_activation(c16, 'lrelu') 
        c16r = self._res_block(c16, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c16_r')
        
        c8 = Conv2D(base_filters*8, (3, 3), padding='same', strides=(2, 2), name=name+'_c8', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(c16r)
        c8 = BatchNormalization(epsilon=1e-5, name=name+'_c8_bn')(c8)
        c8 = self._add_activation(c8, 'lrelu')
        
        c8r = self._res_block(c8, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c8_r')     
        c8r2 = self._res_block(c8r, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c8_r2')     
        c8r3 = self._res_block(c8r2, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c8_r3')     
        c8r4 = self._res_block(c8r3, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c8_r4')     

        fc1 = Dense(512, name=name+'_fc1', kernel_initializer=RandomNormal(stddev=0.02), bias_initializer=Constant(0.1), kernel_regularizer=regularizers.l2(0.005))(Flatten()(c8r4))
        fc2 = Maximum()([Lambda(lambda x: x[:, :256])(fc1), Lambda(lambda x: x[:, 256:])(fc1)])
        
        in_noise = Input(shape=(100,))
        fc2_with_noise = Concatenate()([fc2, in_noise])
        
        fc3 = Dense(8*8*base_filters, name=name+'_fc3', kernel_initializer=RandomNormal(stddev=0.02), bias_initializer=Constant(0.1))(fc2_with_noise)
        
        f8 = Conv2DTranspose(base_filters, (8, 8), padding='valid', strides=(1, 1), name=name+'_f8', activation='relu', kernel_initializer=RandomNormal(stddev=0.02), bias_initializer=Zeros())(Reshape((1, 1, fc3.shape[-1].value))(fc3))
        f32 = Conv2DTranspose(base_filters//2, (3, 3), padding='same', strides=(4, 4), name=name+'_f16', activation='relu', kernel_initializer=RandomNormal(stddev=0.02), bias_initializer=Zeros())(f8)
        f64 = Conv2DTranspose(base_filters//4, (3, 3), padding='same', strides=(2, 2), name=name+'_f64', activation='relu', kernel_initializer=RandomNormal(stddev=0.02), bias_initializer=Zeros())(f32)
        f128 = Conv2DTranspose(base_filters//8, (3, 3), padding='same', strides=(2, 2), name=name+'_f128', activation='relu', kernel_initializer=RandomNormal(stddev=0.02), bias_initializer=Zeros())(f64)
        
        # size8
        d8 = Concatenate(name=name+'_d8')([c8r4, f8])
        d8r = self._res_block(d8, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d8_r')
        d8r2 = self._res_block(d8r, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d8_r2')
        d8r3 = self._res_block(d8r2, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d8_r3')
        
        # size16
        d16 = Conv2DTranspose(base_filters*8, (3, 3), padding='same', strides=(2, 2), name=name+'_d16', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(d8r3)
        d16 = BatchNormalization(epsilon=1e-5, name=name+'_d16_bn')(d16)
        d16 = self._add_activation(d16, 'relu')
        d16r = self._res_block(c16r, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d16_r')
        d16r2 = self._res_block(Concatenate()([d16, d16r]), (3, 3), batch_norm=True, activation='lrelu', name=name+'_d16_r2')
        d16r3 = self._res_block(d16r2, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d16_r3')
        
        # size32
        d32 = Conv2DTranspose(base_filters*4, (3, 3), padding='same', strides=(2, 2), name=name+'_d32', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(d16r3)
        d32 = BatchNormalization(epsilon=1e-5, name=name+'_d32_bn')(d32)
        d32 = self._add_activation(d32, 'relu')
        d32r = self._res_block(Concatenate()([c32r, mc_in_img32, f32]), (3, 3), batch_norm=True, activation='lrelu', name=name+'_d32_r')
        d32r2 = self._res_block(Concatenate()([d32, d32r]), (3, 3), batch_norm=True, activation='lrelu', name=name+'_d32_r2')
        d32r3 = self._res_block(d32r2, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d32_r3')
        img32 = Conv2D(3, (3, 3), padding='same', strides=(1, 1), activation='tanh', name=name+'_img32', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(d32r3)
        
        # size64
        d64 = Conv2DTranspose(base_filters*2, (3, 3), padding='same', strides=(2, 2), name=name+'_d64', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(d32r3)
        d64 = BatchNormalization(epsilon=1e-5, name=name+'_d64_bn')(d64)
        d64 = self._add_activation(d64, 'relu')
        d64r = self._res_block(Concatenate()([c64r, mc_in_img64, f64]), (5, 5), batch_norm=True, activation='lrelu', name=name+'_d64_r')
        
        interpolated64 = Lambda(lambda x: tf.image.resize_bilinear(x, [64, 64]))(img32) # Use Lambda layer to wrap tensorflow func, resize_bilinear
        
        d64r2 = self._res_block(Concatenate()([d64, d64r, interpolated64]), (3, 3), batch_norm=True, activation='lrelu', name=name+'_d64_r2')
        d64r3 = self._res_block(d64r2, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d64_r3')
        img64 = Conv2D(3, (3, 3), padding='same', strides=(1, 1), activation='tanh', name=name+'_img64', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(d64r3)
                
        # size128
        d128 = Conv2DTranspose(base_filters, (3, 3), padding='same', strides=(2, 2), name=name+'_d128', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(d64r3)
        d128 = BatchNormalization(epsilon=1e-5, name=name+'_d128_bn')(d128)
        d128 = self._add_activation(d128, 'relu')
        d128r = self._res_block(Concatenate()([c128r, mc_in_img128, f128]), (5, 5), batch_norm=True, activation='lrelu', name=name+'_d128_r')

        interpolated128 = Lambda(lambda x: tf.image.resize_bilinear(x, [128, 128]))(img64) # Use Lambda layer to wrap tensorflow func, resize_bilinear
        
        in_leye = Input(shape=(multipie_gen.EYE_H, multipie_gen.EYE_W, 3))
        in_reye = Input(shape=(multipie_gen.EYE_H, multipie_gen.EYE_W, 3))
        in_nose = Input(shape=(multipie_gen.NOSE_H, multipie_gen.NOSE_W, 3))
        in_mouth = Input(shape=(multipie_gen.MOUTH_H, multipie_gen.MOUTH_W, 3))
        
        front_leye_img, front_leye_feat, front_reye_img, front_reye_feat, front_nose_img, front_nose_feat, front_mouth_img, front_mouth_feat\
        = self.parts_rotator()([in_leye, in_reye, in_nose, in_mouth])
        
        combined_parts_img = combine_parts([128, 128], front_leye_img, front_reye_img, front_nose_img, front_mouth_img)
        combined_parts_feat = combine_parts([128, 128], front_leye_feat, front_reye_feat, front_nose_feat, front_mouth_feat)
        
        d128r2 = self._res_block(Concatenate()([d128, d128r, interpolated128, combined_parts_feat, combined_parts_img]), (3, 3), batch_norm=True, activation='lrelu', name=name+'_d128_r2')
        d128r2c = Conv2D(base_filters, (5, 5), padding='same', strides=(1, 1), name=name+'_d128_r2c', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(d128r2)
        d128r2c = BatchNormalization(epsilon=1e-5, name=name+'_d128_r2c_bn')(d128r2c)
        d128r2c = self._add_activation(d128r2c, 'lrelu')
        d128r3c = self._res_block(d128r2c, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d128_r3c')
        d128r3c2 = Conv2D(base_filters//2, (3, 3), padding='same', strides=(1, 1), name=name+'_d128_r3c2', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(d128r3c)
        d128r3c2 = BatchNormalization(epsilon=1e-5, name=name+'_d128_r3c2_bn')(d128r3c2)
        d128r3c2 = self._add_activation(d128r3c2, 'lrelu')
        img128 = Conv2D(3, (3, 3), padding='same', strides=(1, 1), activation='tanh', name=name+'_img128', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(d128r3c2)
        
        ret_model = CloudableModel(inputs=[in_img, in_leye, in_reye, in_nose, in_mouth, in_noise], outputs=[img128, img64, img32, fc2, front_leye_img, front_reye_img, front_nose_img, front_mouth_img], name=full_name)        
        ret_model.summary()
        
        return ret_model
    
    def build_classifier(self, name='classifier'):
        """
        build classifier model.
        """
        
        full_name = name
        # shorten name
        name = name[0]
        
        in_feat = Input(shape=(256,))
        X = Dropout(0.7)(in_feat)
        clas = Dense(multipie_gen.NUM_SUBJECTS, activation='softmax', kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=regularizers.l2(0.005),
                     use_bias=False, name=name+'_dense')(X)
        
        ret_classifier = CloudableModel(inputs=in_feat, outputs=clas, name=full_name)
        
        ret_classifier.summary()
        return ret_classifier    
                    
    def build_train_generator_model(self):
        """
        build train model for generator.
        this model wraps generator and classifier, adds interface for loss functions.
        """
        
        in_img = Input(shape=(multipie_gen.IMG_H, multipie_gen.IMG_W, 3))
        in_leye = Input(shape=(multipie_gen.EYE_H, multipie_gen.EYE_W, 3))
        in_reye = Input(shape=(multipie_gen.EYE_H, multipie_gen.EYE_W, 3))
        in_nose = Input(shape=(multipie_gen.NOSE_H, multipie_gen.NOSE_W, 3))
        in_mouth = Input(shape=(multipie_gen.MOUTH_H, multipie_gen.MOUTH_W, 3))
        in_noise = Input(shape=(100,))
        
        img128, img64, img32, fc2, front_leye_img, front_reye_img, front_nose_img, front_mouth_img\
        = self.generator()([in_img, in_leye, in_reye, in_nose, in_mouth, in_noise]) 
        
        subject_id = self.classifier()(fc2)
        
        img128_gray = Lambda(lambda x: tf.image.rgb_to_grayscale(x))(img128)
        lcnn_vec, lcnn_map = self.lcnn.extractor()(img128_gray)
        
        # add name label to connect with each loss functions
        img128_px = Lambda(lambda x:x, name = "00img128px")(img128)
        img128_sym = Lambda(lambda x:x, name = "01img128sym")(img128)
        img128_ip = Lambda(lambda x:x, name = "02ip")(img128)
        img128_adv = Lambda(lambda x:x, name = "03adv")(img128)
        img128_tv = Lambda(lambda x:x, name = "04tv")(img128)
        img64_px = Lambda(lambda x:x, name = "05img64px")(img64)
        img64_sym = Lambda(lambda x:x, name = "06img64sym")(img64)
        img32_px = Lambda(lambda x:x, name = "07img32px")(img32)
        img32_sym = Lambda(lambda x:x, name = "08img32sym")(img32)
        subject_id = Lambda(lambda x:x, name = "09classify")(subject_id)
        leye = Lambda(lambda x:x, name = "10leye")(front_leye_img)
        reye = Lambda(lambda x:x, name = "11reye")(front_reye_img)
        nose = Lambda(lambda x:x, name = "12nose")(front_nose_img)
        mouth = Lambda(lambda x:x, name = "13mouth")(front_mouth_img)
        
        ret_model = CloudableModel(inputs=[in_img, in_leye, in_reye, in_nose, in_mouth, in_noise],
                          outputs=[img128_px, img128_sym, img128_ip, img128_adv, img128_tv, img64_px, img64_sym, img32_px, img32_sym, subject_id, leye, reye, nose, mouth],
                          name='train_genarator_model')
        ret_model.summary()
        
        return ret_model
    
    def build_parts_rotator(self, base_filters=64):
        """
        build models for all each part rotator.
        """       
        
        leye_rotator = self.build_part_rotator('leye', base_filters=base_filters, in_h=multipie_gen.EYE_H , in_w=multipie_gen.EYE_W)
        reye_rotator = self.build_part_rotator('reye', base_filters=base_filters, in_h=multipie_gen.EYE_H , in_w=multipie_gen.EYE_W)
        nose_rotator = self.build_part_rotator('nose', base_filters=base_filters, in_h=multipie_gen.NOSE_H , in_w=multipie_gen.NOSE_W)
        mouth_rotator = self.build_part_rotator('mouth', base_filters=base_filters, in_h=multipie_gen.MOUTH_H , in_w=multipie_gen.MOUTH_W)
            
        in_leye = Input(shape=(multipie_gen.EYE_H, multipie_gen.EYE_W, 3))
        in_reye = Input(shape=(multipie_gen.EYE_H, multipie_gen.EYE_W, 3))
        in_nose = Input(shape=(multipie_gen.NOSE_H, multipie_gen.NOSE_W, 3))
        in_mouth = Input(shape=(multipie_gen.MOUTH_H, multipie_gen.MOUTH_W, 3))
        
        out_leye_img, out_leye_feat = leye_rotator(in_leye)
        out_reye_img, out_reye_feat = reye_rotator(in_reye)
        out_nose_img, out_nose_feat = nose_rotator(in_nose)
        out_mouth_img, out_mouth_feat = mouth_rotator(in_mouth)
        
        ret_model = CloudableModel(inputs=[in_leye, in_reye, in_nose, in_mouth],
                            outputs=[out_leye_img, out_leye_feat, out_reye_img, out_reye_feat, out_nose_img, out_nose_feat, out_mouth_img, out_mouth_feat], name='parts_rotator')
        ret_model.summary()
        
        return ret_model
    
    def build_part_rotator(self, name, in_h, in_w, base_filters=64):
        """
        build model for one part rotator.
        """   
        
        in_img = Input(shape=(in_h, in_w, 3))
        
        c0 = Conv2D(base_filters, (3, 3), padding='same', strides=(1, 1), name=name+'_c0', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(in_img)
        c0r = self._res_block(c0, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c0_r')
        c1 = Conv2D(base_filters*2, (3, 3), padding='same', strides=(2, 2), name=name+'_c1', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(c0r)
        c1 = BatchNormalization(name=name+'_c1_bn')(c1)
        c1 = self._add_activation(c1, 'lrelu')
        
        c1r = self._res_block(c1, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c1_r')
        c2 = Conv2D(base_filters*4, (3, 3), padding='same', strides=(2, 2), name=name+'_c2', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(c1r)
        c2 = BatchNormalization(name=name+'_c2_bn')(c2)
        c2 = self._add_activation(c2, 'lrelu')
        
        c2r = self._res_block(c2, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c2_r')
        c3 = Conv2D(base_filters*8, (3, 3), padding='same', strides=(2, 2), name=name+'_c3', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(c2r)
        c3 = BatchNormalization(name=name+'_c3_bn')(c3)
        c3 = self._add_activation(c3, 'lrelu')
        
        c3r = self._res_block(c3, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c3_r')
        c3r2 = self._res_block(c3r, (3, 3), batch_norm=True, activation='lrelu', name=name+'_c3_r2')
        
        d1 = Conv2DTranspose(base_filters*4, (3, 3), padding='same', strides=(2, 2), name=name+'_d1', use_bias=True, kernel_initializer=RandomNormal(stddev=0.02))(c3r2)
        d1 = BatchNormalization(name=name+'_d1_bn')(d1)
        d1 = self._add_activation(d1, 'lrelu')
        
        after_select_d1 = Conv2D(base_filters*4, (3, 3), padding='same', strides=(1, 1), name=name+'_asd1', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(Concatenate()([d1, c2r]))
        after_select_d1 = BatchNormalization(name=name+'_asd1_bn')(after_select_d1)
        after_select_d1 = self._add_activation(after_select_d1, 'lrelu')
        d1r = self._res_block(after_select_d1, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d1_r')
        d2 = Conv2DTranspose(base_filters*2, (3, 3), padding='same', strides=(2, 2), name=name+'_d2', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(d1r)
        d2 = BatchNormalization(name=name+'_d2_bn')(d2)
        d2 = self._add_activation(d2, 'lrelu')
        
        after_select_d2 = Conv2D(base_filters*2, (3, 3), padding='same', strides=(1, 1), name=name+'_asd2', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(Concatenate()([d2, c1r]))
        after_select_d2 = BatchNormalization(name=name+'_asd2_bn')(after_select_d2)
        after_select_d2 = self._add_activation(after_select_d2, 'lrelu')
        d2r = self._res_block(after_select_d2, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d2_r')
        d3 = Conv2DTranspose(base_filters, (3, 3), padding='same', strides=(2, 2), name=name+'_d3', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(d2r)
        d3 = BatchNormalization(name=name+'_d3_bn')(d3)
        d3 = self._add_activation(d3, 'lrelu')
        
        after_select_d3 = Conv2D(base_filters, (3, 3), padding='same', strides=(1, 1), name=name+'_asd3', use_bias=False, kernel_initializer=TruncatedNormal(stddev=0.02))(Concatenate()([d3, c0r]))
        after_select_d3 = BatchNormalization(name=name+'_asd3_bn')(after_select_d3)
        after_select_d3 = self._add_activation(after_select_d3, 'lrelu')
        part_feat = self._res_block(after_select_d3, (3, 3), batch_norm=True, activation='lrelu', name=name+'_d3_r')
        
        part_img = Conv2D(3, (3, 3), padding='same', strides=(1, 1), activation='tanh', name=name+'_c4', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(part_feat)

        ret_model = CloudableModel(inputs=[in_img], outputs=[part_img, part_feat], name= name + '_rotator')
        ret_model.summary()
        
        return ret_model
    
    def build_discriminator(self, name='discriminator', base_filters=64):
        """
        build model for discriminator.
        """   
        
        full_name = name
        # shorten name
        name = name[0]
        
        in_img = Input(shape=(multipie_gen.IMG_H, multipie_gen.IMG_W, 3))
        
        c64 = Conv2D(base_filters, (3, 3), padding='same', strides=(2, 2), name=name+'_c64', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(in_img)
        c64 = self._add_activation(c64, 'lrelu')
        
        c32 = Conv2D(base_filters*2, (3, 3), padding='same', strides=(2, 2), name=name+'_c32', use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.02))(c64)
        #c32 = BatchNormalization(center=True, scale=True, name=name+'_c32_bn')(c32)
        c32 = self._add_activation(c32, 'lrelu')
        
        c16 = Conv2D(base_filters*4, (3, 3), padding='same', strides=(2, 2), name=name+'_c16', use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.02))(c32)
        #c16 = BatchNormalization(center=True, scale=True, name=name+'_c16_bn')(c16)
        c16 = self._add_activation(c16, 'lrelu')
        
        c8 = Conv2D(base_filters*8, (3, 3), padding='same', strides=(2, 2), name=name+'_c8', use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.02))(c16)
        #c8 = BatchNormalization(center=True, scale=True, name=name+'_c8_bn')(c8)
        c8 = self._add_activation(c8, 'lrelu')
        c8r = self._res_block(c8, (3, 3), batch_norm=False, activation='lrelu', name=name+'_c8_r')
        
        c4 = Conv2D(base_filters*8, (3, 3), padding='same', strides=(2, 2), name=name+'_c4', use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.02))(c8r)
        #c4 = BatchNormalization(center=True, scale=True, name=name+'_c4_bn')(c4)
        c4 = self._add_activation(c4, 'lrelu')
        c4r = self._res_block(c4, (3, 3), batch_norm=False, activation='lrelu', name=name+'_c4_r')
        
        feat = Conv2D(1, (1, 1), padding='same', strides=(1, 1), name=name+'_c4_r_c', activation='sigmoid', kernel_initializer=TruncatedNormal(stddev=0.02), bias_initializer=Zeros())(c4r)
        
        ret_model = CloudableModel(inputs=in_img, outputs=feat, name=full_name)
        ret_model.summary()
        
        return ret_model
        
    class SaveWeightsCallback(Callback):
    
        def __init__(self, target_models, out_dir, period):
            """
            Args:
                target_models (list): list of save target models
                out_dir (str): output dir
                period (int): save interval epochs
            """
            self.target_models = target_models
            self.out_dir = out_dir
            self.period = period
    
        def on_epoch_end(self, epoch, logs):
            if (epoch + 1) % self.period == 0:
                for target_model in self.target_models:
                    out_model_dir = '{}{}/'.format(self.out_dir, target_model.name)
                    tf.gfile.MakeDirs(out_model_dir)
                    
                    target_model.save_weights(out_model_dir + 'epoch{epoch:04d}_loss{loss:.3f}.hdf5'.format(epoch=epoch + 1, loss=logs['loss']), overwrite=True)
    
    def train_gan(self, gen_datagen_creator, gen_train_batch_size, gen_valid_batch_size,
                  disc_datagen_creator, disc_batch_size, disc_gt_shape,
                  optimizer,
                  gen_steps_per_epoch=300, disc_steps_per_epoch=10, epochs=100, 
                  out_dir='../out/', out_period=5, is_output_img=False,
                  lr=0.001, decay=0, lambda_128=1, lambda_64=1, lambda_32=1.5,
                  lambda_sym=3e-1, lambda_ip=1e1, lambda_adv=2e1, lambda_tv=1e-3,
                  lambda_class=4e-1, lambda_parts=3):
        """
        train both generator and discriminator as GAN.
        
        Args:
            gen_datagen_creator (func): function for create generator datagen
            gen_train_batch_size (int): batch size for training generator
            gen_valid_batch_size (int): batch size for validate generator
            disc_datagen_creator (func): function for create discriminator datagen
            disc_batch_size (int): batch size for training/validate discriminator
            disc_gt_shape (int): dicriminator output size    
            optimizer (Optimizer): keras optimizer for training generator. (currently, training discriminator always use Adam)
            gen_steps_per_epoch (int): steps per epoch for training generator
            disc_steps_per_epoch (int): steps per epoch for training discriminator
            epochs (int): train epochs
            out_dir (str): out_dir for weights, logs, sample images
            out_period (int): output interval epochs.
            is_output_img (bool): if True, output sample images each out period epochs.
            lr (float): learning rate for training generator optimizer
            lambda_128 (func): func returns the coefficient of 128px image output
            lambda_64 (func): func returns the coefficient of 64px image output
            lambda_32 (func): func returns the coefficient of 32px image output
            lambda_sym (func): func returns the coefficient of symmetricity loss
            lambda_ip (func): func returns the coefficient of identity preserve loss
            lambda_adv (func): func returns the coefficient of adversarial loss
            lambda_tv (func): func returns the coefficient of total variation loss
            lambda_class (func): func returns the coefficient of classification loss
            lambda_parts (func): func returns the coefficient of part img loss
        """
            
        for i in range(epochs):
            print('train generator {}/{}'.format(i+1, epochs))
            lambda_128_i = lambda_128(i)
            lambda_64_i = lambda_64(i)
            lambda_32_i = lambda_32(i)
            lambda_sym_i = lambda_sym(i)
            lambda_ip_i = lambda_ip(i)
            lambda_adv_i =lambda_adv(i)
            lambda_tv_i = lambda_tv(i)
            lambda_class_i = lambda_class(i)
            lambda_parts_i=lambda_parts(i)
            print('params for this epoch\n\
                  lambda_128:{}\n\
                  lambda_64:{}\n\
                  lambda_32:{}\n\
                  lambda_sym:{}\n\
                  lambda_ip:{}\n\
                  lambda_adv:{}\n\
                  lambda_tv:{}\n\
                  lambda_class:{}\n\
                  lambda_parts:{}\n'.format(
                  lambda_128_i, 
                  lambda_64_i, 
                  lambda_32_i,
                  lambda_sym_i, 
                  lambda_ip_i, 
                  lambda_adv_i, 
                  lambda_tv_i, 
                  lambda_class_i,
                  lambda_parts_i))
        
            gen_train_datagen = gen_datagen_creator(batch_size=gen_train_batch_size, setting='train')
            gen_valid_datagen = gen_datagen_creator(batch_size=gen_valid_batch_size, setting='valid')
            history = self.train_generator(train_gen=gen_train_datagen, valid_gen=gen_valid_datagen, optimizer=optimizer, steps_per_epoch=gen_steps_per_epoch,
                                           epochs=1, is_output_img=is_output_img, out_dir=out_dir, out_period=out_period,
                                           lr=lr, lambda_128=lambda_128_i, lambda_64=lambda_64_i, lambda_32=lambda_32_i,
                                           lambda_sym=lambda_sym_i, lambda_ip=lambda_ip_i, lambda_adv=lambda_adv_i, lambda_tv=lambda_tv_i,
                                           lambda_class=lambda_class_i, lambda_parts=lambda_parts_i)
            print('epoch:{} generator model trained. loss is as follows.\n{}'.format(self.gen_current_epochs, history.history['loss']))
            
            print('train discriminator {}/{}'.format(i+1, epochs))
            disc_datagen = disc_datagen_creator(self.generator(), batch_size=disc_batch_size, setting='train', gt_shape=disc_gt_shape)
            history = self.train_discriminator(train_gen=disc_datagen, valid_gen=disc_datagen, steps_per_epoch=disc_steps_per_epoch,
                                               epochs=1, out_dir=out_dir, out_period=out_period)
            print('epoch:{} discriminator model trained. binary_accuracy is as follows.\n{}'.format(self.disc_current_epochs, history.history['binary_accuracy']))

    def train_generator(self, train_gen, valid_gen, optimizer=Adam(lr=0.0001), steps_per_epoch=100, epochs=1, is_output_img=False, out_dir='../out/', out_period=1,
              lr=0.001, decay=0, lambda_128=1, lambda_64=1, lambda_32=1.5,
              lambda_sym=3e-1, lambda_ip=1e1, lambda_adv=2e1, lambda_tv=1e-3,
              lambda_class=4e-1, lambda_parts=3):
        """
        train generator.
        
        Args:
            train_gen (generator): generator which provides mini-batch train data
            valid_gen (generator): generator which provides mini-batch validation data 
            optimizer (Optimizer): keras optimizer for training generator.
            steps_per_epoch (int): steps per epoch for training generator
            epochs (int): train epochs
            is_output_img (bool): if True, output sample images each out period epochs.
            out_dir (str): out_dir for weights, logs, sample images
            out_period (int): output interval epochs.
            lr (float): learning rate for training generator optimizer
            decay (float): learning rate decay for training generator optimizer
            lambda_128 (float): coefficient of 128px image output
            lambda_64 (float): coefficient of 64px image output
            lambda_32 (float): coefficient of 32px image output
            lambda_sym (float): coefficient of symmetricity loss
            lambda_ip (float): coefficient of identity preserve loss
            lambda_adv (float): coefficient of adversarial loss
            lambda_tv (float): coefficient of total variation loss
            lambda_class (float): coefficient of classification loss
            lambda_parts (float): coefficient of part img loss
        """
        
        class SaveSampleImageCallback(Callback):
            """
            this callback save sample images generated by current trained model.
            """
            
            def __init__(self, generator, out_dir, period): 
                """
                Args:
                    generator (generator): generator which provides input profile images
                    out_dir (str): output dir
                    period (int): save interval epochs
                """
                
                self.generator = generator
                self.out_dir = out_dir
                self.period = period
                
                tf.gfile.MakeDirs('{}img128/'.format(self.out_dir))
                tf.gfile.MakeDirs('{}img64/'.format(self.out_dir))
                tf.gfile.MakeDirs('{}img32/'.format(self.out_dir))
                tf.gfile.MakeDirs('{}leye/'.format(self.out_dir))
                tf.gfile.MakeDirs('{}reye/'.format(self.out_dir))
                tf.gfile.MakeDirs('{}nose/'.format(self.out_dir))
                tf.gfile.MakeDirs('{}mouth/'.format(self.out_dir))
        
            def on_epoch_end(self, epoch, logs):
                
                def imsave(path, imarray):
                    
                    imarray[np.where(imarray<0)] = 0
                    imarray[np.where(imarray>1)] = 1
                    image = Image.fromarray((imarray*np.iinfo(np.uint8).max).astype(np.uint8))
                    
                    with Open(path, 'wb') as f:
                        image.save(f)

                if (epoch + 1) % self.period == 0:
                    inputs, _ = next(self.generator)
                    img128, _, _, _, _, img64, _, img32, _, subject_id, front_leye_img, front_reye_img, front_nose_img, front_mouth_img\
                    = self.model.predict(inputs)
                    for i in range(len(img128)):
                        sub_id = np.argmax(subject_id[i])
                        imsave('{}img128/epoch{:04d}_img128_subject{}_loss{:.3f}_{}.png'.format(self.out_dir, epoch+1, sub_id, logs['loss'], i), img128[i])
                        imsave('{}img64/epoch{:04d}_img64_subject{}_loss{:.3f}_{}.png'.format(self.out_dir, epoch+1, sub_id, logs['loss'], i), img64[i])
                        imsave('{}img32/epoch{:04d}_img32_subject{}_loss{:.3f}_{}.png'.format(self.out_dir, epoch+1, sub_id, logs['loss'], i), img32[i])
                        imsave('{}leye/epoch{:04d}_leye_subject{}_loss{:.3f}_{}.png'.format(self.out_dir, epoch+1, sub_id, logs['loss'], i), front_leye_img[i])
                        imsave('{}reye/epoch{:04d}_reye_subject{}_loss{:.3f}_{}.png'.format(self.out_dir, epoch+1, sub_id, logs['loss'], i), front_reye_img[i])
                        imsave('{}nose/epoch{:04d}_nose_subject{}_loss{:.3f}_{}.png'.format(self.out_dir, epoch+1, sub_id, logs['loss'], i), front_nose_img[i])
                        imsave('{}mouth/epoch{:04d}_mouth_subject{}_loss{:.3f}_{}.png'.format(self.out_dir, epoch+1, sub_id, logs['loss'], i), front_mouth_img[i])

        def _loss_img128_px(y_true, y_pred):
            """
            pixel loss for size 128x128
            """
            loss_pixel = K.mean(losses.mean_absolute_error(y_true, y_pred))
            return lambda_128 * loss_pixel
        
        def _loss_img128_sym(y_true, y_pred):
            """
            symmetricity loss for size 128x128
            """
            loss_sym = K.mean(losses.mean_absolute_error(y_pred[:,:,::-1,:], y_pred))
            return lambda_128 * lambda_sym * loss_sym
                
        def _loss_ip(y_true, y_pred):
            """
            identity preserve loss computed from img128
            """
            y_true_gray = Lambda(lambda x: tf.image.rgb_to_grayscale(x), output_shape=(multipie_gen.IMG_H, multipie_gen.IMG_W, 1))(y_true)
            vec_true, map_true = self.lcnn.extractor()(y_true_gray)
            
            y_pred_gray = Lambda(lambda x: tf.image.rgb_to_grayscale(x))(y_pred)
            vec_pred, map_pred = self.lcnn.extractor()(y_pred_gray)
            
            return lambda_ip * (K.mean(K.abs(vec_pred - vec_true)) + K.mean(K.abs(map_pred - map_true)))
        
        def _loss_adv(y_true, y_pred):
            """
            adversarial loss computed from img128
            """
            disc_score = self.discriminator()(y_pred)
            return lambda_adv * K.mean(losses.binary_crossentropy(K.ones_like(disc_score)*0.9, disc_score))
 
        def _loss_tv(y_true, y_pred):
            """
            total variation loss computed from img128
            """
            return lambda_tv * K.mean(tf.image.total_variation(y_pred))
        
        def _loss_img64_px(y_true, y_pred):
            """
            pixel loss for size 64x64
            """
            loss_pixel = K.mean(losses.mean_absolute_error(y_true, y_pred))
            return lambda_64 * loss_pixel
        
        def _loss_img64_sym(y_true, y_pred):
            """
            symmetricity loss for size 64x64
            """
            loss_sym = K.mean(losses.mean_absolute_error(y_pred[:,:,::-1,:], y_pred))
            return lambda_64 * lambda_sym * loss_sym
       
        def _loss_img32_px(y_true, y_pred):
            """
            pixel loss for size 32x32
            """
            loss_pixel = K.mean(losses.mean_absolute_error(y_true, y_pred))
            return lambda_32 * loss_pixel
        
        def _loss_img32_sym(y_true, y_pred):
            """
            symmetricity loss for size 32x32
            """
            loss_sym = K.mean(losses.mean_absolute_error(y_pred[:,:,::-1,:], y_pred))
            return lambda_32 * lambda_sym * loss_sym
            
        def _loss_classify(y_true, y_pred):
            """
            classification loss
            """
            return lambda_class * K.mean(losses.categorical_crossentropy(y_true, y_pred))

        def _loss_part(y_true, y_pred):
            """
            rotated part image loss
            """
            return lambda_parts * K.mean(losses.mean_absolute_error(y_true, y_pred))
        
        # create model for training        
        if self.generator_train_model is None:
            self.generator_train_model = self.build_train_generator_model()
            if self.gpus > 1:
                self.generator_train_model = multi_gpu_model(self.generator_train_model, gpus=self.gpus)
        
            # set trainable flag and recompile               
            self.generator().trainable = True
            self.classifier().trainable = True
            self.discriminator().trainable = False
            self.lcnn.extractor().trainable = False
            self.generator_train_model.compile(optimizer=optimizer,
                                loss={'00img128px': _loss_img128_px,
                                      '01img128sym': _loss_img128_sym,
                                      '02ip': _loss_ip,
                                      '03adv': _loss_adv,
                                      '04tv': _loss_tv,
                                      '05img64px': _loss_img64_px,
                                      '06img64sym': _loss_img64_sym,
                                      '07img32px': _loss_img32_px,
                                      '08img32sym': _loss_img32_sym,
                                      '09classify': _loss_classify,
                                      '10leye': _loss_part,
                                      '11reye': _loss_part,
                                      '12nose': _loss_part,
                                      '13mouth': _loss_part},
                                metrics={'09classify':'acc'})
            
        callbacks = []
        callbacks.append(TensorBoard(log_dir=out_dir+'logs/generator/'))
        callbacks.append(self.SaveWeightsCallback(target_models=[self.generator(), self.classifier()], out_dir=out_dir+'weights/', period=out_period))
        if is_output_img:
            callbacks.append(SaveSampleImageCallback(generator=valid_gen, out_dir=out_dir+'images/', period=out_period))
        history = self.generator_train_model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs+self.gen_current_epochs,
                                            callbacks=callbacks, workers=0, validation_data=valid_gen, validation_steps=1,
                                            shuffle=False, initial_epoch=self.gen_current_epochs)
        self.gen_current_epochs += epochs
            
        return history
    
    def train_discriminator(self, train_gen, valid_gen, steps_per_epoch=100, epochs=1, out_dir='../out/', out_period=1):
        """
        train discriminator
        
        Args:
            train_gen (generator): generator which provides mini-batch train data
            valid_gen (generator): generator which provides mini-batch validation data 
            steps_per_epoch (int): steps per epoch for training discriminator
            epochs (int): train epochs
            out_dir (str): out_dir for weights, logs, sample images
            out_period (int): output interval epochs.
        """
        
        def loss_disc(y_true, y_pred):
            """
            binary_crossentropy considering one-side label smoothing
            """
            return losses.binary_crossentropy(y_true*0.9, y_pred)
        
        if self.discriminator_train_model is None:
            if self.gpus > 1:
                self.discriminator_train_model = multi_gpu_model(self.discriminator(), gpus=self.gpus)
            else:
                self.discriminator_train_model = self.discriminator()
        
            # set trainable flag and recompile        
            self.generator().trainable = False
            self.classifier().trainable = False
            self.discriminator().trainable = True
            self.lcnn.extractor().trainable = False
            self.discriminator_train_model.compile(optimizer=Adam(lr=0.0001), loss=loss_disc, metrics=['binary_accuracy'])
                
        callbacks = []
        callbacks.append(TensorBoard(log_dir=out_dir+'logs/discriminator/'))
        callbacks.append(self.SaveWeightsCallback(target_models=[self.discriminator()], out_dir=out_dir+'weights/', period=out_period))
        
        history = self.discriminator_train_model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                                    epochs=epochs+self.disc_current_epochs, callbacks=callbacks,
                                    workers=0, validation_data=valid_gen, validation_steps=100,
                                    shuffle=False, initial_epoch=self.disc_current_epochs)
        self.disc_current_epochs += epochs
        
        return history
    
    def generate(self, inputs):
        """
        generate frontal image
        """
        
        if self.generator is None:
            self._init_generator()
        
        img128, img64, img32, fc2, front_leye_img, front_reye_img, front_nose_img, front_mouth_img\
        = self.generator().predict(inputs)
        
        
        img128 = (img128*np.iinfo(np.uint8).max).astype(np.uint8)
        img64 = (img64*np.iinfo(np.uint8).max).astype(np.uint8)
        img32 = (img32*np.iinfo(np.uint8).max).astype(np.uint8)
        front_leye_img = (front_leye_img*np.iinfo(np.uint8).max).astype(np.uint8)
        front_reye_img = (front_reye_img*np.iinfo(np.uint8).max).astype(np.uint8)
        front_nose_img = (front_nose_img*np.iinfo(np.uint8).max).astype(np.uint8)
        front_mouth_img = (front_mouth_img*np.iinfo(np.uint8).max).astype(np.uint8)
        
        return img128, img64, img32, front_leye_img, front_reye_img, front_nose_img, front_mouth_img
    
    def rotate_parts(self, inputs):
        """
        generate rotated part images
        """
        
        out_leyes, _, out_reyes, _, out_noses, _, out_mouthes, _ = self.parts_rotator().predict(inputs)
        
        out_leyes = (out_leyes*np.iinfo(np.uint8).max).astype(np.uint8)
        out_reyes = (out_reyes*np.iinfo(np.uint8).max).astype(np.uint8)
        out_noses = (out_noses*np.iinfo(np.uint8).max).astype(np.uint8)
        out_mouthes = (out_mouthes*np.iinfo(np.uint8).max).astype(np.uint8)
        
        return out_leyes, out_reyes, out_noses, out_mouthes
    
    def discriminate(self, frontal_img):
        """
        discriminate frontal image.
        
        Returns: discriminated score map
        """
        
        if self.discriminator is None:
            self._init_discriminator()
            
        out_img = self.discriminator().predict(frontal_img[np.newaxis, ...])[0]
        out_img = (out_img*np.iinfo(np.uint8).max).astype(np.uint8)
        out_img = cv2.resize(out_img, (frontal_img.shape[1], frontal_img.shape[0]))
        
        return out_img
    

class LightCNN():
    
    class SaveWeightsCallback(Callback):
    
        def __init__(self, target_models, out_dir, period):
            self.target_models = target_models
            self.out_dir = out_dir
            self.period = period
    
        def on_epoch_end(self, epoch, logs):
            if (epoch + 1) % self.period == 0:
                for target_model in self.target_models:
                    target_model.save_weights(self.out_dir + 'weights/lcnn_finetune/epoch{epoch:04d}_lr{lr:.5f}_loss{loss:.3f}_valacc{val_acc:.3f}.hdf5'.format(epoch=epoch + 1, lr=K.get_value(self.model.optimizer.lr), loss=logs['loss'], val_acc=logs['val_acc']), overwrite=True)
                    
    def __init__(self, classes=None, extractor_type='29v2', extractor_weights=None, classifier_weights=None, in_size_hw=(128, 128)):
        """
        initialize light cnn network with given weights file. if weights file is None, the weights are initialized by default initializer.
        
        Args:
            classes (int): number of output classes. required when training or using classifier. not required when using only exractor.
            extractor_type (str): string of network type. must be one of the following strings "29v2", "29", "9".
            extractor_weights (str): trained extractor weights file path. it is used to resume training. not required when train from scratch.
            classifier_weights (str): trained classifier weights file path. it is used to resume training. not required when training from scratch or only using extractor.
            in_size_hw (tuple): height and width of input image. 
        """

        self.in_size_hw = in_size_hw
        self.num_classes = classes
        self.extractor_weights = extractor_weights
        self.classifier_weights = classifier_weights
        self._extractor = None
        self._classifier = None
        
        # if extractor_weights is not None, attempt to resume current epoch number from file name.
        if self.extractor_weights is not None:
            try:
                self.current_epochs = int(re.match(r'.+[_h]([0-9]+)\.hdf5', self.extractor_weights).groups()[0])
            except:
                print('trained epochs was not found in extractor_weights_file name. use 0 as current_epochs.')
                self.current_epochs = 0
        else:
            self.current_epochs = 0
            
        self.extractor_type = extractor_type
            
        
    def extractor(self):
        """
        getter for singleton extractor.
        """
        
        if self._extractor is None:
            if self.extractor_type == '29v2':
                self._extractor = self.build_extractor_29layers_v2(name='extract29v2', block=self._res_block, layers=[1, 2, 3, 4])
            elif self.extractor_type == '29':
                self._extractor = self.build_extractor_29layers(name='extract29', block=self._res_block, layers=[1, 2, 3, 4])
            elif self.extractor_type == '9':
                self._extractor = self.build_extractor_9layers(name='extract9')
        
            if self.extractor_weights is not None:
                self._extractor.load_weights(self.extractor_weights)
                
        return self._extractor
    
    def classifier(self):
        """
        getter for singleton classifier.
        """
        
        if self._classifier is None:
            self._classifier = self.build_classifier(name='classify')
            
        if self.classifier_weights is not None:
            self._classifier.load_weights(self.classifier_weights)
        
        return self._classifier
    
    
    def _mfm(self, X, name, out_channels, kernel_size=3, strides=1, dense=False):
        """
        private func for creating mfm layer.
        
        Todo:
            * maybe more natural if implemented as custom layer like the comment out code at the bottom of this file.
        """
        
        if dense:
            X = Dense(out_channels*2, name = name + '_dense1', kernel_regularizer=regularizers.l2(0.0005))(X)
        else:
            X = Conv2D(out_channels*2, name = name + '_conv2d1', kernel_size=kernel_size, kernel_regularizer=regularizers.l2(0.0005), strides=strides, padding='same')(X)
            
        X = Maximum()([Lambda(lambda x, c: x[..., :c], arguments={'c':out_channels})(X), Lambda(lambda x, c: x[..., c:], arguments={'c':out_channels})(X)])
        
        return X
    
    def _group(self, X, name, in_channels, out_channels, kernel_size, strides):
        
        X = self._mfm(X, name = name + '_mfm1', out_channels=in_channels, kernel_size=1, strides=1, dense=False)
        X = self._mfm(X, name = name + '_mfm2', out_channels=out_channels, kernel_size=kernel_size, strides=strides)
        
        return X
    
    def _res_block(self, X, name, out_channels):
        """
        private func for creating residual block with mfm layers.
        """
        
        X_shortcut = X
        X = self._mfm(X, name = name + '_mfm1', out_channels=out_channels, kernel_size=3, strides=1)
        X = self._mfm(X, name = name + '_mfm2', out_channels=out_channels, kernel_size=3, strides=1)
        X = Add()([X, X_shortcut])
        return X
    
    def _make_layer(self, X, name, block, num_blocks, out_channels):
        """
        private func for creating multiple blocks. block is usualy res_block.
        """
        
        for i in range(0, num_blocks):
            X = block(X, name = name + '_block{}'.format(i), out_channels=out_channels)
        return X

    def build_extractor_9layers(self, name):
        
        in_img = Input(shape=(*self.in_size_hw, 1))
        
        X = self._mfm(in_img, name = name + '_mfm1', out_channels=48, kernel_size=5, strides=1)
        X = MaxPooling2D(pool_size=2, padding='same')(X)
        X = self._group(X, name = name + '_group1', in_channels=48, out_channels=96, kernel_size=3, strides=1)
        X = MaxPooling2D(pool_size=2, padding='same')(X)        
        X = self._group(X, name = name + '_group2', in_channels=96, out_channels=192, kernel_size=3, strides=1)
        X = MaxPooling2D(pool_size=2, padding='same')(X) 
        X = self._group(X, name = name + '_group3', in_channels=192, out_channels=128, kernel_size=3, strides=1)
        X = self._group(X, name = name + '_group4', in_channels=128, out_channels=128, kernel_size=3, strides=1)
        feat_map = MaxPooling2D(pool_size=2, padding='same')(X)       
        feat_vec = Dense(256, name = name + '_dense1', kernel_regularizer=regularizers.l2(0.0005))(Flatten()(X))

        ret_extractor = CloudableModel(inputs=in_img, outputs=[feat_vec, feat_map], name=name)
        ret_extractor.summary()
        
        return ret_extractor
    
    def build_extractor_29layers(self, name, block, layers):
        
        in_img = Input(shape=(*self.in_size_hw, 1))
        
        X = self._mfm(in_img, name = name + '_mfm1', out_channels=48, kernel_size=5, strides=1)
        X = MaxPooling2D(pool_size=2, padding='same')(X)
        X = self._make_layer(X, name = name + '_layers1', block=block, num_blocks=layers[0], out_channels=48)
        X = self._group(X, name = name + '_group1', in_channels=48, out_channels=96, kernel_size=3, strides=1)
        X = MaxPooling2D(pool_size=2, padding='same')(X)
        X = self._make_layer(X, name = name + '_layers2', block=block, num_blocks=layers[1], out_channels=96)
        X = self._group(X, name = name + '_group2', in_channels=96, out_channels=192, kernel_size=3, strides=1)
        X = MaxPooling2D(pool_size=2, padding='same')(X)
        X = self._make_layer(X, name = name + '_layers3', block=block, num_blocks=layers[2], out_channels=192)
        X = self._group(X, name = name + '_group3', in_channels=192, out_channels=128, kernel_size=3, strides=1)
        X = self._make_layer(X, name = name + '_layers4', block=block, num_blocks=layers[3], out_channels=128)
        X = self._group(X, name = name + '_group4', in_channels=128, out_channels=128, kernel_size=3, strides=1)
        feat_map = MaxPooling2D(pool_size=2, padding='same')(X)
        feat_vec = self._mfm(Flatten()(feat_map), name = name + '_mfm2', out_channels=256, dense=True)

        ret_extractor = CloudableModel(inputs=in_img, outputs=[feat_vec, feat_map], name=name)
        ret_extractor.summary()
        
        return ret_extractor
                    
    def build_extractor_29layers_v2(self, name, block, layers):
        
        in_img = Input(shape=(*self.in_size_hw, 1))
        
        X = self._mfm(in_img, name = name + '_mfm1', out_channels=48, kernel_size=5, strides=1)
        X = Average()([MaxPooling2D(pool_size=2, padding='same')(X), AveragePooling2D(pool_size=2, padding='same')(X)])
        X = self._make_layer(X, name = name + '_layers1', block=block, num_blocks=layers[0], out_channels=48)
        X = self._group(X, name = name + '_group1', in_channels=48, out_channels=96, kernel_size=3, strides=1)
        X = Average()([MaxPooling2D(pool_size=2, padding='same')(X), AveragePooling2D(pool_size=2, padding='same')(X)])
        X = self._make_layer(X, name = name + '_layers2', block=block, num_blocks=layers[1], out_channels=96)
        X = self._group(X, name = name + '_group2', in_channels=96, out_channels=192, kernel_size=3, strides=1)
        X = Average()([MaxPooling2D(pool_size=2, padding='same')(X), AveragePooling2D(pool_size=2, padding='same')(X)])
        X = self._make_layer(X, name = name + '_layers3', block=block, num_blocks=layers[2], out_channels=192)
        X = self._group(X, name = name + '_group3', in_channels=192, out_channels=128, kernel_size=3, strides=1)
        X = self._make_layer(X, name = name + '_layers4', block=block, num_blocks=layers[3], out_channels=128)
        X = self._group(X, name = name + '_group4', in_channels=128, out_channels=128, kernel_size=3, strides=1)
        feat_map = Average()([MaxPooling2D(pool_size=2, padding='same')(X), AveragePooling2D(pool_size=2, padding='same')(X)])
        feat_vec = Dense(256, name = name + '_dense1', kernel_regularizer=regularizers.l2(0.0005))(Flatten()(feat_map))
        
        ret_extractor = CloudableModel(inputs=in_img, outputs=[feat_vec, feat_map], name=name)        
        ret_extractor.summary()
        
        return ret_extractor
    
    def build_classifier(self, name):
        
        in_feat = Input(shape=(256,))
        X = Dropout(0.7)(in_feat)
               
        X = Dense(500, activation='relu', name = name + '_dense1', kernel_regularizer=regularizers.l2(0.005))(X)
        X = Dropout(0.7)(X)
    
        clas = Dense(self.num_classes, activation='softmax', name = name + '_dense2', use_bias=False , kernel_regularizer=regularizers.l2(0.005))(X)
        
        ret_classifier = CloudableModel(inputs=in_feat, outputs=clas, name=name)
        
        ret_classifier.summary()
        return ret_classifier
        

    def train(self, train_gen, valid_gen=None, optimizer=SGD(lr=0.001, momentum=0.9, decay=0.00004, nesterov=True),
              classifier_dropout=0.7, steps_per_epoch=100, validation_steps=100,
              epochs=1, out_dir='../out/', out_period=1, fix_extractor=False):
        """
        train extractor and classifier.
        
        Args:
            train_gen (generator): train data generator provided by celeb_gen.
            valid_gen (generator): valid data generator provided by celeb_gen.
            optimizer (Optimizer): keras optimizer used to train.
            classifier_dropout (float): dropout ratio for training classifier.
            steps_per_epoch (int): steps for each epoch. 
            validation_steps (int): steps for validation on the end of each epoch.
            epochs (int): epochs to train.
            out_prefix (str): prefix str for output weights file.
            out_period (int): interval epochs for output weights file.
            fix_extractor (bool): if true, train only classifier. if false, train both extractor and classifier.
        """        
        
        self.classifier().trainable = True
        self.extractor().trainable = not fix_extractor           
        
        for layer in self.classifier().layers:
            if type(layer) == Dropout:
                layer.rate = classifier_dropout
        
        train_model = Sequential([CloudableModel(inputs=self.extractor().inputs, outputs=self.extractor().outputs[0]), self.classifier()])
        train_model.summary()
        
        train_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
        
        if out_dir != '':
            tf.gfile.MakeDirs(out_dir)
        
        callbacks = []
        callbacks.append(TensorBoard(log_dir='./logs/lcnn_finetune'))
        callbacks.append(self.SaveWeightsCallback(target_models=[self.extractor(), self.classifier()], out_dir=out_dir+'weights/lcnn_finetune/', period=out_period))            
        history = train_model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs+self.current_epochs, callbacks=callbacks, workers=0, validation_data=valid_gen, validation_steps=validation_steps, initial_epoch=self.current_epochs)
        self.current_epochs += epochs
                    
        return history
    
    def evaluate(self, generator, steps=100):
        
        train_model = Sequential([CloudableModel(inputs=self.extractor().inputs, outputs=self.extractor().outputs[0]), self.classifier()])
        train_model.summary()
        
        train_model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['acc'])

        score = train_model.evaluate_generator(generator, steps=steps)
                    
        return {'loss':score[0], 'acc':score[1]}
    
    def predict_class(self, gray_img_batch):
        feat = self.extractor().predict(gray_img_batch)
        clas = self.classifier().predict(feat)
        return np.argmax(clas, axis=-1)
    
    def exract_features(self, gray_img_batch):
        feat = self.extractor().predict(gray_img_batch)
        return feat
    
class CloudableModel(Model):
    """
    wrapper of keras model. this class override some functions to be available for Google Cloud Strorage.
    """
    
    def load_weights(self, filepath, by_name=False):
        print('begin loading weights file. target file: {}'.format(filepath))
        
        if len(filepath) >= 3 and filepath[:3] == 'gs:':
            tmp_file = 'tmp_' + os.path.basename(filepath)
            print('start copy weights file in GCS to local disk. {} -> {}'.format(filepath, tmp_file))
            tf.gfile.Copy(filepath, tmp_file, overwrite=True)
            filepath = tmp_file
            print('copy completed. copied file size:{}'.format(os.path.getsize(filepath)))

        super().load_weights(filepath=filepath, by_name=by_name)
        print('end loading weights file. target file: {}'.format(filepath))
    
    def save_weights(self, filepath, overwrite=True):
        print('begin saving weights file. target file: {}'.format(filepath))
        
        if len(filepath) >= 3 and filepath[:3] == 'gs:':
            tmp_file = 'tmp_' + os.path.basename(filepath)
            super().save_weights(tmp_file, overwrite=overwrite)
            
            print('start copy weights file in local disk to GCS {} -> {}'.format(tmp_file, filepath))
            tf.gfile.Copy(tmp_file, filepath, overwrite=overwrite)
            print('copy completed. copied file size:{}'.format(os.path.getsize(tmp_file)))
        else:
            super().save_weights(filepath, overwrite=overwrite)
            
        print('end saving weights file. target file: {}'.format(filepath))
        
def Open(name, mode='r'):
    """
    because, in my environment, sometimes gfile.Open is very slow when target file is localpath(not gs://),
    so, use normal open if the target path is localpath.
    """
    if len(name) >= 5 and name[:5] == 'gs://':
        return tf.gfile.Open(name, mode)
    else:
        return open(name, mode)

        
