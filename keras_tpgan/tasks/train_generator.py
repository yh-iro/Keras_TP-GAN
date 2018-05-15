import argparse

import os
from keras_tpgan.tpgan import TPGAN, multipie_gen
from tensorflow.contrib.training.python.training import hparam
from keras.optimizers import SGD, Adam

def run(hparams):
    
    gan = TPGAN(base_filters=64, gpus=hparams.gpus,
                lcnn_extractor_weights=hparams.lcnn_weights,
                generator_weights=hparams.generator_weights,
                classifier_weights=hparams.classifier_weights,
                discriminator_weights=hparams.discriminator_weights)

    datagen = multipie_gen.Datagen(dataset_dir=hparams.multipie_dir, landmarks_dict_file=hparams.landmarks_dict_file, 
                                   datalist_dir=hparams.datalist_dir, valid_count=4)
    train_gen = datagen.get_generator(setting='train', batch_size=hparams.batch_size)
    valid_gen = datagen.get_generator(setting='valid', batch_size=4)

    if hparams.optimizer == 'adam':
        optimizer = Adam(lr=hparams.lr, beta_1=hparams.beta1, beta_2=hparams.beta2, decay=hparams.decay)
    elif hparams.optimizer == 'sgd':
        optimizer = SGD(lr=hparams.lr, momentum=hparams.beta1, decay=hparams.decay, nesterov=True)
    else:
        raise Exception('undefined optimizer type: "{}"'.format(hparams.optimizer))
        
    gan.train_generator(train_gen, valid_gen, steps_per_epoch=hparams.steps_per_epoch,
                        epochs=hparams.epochs, is_output_img=True,
                        out_dir=hparams.out_dir, out_period=hparams.out_period,
                        lr=hparams.lr, decay=hparams.decay,
                        optimizer=optimizer,
                        lambda_128=hparams.lambda_128, lambda_64=hparams.lambda_64, lambda_32=hparams.lambda_32,
                        lambda_sym=hparams.lambda_sym, lambda_ip=hparams.lambda_ip, lambda_adv=hparams.lambda_adv, 
                        lambda_tv=hparams.lambda_tv, lambda_class=hparams.lambda_class, lambda_parts=hparams.lambda_parts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--multipie_dir',
        help='multipie dataset dir',
        default='C:/work_tmp/MULTI-PIE_jpg/',
    ) 
    parser.add_argument(
        '--landmarks_dict_file',
        help='landmarks dict file (pickle format) of multipie dataset',
        default='landmarks.pkl',
    ) 
    parser.add_argument(
        '--datalist_dir',
        help='dataliset dir.',
        default='datalist/',
    ) 
    parser.add_argument(
        '--lcnn-weights',
        help='weights file of LCNN feature extractor',
        required=True
    )
    parser.add_argument(
        '--generator-weights',
        help='weights file of generator',
        default=None
    )
    parser.add_argument(
        '--classifier_weights',
        help='weights file of classifier',
        default=None
    )
    parser.add_argument(
        '--discriminator_weights',
        help='weights file of discriminator',
        default=None
    )
    parser.add_argument(
        '--gpus',
        help='use gpu count',
        default=1,
        type=int
    )
    parser.add_argument(
        '--batch_size',
        help='batch_size',
        default=64,
        type=int
    )
    parser.add_argument(
        '--steps_per_epoch',
        help='steps_per_epoch',
        default=300,
        type=int
    )
    parser.add_argument(
        '--epochs',
        help='epochs',
        default=100,
        type=int
    )
    parser.add_argument(
        '--out_dir',
        help='out dir for model, images, log ',
        default='out/'
    )
    parser.add_argument(
        '--out_period',
        help='interval epoch count for model and images',
        default=5,
        type=int
    )
    parser.add_argument(
        '--optimizer',
        help='optimizer type',
        default='sgd',
        type=str
    )    
    parser.add_argument(
        '--lr',
        help='start learning rate',
        default=0.001,
        type=float
    )
    parser.add_argument(
        '--decay',
        help='decay of learning rate',
        default=0.0,
        type=float
    )
    parser.add_argument(
        '--beta1',
        help='beta1 in Adam, momentum in SGD',
        default=0.9,
        type=float
    )    
    parser.add_argument(
        '--beta2',
        help='beta2 in Adam',
        default=0.999,
        type=float
    )    
    parser.add_argument(
        '--lambda_128',
        help='lambda for 128 image loss',
        required=True,
        type=float
    )
    parser.add_argument(
        '--lambda_64',
        help='lambda for 64 image loss',
        required=True,
        type=float
    )
    parser.add_argument(
        '--lambda_32',
        help='lambda for 32 image loss',
        required=True,
        type=float
    )
    parser.add_argument(
        '--lambda_sym',
        help='lambda for symmetricity loss',
        required=True,
        type=float
    )
    parser.add_argument(
        '--lambda_ip',
        help='lambda for identity preserve loss',
        required=True,
        type=float
    )
    parser.add_argument(
        '--lambda_adv',
        help='lambda for adversarial loss',
        required=True,
        type=float
    )
    parser.add_argument(
        '--lambda_tv',
        help='lambda for total variation loss',
        required=True,
        type=float
    )
    parser.add_argument(
        '--lambda_class',
        help='lambda for classification loss',
        required=True,
        type=float
    )
    parser.add_argument(
        '--lambda_parts',
        help='lambda for part patch image pixel loss',
        required=True,
        type=float
    )
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
    )


    # Run the training job
    hparams=hparam.HParams(**parser.parse_args().__dict__)
    if hparams.job_dir is not None:
        hparams.out_dir = os.path.join(hparams.job_dir, hparams.out_dir)
    
    run(hparams)
