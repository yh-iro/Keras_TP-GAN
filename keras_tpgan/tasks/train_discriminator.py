import argparse

import os
from keras_tpgan.tpgan import TPGAN, multipie_gen
from tensorflow.contrib.training.python.training import hparam

def run(hparams):
    
    gan = TPGAN(base_filters=64, gpus=hparams.gpus,
                lcnn_extractor_weights=None,
                generator_weights=hparams.generator_weights,
                classifier_weights=None,
                discriminator_weights=hparams.discriminator_weights)

    datagen = multipie_gen.Datagen(dataset_dir=hparams.multipie_dir, landmarks_dict_file=hparams.landmarks_dict_file, 
                                   datalist_dir=hparams.datalist_dir, valid_count=4)
    train_gen = datagen.get_discriminator_generator(gan.generator(), batch_size=hparams.batch_size,
                                                    gt_shape=gan.discriminator().output_shape[1:3],
                                                    setting = 'train')
    
    gan.train_discriminator(train_gen=train_gen, valid_gen=train_gen, steps_per_epoch=hparams.steps_per_epoch,
                                      epochs=hparams.epochs, out_dir=hparams.out_dir, out_period=hparams.out_period)

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
        '--generator-weights',
        help='weights file of generator',
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
        '--job-dir',
        help='GCS location to write checkpoints and export models',
    )


    # Run the training job
    hparams=hparam.HParams(**parser.parse_args().__dict__)
    if hparams.job_dir is not None:
        hparams.out_dir = os.path.join(hparams.job_dir, hparams.out_dir)
    
    run(hparams)
