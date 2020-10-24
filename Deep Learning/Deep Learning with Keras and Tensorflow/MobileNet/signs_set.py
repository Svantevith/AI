import os
import shutil
import random


def prepare_signs_dataset():
    os.chdir('data\MobileNet Samples\Sign Language Samples')
    if os.path.isdir('train/0/') is False:
        # if train folder inside Sign Language Samples does not exist, create subfolders train, valid and test
        os.mkdir('train')
        os.mkdir('valid')
        os.mkdir('test')

        # for(int i = 0; i < 10; i++)
        for i in range(0, 10):
            # copy all sub folders 0,1,..9 to the train folders
            shutil.move(f'{i}', 'train')
            # additionally create sub folders 0,1,..9 inside of the valid and test folders
            os.mkdir(f'valid/{i}')
            os.mkdir(f'test/{i}')

            # gather 30 random samples from each 0,1,..9 subfolders in the train folder
            valid_samples = random.sample(os.listdir(f'train/{i}'), 30)
            # for each random sample, copy that sample from train/i directory into valid/i directory
            for j in valid_samples:
                shutil.move(f'train/{i}/{j}', f'valid/{i}')

            # gather 5 random samples from each 0,1,..9 subfolders in the train folder
            test_samples = random.sample(os.listdir(f'train/{i}'), 5)
            # for each random sample, copy that sample from the train/i directory into test/i directory
            for k in test_samples:
                shutil.move(f'train/{i}/{k}', f'test/{i}')

    os.chdir('../../../')
