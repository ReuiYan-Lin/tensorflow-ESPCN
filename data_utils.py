import argparse
import os
#import tensorflow as tf
from os import listdir
from os.path import join
from PIL import Image
from tqdm import tqdm
import numpy as np
import random

TRAIN_DATA = 'D:/github/VOCdevkit/VOC2012/JPEGImages/'

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png','jpg'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def input_transform(image, crop_size, upscale_factor):
    return image.resize((crop_size // upscale_factor, crop_size // upscale_factor), Image.BICUBIC)

def target_transform(image, crop_size):
    return image.resize((crop_size, crop_size), Image.BICUBIC)

class DataSetFromFolder(object):
    def __init__(self, dataset_dir, upscale_factor, crop_size, batch_size):
        super(DataSetFromFolder, self).__init__()
        self.image_dir = dataset_dir + '/SRF_' + str(upscale_factor) + '/data'
        self.target_dir = dataset_dir + '/SRF_' + str(upscale_factor) + '/target'
        self.image_filenames = [join(self.image_dir, x) for x in listdir(self.image_dir) if is_image_file(x)]
        self.target_filenames = [join(self.target_dir,x) for x in listdir(self.target_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.batch_count = 0
        self.num_batch = int(np.ceil(len(self.image_filenames) // self.batch_size))

    def __iter__(self):
        return self    
    
    def __next__(self):
        batch_image = np.zeros((self.batch_size, self.crop_size // self.upscale_factor, self.crop_size // self.upscale_factor, 1))
        batch_target = np.zeros((self.batch_size, self.crop_size, self.crop_size, 1))
        num = 0
        
        if self.batch_count < self.num_batch:
            while num < self.batch_size:
                index = self.batch_count * self.batch_size + num
                if index >= len(self.image_filenames):
                    index -= len(self.image_filenames)
                    
                image, _, _ = Image.open(self.image_filenames[index]).convert('YCbCr').split()
                target, _, _ = Image.open(self.target_filenames[index]).convert('YCbCr').split()
                image = np.asarray(image).reshape([self.crop_size // self.upscale_factor, self.crop_size // self.upscale_factor, 1])
                target = np.asarray(target).reshape([self.crop_size, self.crop_size, 1])
                image, target = normalization(image), normalization(target)
                batch_image[num, :, :, :] = image
                batch_target[num, :, :, :] = target
                
                num += 1 
            self.batch_count += 1
            
            return batch_image, batch_target
        else:
            self.batch_count = 0
            #np.random.shuffle(self.image_filenames)
            raise StopIteration
            
    def get_sampler(self):
        index = np.random.choice(len(self.image_filenames))
        image, _, _ = Image.open(self.image_filenames[index]).convert('YCbCr').split()
        target, target_cb, target_cr = Image.open(self.target_filenames[index]).convert('YCbCr').split()
        image = np.asarray(image).reshape([self.crop_size // self.upscale_factor, self.crop_size // self.upscale_factor, 1])
        target = np.asarray(target).reshape([self.crop_size, self.crop_size, 1])
        image, target = normalization(image), normalization(target)
        
        return image, target, target_cb, target_cr
        
    
    def __len__(self):
        return len(self.image_filenames)

def normalization(image):
        return image/255.0        

def generate_dataset(data_type, upscale_factor):
    images_name = [x for x in listdir(TRAIN_DATA) if is_image_file(x)]
    if data_type == 'train':
        images_name = images_name[:10000]
    elif data_type == 'val':
        images_name = images_name[10000:]
    else:
        return
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    
    root = 'data/' + data_type
    if not os.path.exists(root):
        os.makedirs(root)
    path = root + '/SRF_' + str(upscale_factor)
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = path + '/data'
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    target_path = path + '/target'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        
    for image_name in tqdm(images_name, desc='generate ' + data_type + ' dataset with upscale factor = '
                           + str(upscale_factor) + ' from VOC2012'):
        image = Image.open(TRAIN_DATA + image_name)
        target = image.copy()
        image = input_transform(image, crop_size, upscale_factor)
        target = target_transform(image, crop_size)
        
        image.save(image_path + '/' + image_name)
        target.save(target_path + '/' + image_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Super Resolution Dataset')
    parser.add_argument('--upscale_factor', default=2, type=int, help='Super Resolution Upscale Factor')
    flags = parser.parse_args()
    UPSCALE_FACTOR = flags.upscale_factor
    
    generate_dataset(data_type='train', upscale_factor=UPSCALE_FACTOR)
    generate_dataset(data_type='val', upscale_factor=UPSCALE_FACTOR)