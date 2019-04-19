import argparse
import tensorflow as tf
import os
import numpy as np
from data_utils import DataSetFromFolder
from model import Net
import time
from PIL import Image
from psnr import *

class ESPCN(object):
    def __init__(self, upscale_factor, epochs, crop_size, batch_size, learning_rate):
        self.upscale_factor = upscale_factor
        self.epochs = epochs
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.logdir = './log'
        
        self.train_data = DataSetFromFolder('data/train', upscale_factor=UPSCALE_FACTOR, crop_size=CROP_SIZE, batch_size=BATCH_SIZE)
        self.val_data = DataSetFromFolder('data/val', upscale_factor=UPSCALE_FACTOR, crop_size=CROP_SIZE, batch_size=BATCH_SIZE)
        
        with tf.name_scope('input'):
            self.input = tf.placeholder(dtype = tf.float32)
            self.label = tf.placeholder(dtype = tf.float32)
            
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        
        self.net = Net(upscale_factor)
        pred = self.net.build_network(self.input)
        self.loss = self.net.loss(pred,self.label)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        trainable = tf.trainable_variables()
        self.op = optimizer.minimize(self.loss, var_list=trainable) 

        with tf.name_scope('sampler'):
            self.sampler = tf.identity(self.net.build_network(self.input))
        
        with tf.name_scope('summary'):
            self.summaries =  tf.summary.scalar('Loss ',self.loss)
            self.writer = tf.summary.FileWriter(self.logdir)
            self.writer.add_graph(tf.get_default_graph())
        
    def train(self):
        sess = tf.Session()
        saver = tf.train.Saver()
        
        init = tf.initialize_all_variables()
        sess.run(init)
        
        if self.net.load(sess, saver, self.logdir):
            print("[*] Checkpoint load success!")
        else:
            print("[*] Checkpoint load failed/no checkpoint found")
        steps, start_average, end_average = 0, 0, 0 
        start_time = time.time()
        for ep in range(1, self.epochs + 1):
            batch_average = 0
            for batch_image, batch_target in self.train_data:  
                summary, loss_value, _ = sess.run([self.summaries, self.loss, self.op],
                                                  feed_dict = {self.input:batch_image, self.label:batch_target})
                self.writer.add_summary(summary,steps)
                batch_average += loss_value
            
            batch_average = float(batch_average) / self.train_data.num_batch
            if ep < (self.epochs * 0.2):
                start_average += batch_average
            elif ep >= (self.epochs * 0.8):
                end_average += batch_average
            duration = time.time() - start_time
            print('Epoch: {}, step: {:d}, loss: {:.9f}, ({:.3f} sec/epoch)'.format(ep, steps, batch_average, duration))
            start_time = time.time()
            self.net.save(sess, saver, self.logdir, steps)
            
            if ep % 1 == 0:
                
                image, target,target_cb, target_cr = self.train_data.get_sampler()
                batch_image = np.zeros((1, self.crop_size // self.upscale_factor, self.crop_size // self.upscale_factor, 1))
                batch_target = np.zeros((1, self.crop_size, self.crop_size, 1))
                batch_image[0, :, :, :] = image
                batch_target[0, :, :, :] = target
                
                sr_image, _ = sess.run([self.sampler, self.op],
                                feed_dict = {self.input:batch_image, self.label:batch_target})
                
                target *= 255.0
                hr_image = np.reshape(target, [256,256])
                hr_image = hr_image.clip(0, 255)
                hr_image = Image.fromarray(np.uint8(hr_image), mode='L')
                out_hr_image = Image.merge('YCbCr', [hr_image, target_cb, target_cr]).convert('RGB')
                out_hr_image.save("./result/hr_image.jpg")
                
                
                sr_image *= 255.0
                sr_image = np.reshape(sr_image, [256,256])
                sr_image = sr_image.clip(0, 255)
                sr_image = Image.fromarray(np.uint8(sr_image), mode='L')
                
                out_sr_image = Image.merge('YCbCr', [sr_image, target_cb, target_cr]).convert('RGB')
                out_sr_image.save("./result/sr_image.jpg")

                psnr_val = psnr(out_hr_image, out_sr_image)
                print("PSNR : {}".format(psnr_val))                  
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train Super Resolution')
    parser.add_argument('--upscale_factor', default=2, type=int, help='Super Resolution Upscale Factor')
    parser.add_argument('--epochs', default=300, type=int, help='Training Epochs')
    parser.add_argument('--crop_size', default=256, type=int, help='HR Image Size')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Batch Size')
    flags = parser.parse_args()
    
    UPSCALE_FACTOR = flags.upscale_factor
    EPOCHS = flags.epochs
    CROP_SIZE = flags.crop_size
    BATCH_SIZE = flags.batch_size 
    LEARNING_RATE = flags.learning_rate
    
    espcn = ESPCN(UPSCALE_FACTOR, EPOCHS, CROP_SIZE, BATCH_SIZE, LEARNING_RATE)
    espcn.train()