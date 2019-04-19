import tensorflow as tf
import argparse
import numpy as np
from model import Net
from data_utils import normalization
from PIL import Image

class Generate(object):
    def __init__(self, image_path, ckpt_path, upscale_factor):
        self.image_path = image_path
        self.ckpt_path = ckpt_path
        self.upscale_factor = upscale_factor
        self.image, self.image_Cb, self.image_Cr = self.Read_Image(self.image_path)
        self.Create_Image(self.image, self.image_Cb, self.image_Cr)  
    
    def Read_Image(self, image_path):
        
        image, image_Cb, image_Cr = Image.open(self.image_path).convert('YCbCr').split()
        self.width, self.height = image.size
        image = np.asarray(image).reshape([self.width, self.height, 1])
        image = normalization(image)
        batch_images = np.zeros((1, self.width, self.height, 1))
        batch_images[0, :, :, :] = image
        return batch_images, image_Cb, image_Cr
    
    def Create_Image(self, image, Cb, Cr):
        
        self.net = Net(self.upscale_factor)
        
        with tf.name_scope('input'):
            self.input = tf.placeholder(dtype= tf.float32)
            self.label = tf.placeholder(dtype = tf.float32)
        
        with tf.name_scope('sampler'):
            self.sampler = tf.identity(self.net.build_network(self.input))
            
        
        sess = tf.Session()
        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        sess.run(init)
        
        if self.net.load(sess, saver, self.ckpt_path):
            print("[*] Checkpoint load success!")
        else:
            print("[*] Checkpoint load failed/no checkpoint found")
            return 
        
        sr_image = sess.run(self.sampler, feed_dict={self.input:image})
        sr_image *= 255.0
        sr_image = np.reshape(sr_image,(self.width*2, self.height*2))
        sr_image = Image.fromarray(np.uint8(sr_image), mode='L') # Here have some error, sr_image.shape=(2074,1646)(numpy) but the sr_image.size=(1646,2074)(PIL)

        Cb = Cb.resize((self.width*2, self.height*2), Image.BICUBIC)
        Cr = Cr.resize((self.width*2, self.height*2), Image.BICUBIC)
        print(sr_image.size)
        print(Cb.size)
        print(Cr.size)
        out_sr_image = Image.merge('YCbCr', [sr_image, Cb, Cr]).convert('RGB')
        out_path,extension  = self.image_path.split('.')
        out_path = out_path + '_HR.' + extension
        out_sr_image.save(out_path)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate HR Image')
    parser.add_argument('--image_path', type=str, help='Image Path')
    parser.add_argument('--ckpt_path', type=str, help='Checkpoint Path')
    parser.add_argument('--upscale_factor', default=2, type=int, help='Super Resolution Upscale Factor')
   
    
    flags = parser.parse_args()
    IMAGE_PATH = flags.image_path
    CKPT_PATH = flags.ckpt_path
    UPSCALE_FACTOR = flags.upscale_factor 
    Generate(IMAGE_PATH, CKPT_PATH, UPSCALE_FACTOR)