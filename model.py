import tensorflow as tf
from layer import * 
import sys
import os
class Net(object):
    def __init__(self, upscale_factor):
        self.upscale_factor = upscale_factor
    
    def build_network(self, input_data, val_reuse=False):
        with tf.variable_scope('ESPCN', reuse = val_reuse):
            conv1 = convolutional(name='conv1', input_data = input_data, filters_shape=(5, 5, 1, 64), strides=[1,1,1,1], padding='SAME')
            conv1 = tf.tanh(conv1)
            conv2 = convolutional(name='conv2', input_data = conv1, filters_shape=(3, 3, 64, 32), strides=[1,1,1,1], padding='SAME')
            conv2 = tf.tanh(conv2)
            conv3 = convolutional(name='conv3', input_data = conv2, filters_shape=(3, 3, 32, 1* (self.upscale_factor**2)), strides=[1,1,1,1], padding='SAME')
            conv3 = tf.tanh(conv3)
            out = tf.depth_to_space(conv3, self.upscale_factor)
            out = tf.tanh(out)
        return out
      
    def loss(self, pred, label):
        with tf.name_scope('loss'):
            loss = tf.losses.mean_squared_error(pred,label)
        return loss
    
    def save(self, sess, saver, logdir, step):
        sys.stdout.flush()
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        checkpoint = os.path.join(logdir,"model.ckpt")
        saver.save(sess,checkpoint,global_step = step)
        
    def load(self,sess,saver,logdir):
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(logdir, ckpt_name))
            return True
        else:
            return False