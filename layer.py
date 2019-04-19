import tensorflow as tf

def convolutional(name,input_data,filters_shape,strides,padding):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True, shape=filters_shape,
                                 initializer=tf.random_normal_initializer(stddev=0.01))
        bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                               dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        
        conv = tf.nn.conv2d(input=input_data,filter=weight,strides=strides,padding=padding)
        conv = tf.nn.bias_add(conv,bias)
    return conv