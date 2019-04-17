import tensorflow as tf

def generator(x, start_size, out_dims=[1024, 512, 256, 128, 3], is_training=True, reuse=False, name=''):
    inputs = tf.convert_to_tensor(x)
    with tf.variable_scope('generator', reuse=reuse):
        with tf.variable_scope('reshape'):
            outputs = dense(inputs, out_dims[0] * start_size * start_size)
            outputs = tf.reshape(outputs, [-1, start_size, start_size, 1024])
            outputs = relu(tf.layers.batch_normalization(outputs, training=is_training))
        with tf.variable_scope('deconv1'):
            outputs = conv2d_transpose(outputs, out_dims[1], kernel=5, stride=2, padding='SAME')
            outputs = relu(tf.layers.batch_normalization(outputs, training=is_training))
        with tf.variable_scope('deconv2'):
            outputs = conv2d_transpose(outputs, out_dims[2], kernel=5, stride=2, padding='SAME')
            outputs = relu(tf.layers.batch_normalization(outputs, training=is_training))
        with tf.variable_scope('deconv3'):
            outputs = conv2d_transpose(outputs, out_dims[3], kernel=5, stride=2, padding='SAME')
            outputs = relu(tf.layers.batch_normalization(outputs, training=is_training))
        with tf.variable_scope('deconv4'):
            outputs = conv2d_transpose(outputs, out_dims[4], kernel=5, stride=2, padding='SAME')
        with tf.variable_scope('tanh'):
            outputs = tf.tanh(outputs, name='outputs')

    return outputs

def discriminator(images, out_dims=[64, 128, 256, 512], is_training=True, reuse=False, name=''):
    inputs = tf.convert_to_tensor(images)
    with tf.name_scope('discriminator' + name), tf.variable_scope('discriminator', reuse=reuse):
        with tf.variable_scope('conv1'):
            outputs = conv2d(inputs, out_dims[0], kernel=5, stride=2, padding='SAME')
            outputs = lrelu(tf.layers.batch_normalization(outputs, training=is_training))
        with tf.variable_scope('conv2'):
            outputs = conv2d(outputs, out_dims[1], kernel=5, stride=2, padding='SAME')
            outputs = lrelu(tf.layers.batch_normalization(outputs, training=is_training))
        with tf.variable_scope('conv3'):
            outputs = conv2d(outputs, out_dims[2], kernel=5, stride=2, padding='SAME')
            outputs = lrelu(tf.layers.batch_normalization(outputs, training=is_training))
        with tf.variable_scope('conv4'):
            outputs = conv2d(outputs, out_dims[3], kernel=5, stride=2, padding='SAME')
            outputs = lrelu(tf.layers.batch_normalization(outputs, training=is_training))
        with tf.variable_scope('classify'):
            batch_size = outputs.get_shape()[0].value
            reshape = tf.reshape(outputs, [batch_size, -1])
            outputs = dense(reshape, 2, name='outputs')

    return outputs

def conv2d(x, output_dim, kernel=3, stride=2, padding='SAME'):
    return tf.layers.conv2d(x, output_dim, [kernel, kernel], strides=(stride, stride), padding=padding)

def conv2d_transpose(x, output_dim, kernel, stride ,padding):
    return tf.layers.conv2d_transpose(x, output_dim, [kernel, kernel], strides=(stride, stride), padding=padding)

def dense(x,output_size, activation=tf.nn.relu, name=''):
    return tf.layers.dense(x , output_size , activation, name=name)

def lrelu(x, threshold=0.2, name='outputs'):
    return tf.maximum(x, x * threshold, name=name)

def relu(x, name='outputs'):
    return tf.nn.relu(x, name='outputs')
