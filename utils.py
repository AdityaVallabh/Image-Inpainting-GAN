from model import generator
import tensorflow as tf
import os

def parse_tfrecords(data_dir, batch_size, s_size, FLAGS):
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tfrecords')]
    fqueue = tf.train.string_input_producer(files)
    reader = tf.TFRecordReader()
    _, serialized = reader.read(fqueue)
    features = tf.parse_single_example(serialized, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    image = tf.cast(tf.decode_raw(features['image_raw'], tf.uint8), tf.float32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image = tf.reshape(image, [height, width, FLAGS.nb_channels])
    image = tf.image.resize_image_with_crop_or_pad(image, FLAGS.crop_image_size, FLAGS.crop_image_size)
    min_queue_examples = FLAGS.num_examples_per_epoch
    images = tf.train.shuffle_batch(
        [image],
        batch_size=batch_size,
        capacity=min_queue_examples + FLAGS.nb_channels * batch_size,
        min_after_dequeue=min_queue_examples
    )
    return tf.subtract(tf.div(tf.image.resize_images(images, [s_size * 2 ** 4, s_size * 2 ** 4]), 127.5), 1.0)

def collage(inputs, row, col, start_size, batch_size):
    images = generator(inputs, start_size, is_training=True, reuse=tf.AUTO_REUSE)
    images = tf.image.convert_image_dtype(tf.div(tf.add(images, 1.0), 2.0), tf.uint8)
    images = [image for image in tf.split(images, batch_size, axis=0)]
    rows = [tf.concat(images[col * i + 0:col * i + col], 2) for i in range(row)]
    image = tf.concat(rows, 1)
    return tf.image.encode_jpeg(tf.squeeze(image, [0]))
