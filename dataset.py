import tensorflow as tf
import os

def dataset_filters(train_frac=0.8, test_frac=0.1):
    """
    Prepare filter functions for dataset spliting into test, train and validation sets
    :return: 3 filter functions: train, test, validation with 1 string argument
    """

    num_buckets = 100000
    b0 = 0
    b1 = int(train_frac * num_buckets)
    b2 = b1 + int(test_frac * num_buckets)
    b3 = num_buckets

    def filter(string, low, high):
        bucket_id = tf.string_to_hash_bucket_fast(string, num_buckets)
        return tf.logical_and(tf.greater_equal(bucket_id, low), tf.less(bucket_id, high))

    def filter_train(string): return filter(string, b0, b1)

    def filter_test(string): return filter(string, b1, b2)

    def filter_validation(string): return filter(string, b2, b3)

    return filter_train, filter_test, filter_validation

def get_conditional_files(path, dataset_folder):
    file_names = os.listdir(path + '/' + dataset_folder)
    file_names = [file for file in file_names if '.png' in file]
    file_paths = [path + '/' + dataset_folder + '/' + f for f in file_names]
    # labels = [int(x[(x.index('obj') + 3):x.index('_')]) for x in file_names]
    # file_paths = tf.convert_to_tensor(file_paths)
    return file_paths

def clear_filenames(image_decoded, label, _):
    return image_decoded, label

def _data_source(path,folrgb,follabel, width, height, parallel_calls, augment):

    def read_image(path, width, height, channels):
        image_string = tf.read_file(path)
        image_decoded = tf.image.decode_png(image_string, channels)
        image_decoded.set_shape([None, None, None])
        image_decoded = tf.image.resize_images(image_decoded, (height, width), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return image_decoded

    def parse_function(data_path, label_path):
        with tf.device('/cpu:0'):
            data = read_image(data_path, width, height, 3)
            label = read_image(label_path, width, height,3)
            label=tf.reshape(label[:,:,0],[256,512,1])

            def augment_func(img):
                img = tf.image.random_brightness(img, max_delta=16. / 255.)
                img = tf.image.random_contrast(img, lower=0.75, upper=1.25)
                img = tf.image.random_saturation(img, lower=0.75, upper=1.25)
                img = tf.image.random_hue(img, max_delta=0.1)
                return img

            if augment:
                data = augment_func(data)
                data, label = tf.cond(tf.greater(tf.random_uniform([], 0, 1), 0.5),
                                      lambda: (data, label),
                                      lambda: (tf.image.flip_left_right(data), tf.image.flip_left_right(label)))
            data = tf.image.convert_image_dtype(data, tf.float32)
            data = tf.transpose(data, [2, 0, 1])
            label = tf.transpose(label, [2, 0, 1])
            return data, label, data_path


    data_files = get_conditional_files(path, folrgb)
    target_files = get_conditional_files(path, follabel)
    assert len(data_files) == len(target_files)

    return (tf.data.Dataset.from_tensor_slices((data_files, target_files))
            .shuffle(8000)
            .map(parse_function, num_parallel_calls=parallel_calls))




def carlaimgs(path,folrgb, follabel, width=2048, height=1024, parallel_calls=8, augment=True):

    ds = _data_source(path, folrgb, follabel, width, height, parallel_calls, augment)

    ftr, fte, fval = dataset_filters(0.8, 0.10)

    train = (ds.filter(lambda img, label, path: ftr(path))
             .map(clear_filenames))

    test = (ds.filter(lambda img, label, path: fte(path))
            .map(clear_filenames))

    validation = (ds.filter(lambda img, label, path: fval(path))
                  .map(clear_filenames))

    return train, test, validation
