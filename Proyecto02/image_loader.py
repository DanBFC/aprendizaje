import os, sys
import tensorflow as tf
import numpy as np


# This just create a file to indicate the labels and path + name of the images.
def image_file_creator():
    # linux paths
#    curiePath = "/home/tredok/Documents/aprendizaje/Proyecto02/Curie/"
#    nonCuriePath = "/home/tredok/Documents/aprendizaje/Proyecto02/nonCurie/"

    # Windows paths
    curiePath = r"D:\\Documents\aprendizaje\Proyecto02\curieImages\Curie\\"
    nonCuriePath = r"D:\\Documents\aprendizaje\Proyecto02\nonCurie\\"

    curieImages = os.listdir(curiePath)
    nonCurieImages = os.listdir(nonCuriePath)

    with open('images.txt', 'w') as f:
        for img in curieImages:
            f.write(curiePath + img + " " + str(1) + "\n")
        for img in nonCurieImages:
            f.write(nonCuriePath + img + " " + str(0) + "\n")


# This reads the images and transforms them into TFRecord
def images_to_tensor(dataset_path, mode, batch_size):

    CHANNELS = 3
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    N_CLASSES = 2
    print('entro')

    imagepaths, labels = list(), list()
    if mode == 'file':
        data = open(dataset_path, 'r').read().splitlines()
        for d in data:
            imagepaths.append(d.split(' ')[0])
            labels.append(int(d.split(' ')[1]))
    elif mode == 'folder':
        label = 0
        try:
            classes = sorted(os.walk(dataset_path).next()[1])
        except Exception:
            classes = sorted(os.walk(dataset_path).__next__()[1])
        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            try:
                walk = os.walk(c_dir).next()
            except Exception:
                walk = os.walk(c_dir).__next__()
            for sample in walk[2]:
                if sample.endswith('.jpg') or sample.endswith('.jpeg'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    lables.append(label)
            label += 1
    else:
        raise Exception("Unknown mode.")

    #  Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype = tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    image, label = tf.train.slice_input_producer([imagepaths, labels], shuffle = True)

    # Read images from disk
    image =  tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels = CHANNELS)

    # Resize images to a common size
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
    image = image * 1.0/127.5 - 1.0

    # Create batches
    X, Y = tf.train.batch([image, label],
                          batch_size=batch_size,
                          capacity=batch_size * 8)

    return X, Y

# This reads the images and transforms them into TFRecord
def image_tensor_test(dataset_path, mode):

    CHANNELS = 3
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    N_CLASSES = 2
    print('entro')

    imagepaths, labels = list(), list()
    if mode == 'file':
        data = open(dataset_path, 'r').read().splitlines()
        for d in data:
            imagepaths.append(d.split(' ')[0])
            labels.append(int(d.split(' ')[1]))
    elif mode == 'folder':
        label = 0
        try:
            classes = sorted(os.walk(dataset_path).next()[1])
        except Exception:
            classes = sorted(os.walk(dataset_path).__next__()[1])
        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            try:
                walk = os.walk(c_dir).next()
            except Exception:
                walk = os.walk(c_dir).__next__()
            for sample in walk[2]:
                if sample.endswith('.jpg') or sample.endswith('.jpeg'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    lables.append(label)
            label += 1
    else:
        raise Exception("Unknown mode.")

    #  Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype = tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    image, label = tf.train.slice_input_producer([imagepaths, labels], shuffle = True)

    # Read images from disk
    image =  tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels = CHANNELS)

    # Resize images to a common size
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
    image = image * 1.0/127.5 - 1.0

    # Create batches
    X, Y = tf.train.batch([image, label], 1)

    return X, Y


#if __name__ == "__main__":
#     path_to_curie = "/home/tredok/Documents/aprendizaje/Proyecto02/Curie/"
#     path_to_nonCurie = "/home/tredok/Documents/aprendizaje/Proyecto02/nonCurie/"

#     path = "/home/tredok/Documents/aprendizaje/Proyecto02/images.txt"
#     MODE = 'file'
#     batch_size = 4
#     x, y = images_to_tensor(path, MODE, batch_size)
#     print "here is x"
#     print x

#     print "here is y"
#     print y
#    image_file_creator()
