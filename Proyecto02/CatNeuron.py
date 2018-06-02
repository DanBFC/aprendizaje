# importamos la libreria
import tensorflow as tf

# importar el programa que regresa los arreglos de las imagenes
from image_loader import images_to_tensor, image_tensor_test

# importamos librerias adicionales
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import pandas as pd

#logs_path = "/tmp/tensorflow_logs/perceptron"
logs_path = "/perceptron"
# Parametros
learning_rate = 0.01
epocas = 20
display_step = 1
num_steps = 1
dropout = 0.75
n_clases = 2 # Total de clases a clasificar (1 o 0)
n_entradas = 22500
lote = 100



# Decimos el modo en el que est descrito el dataset y su directorio y archivo
# Test path linux
#saver_path = "/home/tredok/Documents/aprendizaje/Proyecto02/Log/KittyModel"
#path = "/home/tredok/Documents/aprendizaje/Proyecto02/images.txt"

# Test path windows
saver_path = r"D:\\Documents\GitHub\aprendizaje\\Proyecto02\\Log\\kittyModel"
path = r"D:\Documents\GitHub\aprendizaje\Proyecto02\filesImages\images.txt"

# Test paths
# test path linux
#pathtest = "/home/tredok/Documents/aprendizaje/Proyecto02/filesImages/images_test.txt"

# test path windows
pathtest = r"D:\Documents\GitHub\aprendizaje\Proyecto02\filesImages\images_test.txt"
mode = 'file'

# Obtenemos el dataset
X, Y = images_to_tensor(path, mode, lote)
test_image, test_label = image_tensor_test(pathtest, mode)

# Creamos el modelo
def conv_net(x, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse = reuse):
        print("Convoluting and pooling")
        # Convolution layer with 32 filters and a kernel size of 400
        conv1 = tf.layers.conv2d(x, 32, 5, activation = tf.nn.relu)
        # Pooling layer with strides of 5 and a kernel size of 40
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution layer 2 with 32 filters and a kernel sizze of 200
        conv2 = tf.layers.conv2d(conv1, 32, 3, activation = tf.nn.relu)
        # Pooling layer with strides of 5 and a kernel of 20
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)
        # Fully connected layer (in contrib folder for now)
        print("dense layer")
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply dropout (if is_training is false, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate = dropout, training = is_training)

        print("prediction layer")
        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # We only aply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out


# Create a graph for training
with tf.name_scope('Modelo'):
    pred = conv_net(X, n_clases, dropout, reuse=False, is_training=True)
    # Create another graph for testing that reuse the same weights
    logits_test = conv_net(X, n_clases, dropout, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
with tf.name_scope('costo'):
    costo = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=Y))

# Algoritmo de optimimzacion
with tf.name_scope('optimizador'):
    optimizador = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizador.minimize(costo)

# Evaluate model (with test logits, for dropout to be disabled)
with tf.name_scope('Presicion'):
    pred_correcta = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y, tf.int64))
    # Evaluar el modelo
    # pred_correcta = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calcular la presicion
    accuracy = tf.reduce_mean(tf.cast(pred_correcta, "float"))

# Seinicializan las variables
init = tf.global_variables_initializer()


# Crear la sumarizacion para controlar el Costo
tf.summary.scalar("Costo", costo)
# Juntar los resumenes en una sola operacion
merged_summary_op = tf.summary.merge_all()
# Saver object
saver = tf.train.Saver()

# input para los grafos
x = tf.placeholder(tf.float32, shape = (1, 150, 150, 3))
y = tf.placeholder(tf.float32, shape = (1, 150, 150, 3))


# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    print("Starting session")

    # Start the data queue
    tf.train.start_queue_runners()

    # Training cycle
    for step in range(1, num_steps+1):

        if step % display_step == 0:
            # Run optimization and calculate batch loss and accuracy
            _, loss, acc = sess.run([pred, costo, accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
        else:
            # Only run the optimization op (backprop)
            sess.run(train_op)
    print("Optimization Finished!")

    # Calcular
    print("Running a test")
    print("Presicion: {0: 2f}".format(accuracy.eval({x: test_image})))

    # Save your model
    #saved_path = saver.save(sess, saver_path)
    #print("kitty model saved at: %s" % saved_path)
