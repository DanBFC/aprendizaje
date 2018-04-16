# importamos la libreria
import tensorflow as tf

# importar el programa que regresa los arreglos de las imagenes
from image_loader import images_to_tensor

# importamos librerias adicionales
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import pandas as pd

#logs_path = "/tmp/tensorflow_logs/perceptron"
logs_path = r"D:\Documents\aprendizaje\Proyecto02\perceptron"
# Parametros
learning_rate = 0.001
epocas = 5
display_step = 100
num_steps = 500
dropout = 0.75
n_clases = 2 # Total de clases a clasificar (1 o 0)
n_entradas = 22500
lote = 20
# Parametros de la red
#n_oculta_1 = 256 # 1ra capa de atributos
#n_oculta_2 = 256 # 2ra capa de atributos
#n_entradas = 640000 # datos de MNIST(forma img: 28*28)


# Decimos el modo en el que está descrito el dataset y su directorio y archivo
# path = "/home/tredok/Documents/aprendizaje/Proyecto02/images.txt"
path = r"D:\Documents\aprendizaje\Proyecto02\images.txt"
mode = 'file'

# Obtenemos el dataset
X, Y = images_to_tensor(path, mode, lote)

# input para los grafos
#x = tf.placeholder("float", [None, n_entradas],  name='DatosEntrada')
#y = tf.placeholder("float", [None, n_clases], name='Clases')
# x = conv_net(X, n_clases, dropout, reuse = False, is_Training = True)
# y = conv_net(X, n_clases, dropout, reuse = True, is_Training = False)

# Creamos el modelo
def conv_net(x, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse = reuse):
        print("Convoluting and pooling")
        # Convolution layer with 32 filters and a kernel size of 400
        conv1 = tf.layers.conv2d(x, 32, 5, activation = tf.nn.relu)
        # Pooling layer with strides of 5 and a kernel size of 40
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution layer 2 with 32 filters and a kernel sizze of 200
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation = tf.nn.relu)
        # Pooling layer with strides of 5 and a kernel of 20
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)
        # Fully connected layer (in contrib folder for now)
        print("dense layer")
        fc1 = tf.layers.dense(fc1, 512)
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
    # pred_correcta = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y, tf.int64))
    # Evaluar el modelo
    pred_correcta = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calcular la presicion
    accuracy = tf.reduce_mean(tf.cast(pred_correcta, "float"))

# Se inicializan las variables
init = tf.global_variables_initializer()
# Crear la sumarizacin para controlar el Costo
tf.summary.scalar("Costo", costo)
# Juntar los resumenes en una sola operacion
merged_summary_op = tf.summary.merge_all()
# Saver object
saver = tf.train.Saver()

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

    # Save your model
    saver.save(sess, 'my_tf_model')

#    summary_writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())

    # The placeholders
#    x = tf.placeholder("float", [20, n_entradas, 3],  name='DatosEntrada')
#    y = tf.placeholder("float", [20, n_clases, 3], name='Clases')

    # Entrenamiento
#    for epoca in range(epocas):
#        avg_cost = 0
#        lote_total = lote

#        for i in range(lote_total):
#            X, Y = images_to_tensor(path, mode, lote)
            # Optimizacion por backprop y funcion de costo
#            _, c, summary = sess.run([optimizador, costo, merged_summary_op], feed_dict = {x: X, y: Y})
            # Escribimos la iteracion en los registros
#            summary_writer.add_summary(summary, epoca * lote_total + i)
#            avg_cost += c / lote_total
#        if epoca % display_step == 0:
#            print("Iteración: {0: 04d} costo = {1:.9f}".format(epoca + 1, avg_cost))
#    print("Optimización Terminada!\n")

    # Save your model
#    saver.save(sess, 'my_tf_model')
