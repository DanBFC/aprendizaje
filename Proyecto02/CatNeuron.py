# importamos la libreria
import tensorflow as tf

# importar el programa que regresa los arreglos de las imagenes
from image_loader import images_to_tensor

# importamos librerias adicionales
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import pandas as pd

logs_path = "/tmp/tensorflow_logs/perceptron"
# Parametros
learning_rate = 0.001
epocas = 5
display_step = 100
num_steps = 500
dropout = 0.75
n_clases = 2 # Total de clases a clasificar (1 o 0)
lote = 20
# Parametros de la red
#n_oculta_1 = 256 # 1ra capa de atributos
#n_oculta_2 = 256 # 2ra capa de atributos
#n_entradas = 640000 # datos de MNIST(forma img: 28*28)


# Decimos el modo en el que está descrito el dataset y su directorio y archivo
path = "/home/tredok/Documents/aprendizaje/Proyecto02/images.txt"
# path = r"C:\Users\Tredok Vayntrub\Documents\GitHub\aprendizaje\Proyecto02\images.txt"
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
logits_train = conv_net(X, n_clases, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights
logits_test = conv_net(X, n_clases, dropout, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train,
                                                                        labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

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
            _, loss, acc = sess.run([train_op, loss_op, accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
        else:
            # Only run the optimization op (backprop)
            sess.run(train_op)

    print("Optimization Finished!")

    # Save your model
    saver.save(sess, 'my_tf_model')

    # # Función de activación de la capa escondida
    # capa_1 = tf.add(tf.matmul(x, pesos['h1']), sesgo['b1'])
    # # activacion relu
    # capa_1 = tf.nn.relu(capa_1)
    # # Función de activación de la capa escondida
    # capa_2 = tf.add(tf.matmul(capa_1, pesos['h2']), sesgo['b2'])
    # # activación relu
    # capa_2 = tf.nn.relu(capa_2)
    # # Salida con activación lineal
    # salida = tf.matmul(capa_2, pesos['out']) + sesgo['out']
    # return salida

# # Definimos los pesos y sesgo de cada capa.
# pesos = {
#     'h1': tf.Variable(tf.random_normal([n_entradas, n_oculta_1])),
#     'h2': tf.Variable(tf.random_normal([n_oculta_1, n_oculta_2])),
#     'out': tf.Variable(tf.random_normal([n_oculta_2, n_clases]))
# }
# sesgo = {
#     'b1': tf.Variable(tf.random_normal([n_oculta_1])),
#     'b2': tf.Variable(tf.random_normal([n_oculta_2])),
#     'out': tf.Variable(tf.random_normal([n_clases]))
# }

# with tf.name_scope('Modelo'):
#     # Construimos el modelo
#     pred = perceptron_multicapa(x, pesos, sesgo)

# with tf.name_scope('Costo'):
#     # Definimos la funcion de costo
#     costo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))

# with tf.name_scope('optimizador'):
#     # Algoritmo de optimización
#     optimizar = tf.train.AdamOptimizer(
#         learning_rate=tasa_aprendizaje).minimize(costo)

# with tf.name_scope('Precision'):
#     # Evaluar el modelo
#     pred_correcta = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#     # Calcular la precisión
#     Precision = tf.reduce_mean(tf.cast(pred_correcta, "float"))

# # Inicializamos todas las variables
# init = tf.global_variables_initializer()

# # Crear sumarización para controlar el costo
# tf.summary.scalar("Costo", costo)

# # Juntar los resumenes en una sola operación
# merged_summary_op = tf.summary.merge_all()


# # Lanzamos la sesión
# with tf.Session() as sess:
#     sess.run(init)

#     # op to write logs to Tensorboard
#     summary_writer = tf.summary.FileWriter(
#         logs_path, graph=tf.get_default_graph())

#     # Entrenamiento
#     for epoca in range(epocas):
#         avg_cost = 0.
#         # lote_total = int(mnist.train.num_examples/lote)
#         lote_total = batch_size

#         for i in range(lote_total):
#             # lote_x, lote_y = mnist.train.next_batch(lote)
#             lote_x = dataSet
#             lote_y = labels
#             # Optimización por backprop y funcion de costo
#             _, c, summary = sess.run([optimizar, costo, merged_summary_op],
#                                      feed_dict={x: lote_x, y: lote_y})
#             # escribir logs en cada iteracion
#             summary_writer.add_summary(summary, epoca * lote_total + i)
#             # perdida promedio
#             avg_cost += c / lote_total
#         # imprimir información de entrenamiento
#         if epoca % display_step == 0:
#             print("Iteración: {0: 04d} costo = {1:.9f}".format(epoca+1,
#                                                             avg_cost))
#     print("Optimización Terminada!\n")

#     #calcula precision con las imagenes de test
#     # print("Precisión: {0:.2f}".format(Precision.eval({x: mnist.test.images, y: mnist.test.labels})))
#     print("Precisión: {0:.2f}".format(Precision.eval({x: dataSet, y: labels})))

# print('matriz de test')
# print (mnist.test.images[0])
# print('vector de test indica que es 7')
# print (mnist.test.labels[0])
