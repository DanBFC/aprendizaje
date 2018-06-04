# importamos la libreria
import tensorflow as tf

# importar el programa que regresa los arreglos de las imagenes
from image_loader import images_to_tensor, image_tensor_test, get_dataset

# importamos librerias adicionales
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import pandas as pd

# Decimos el modo en el que est descrito el dataset y su directorio y archivo
# Test path linux
#saver_path = "/home/tredok/Documents/aprendizaje/Proyecto02/Log/KittyModel"
#path = "/home/tredok/Documents/aprendizaje/Proyecto02/images.txt"

# Test path windows
saver_path = r"D:\\Documents\GitHub\aprendizaje\Proyecto04\Log\kittyModel"
path = r"D:\Documents\GitHub\aprendizaje\Proyecto04\images.txt"

# Test paths
# test path linux
#pathtest = "/home/tredok/Documents/aprendizaje/Proyecto02/filesImages/images_test.txt"

# test path windows
pathtest = r"D:\Documents\GitHub\aprendizaje\Proyecto04\images_test.txt"
mode = 'file'

#logs_path = "/tmp/tensorflow_logs/perceptron"
logs_path = "/perceptron"
# Parametros
learning_rate = 0.01
epocas = 150
display_step = 1
num_steps = 5
dropout = 0.75
n_clases = 2 # Total de clases a clasificar (1 o 0)
lote = 50
num_examples = 731

# Parametros del perceptron multicapa
capa_oculta1 = 256
capa_oculta2 = 256
capa_oculta3 = 256
capa_oculta4 = 256
capa_oculta5 = 256
n_entradas = 22500
clases = 2

# Obtenemos el dataset
X, Y = images_to_tensor(path, mode, lote)
test_image, test_label = image_tensor_test(pathtest, mode)

# input para los grafos
# x = tf.placeholder(tf.float32, shape = (1, 150, 150, 3))
# y = tf.placeholder(tf.float32, shape = (1, 150, 150, 3))
x = tf.placeholder("float", [n_entradas, None], name = 'entradas')
y = tf.placeholder("float", [clases, None], name = 'Clases')

def multilayer_perceptron(x, peso, sesgo):
    capa_1 = tf.add(tf.matmul(x, pesos['h1']), sesgo['b1'])
    capa_1 = tf.nn.relu(capa_1)

    capa_2 = tf.add(tf.matmul(capa_1, pesos['h2']), sesgo['b2'])
    capa_2 = tf.nn.relu(capa_2)

    capa_3 = tf.add(tf.matmul(capa_2, pesos['h3']), sesgo['b3'])
    capa_3 = tf.nn.relu(capa_3)

    capa_4 = tf.add(tf.matmul(capa_3, pesos['h4']), sesgo['b4'])
    capa_4 = tf.nn.relu(capa_4)

    capa_5 = tf.add(tf.matmul(capa_4, pesos['h5']), sesgo['b5'])
    capa_5 = tf.nn.relu(capa_5)

    out_layer = tf.matmul(capa_5, pesos['out']) + sesgo['out']
    return out_layer

pesos = {
    'h1': tf.Variable(tf.random_normal([n_entradas, capa_oculta1])),
    'h2': tf.Variable(tf.random_normal([capa_oculta1, capa_oculta2])),
    'h3': tf.Variable(tf.random_normal([capa_oculta2, capa_oculta3])),
    'h4': tf.Variable(tf.random_normal([capa_oculta3, capa_oculta4])),
    'h5': tf.Variable(tf.random_normal([capa_oculta4, capa_oculta5])),
    'out': tf.Variable(tf.random_normal([capa_oculta5, clases]))
}
sesgo = {
    'b1': tf.Variable(tf.random_normal([capa_oculta1])),
    'b2': tf.Variable(tf.random_normal([capa_oculta2])),
    'b3': tf.Variable(tf.random_normal([capa_oculta3])),
    'b4': tf.Variable(tf.random_normal([capa_oculta4])),
    'b5': tf.Variable(tf.random_normal([capa_oculta5])),
    'out': tf.Variable(tf.random_normal([clases]))
}

# Create a graph for training
with tf.name_scope('Modelo'):
    pred = multilayer_perceptron(x, pesos, sesgo)
    #pred = conv_net(X, n_clases, dropout, reuse=False, is_training=True)
    # Create another graph for testing that reuse the same weights
    #logits_test = conv_net(X, n_clases, dropout, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
with tf.name_scope('costo'):
    costo = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=Y))

# Algoritmo de optimimzacion
with tf.name_scope('optimizador'):
    optimizador = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizador.minimize(costo)

# Evaluate model (with test logits, for dropout to be disabled)
with tf.name_scope('Presicion'):
    # Evaluar el modelo
    pred_correcta = tf.equal(tf.argmax(pred, 1), tf.cast(Y, tf.int64))
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

#x1 = tf.placeholder("float", [n_entradas, None], name = 'batch_3')
#y1 = tf.placeholder("float", [clases, None], name = 'Clases')

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    #sess.run([images, labels])

    print("Starting session")

    # Start the data queue
    tf.train.start_queue_runners()

    # Training cycle
    for epoca in range(epocas):
        avg_cost = 0
        lote_total = int(num_examples / lote)

        for i in range(lote_total):
            lote_x, lote_y = X, Y
            _, c, summary = sess.run([optimizador, costo, merged_summary_op],
                            feed_dict = {x: lote_x, y: lote_y})
            summary_writer.add_summary(summary, epoca * lote_total + i)
            avg_cost =+ c / lote_total
        if epoca % display_step == 0:
            print("Iteracion: {0: 04d} costo = {1:.9f}".format(epoca + 1, avg_cost))
    print("Optimization Finished!")

    # Calcular
    print("Running a test")
    print("Presicion: {0: 2f}".format(accuracy.eval({x: test_image, y: test_label})))

    # Save your model
    #saved_path = saver.save(sess, saver_path)
    #print("kitty model saved at: %s" % saved_path)
