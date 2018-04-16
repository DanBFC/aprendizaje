# importamos la libreria
import tensorflow as tf

# importamos librerías adicionales
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import pandas as pd

# importando el dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("aqui está la caca")
print(type(mnist))


# forma del dataset 55000 imagenes
mnist.train.images.shape

# cada imagen es un array de 28x28 con cada pixel
# definido como escala de grises.
digito1 = mnist.train.images[0].reshape((28, 28))
nnn = type(digito1)
print(nnn)
# visualizando el primer digito
#plt.imshow(digito1, cmap = cm.Greys)
#plt.show()

# Parametros
tasa_aprendizaje = 0.001
epocas = 15
lote = 100
display_step = 1
logs_path = "/tmp/tensorflow_logs/perceptron"

# Parametros de la red
n_oculta_1 = 256 # 1ra capa de atributos
n_oculta_2 = 256 # 2ra capa de atributos
n_entradas = 784 # datos de MNIST(forma img: 28*28)
n_clases = 10 # Total de clases a clasificar (0-9 digitos)

# input para los grafos
x = tf.placeholder("float", [None, n_entradas],  name='DatosEntrada')
y = tf.placeholder("float", [None, n_clases], name='Clases')

# Creamos el modelo
def perceptron_multicapa(x, pesos, sesgo):
    # Función de activación de la capa escondida
    capa_1 = tf.add(tf.matmul(x, pesos['h1']), sesgo['b1'])
    # activacion relu
    capa_1 = tf.nn.relu(capa_1)
    # Función de activación de la capa escondida
    capa_2 = tf.add(tf.matmul(capa_1, pesos['h2']), sesgo['b2'])
    # activación relu
    capa_2 = tf.nn.relu(capa_2)
    # Salida con activación lineal
    salida = tf.matmul(capa_2, pesos['out']) + sesgo['out']
    return salida

# Definimos los pesos y sesgo de cada capa.
pesos = {
    'h1': tf.Variable(tf.random_normal([n_entradas, n_oculta_1])),
    'h2': tf.Variable(tf.random_normal([n_oculta_1, n_oculta_2])),
    'out': tf.Variable(tf.random_normal([n_oculta_2, n_clases]))
}
sesgo = {
    'b1': tf.Variable(tf.random_normal([n_oculta_1])),
    'b2': tf.Variable(tf.random_normal([n_oculta_2])),
    'out': tf.Variable(tf.random_normal([n_clases]))
}

with tf.name_scope('Modelo'):
    # Construimos el modelo
    pred = perceptron_multicapa(x, pesos, sesgo)

with tf.name_scope('Costo'):
    # Definimos la funcion de costo
    costo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))

with tf.name_scope('optimizador'):
    # Algoritmo de optimización
    optimizar = tf.train.AdamOptimizer(
        learning_rate=tasa_aprendizaje).minimize(costo)

with tf.name_scope('Precision'):
    # Evaluar el modelo
    pred_correcta = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calcular la precisión
    Precision = tf.reduce_mean(tf.cast(pred_correcta, "float"))

# Inicializamos todas las variables
init = tf.global_variables_initializer()

# Crear sumarización para controlar el costo
tf.summary.scalar("Costo", costo)

# Juntar los resumenes en una sola operación
merged_summary_op = tf.summary.merge_all()


# Lanzamos la sesión
with tf.Session() as sess:
    sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(
        logs_path, graph=tf.get_default_graph())

    # Entrenamiento
    for epoca in range(epocas):
        avg_cost = 0.
        lote_total = int(mnist.train.num_examples/lote)

        for i in range(lote_total):
            lote_x, lote_y = mnist.train.next_batch(lote)
            # Optimización por backprop y funcion de costo
            _, c, summary = sess.run([optimizar, costo, merged_summary_op],
                                     feed_dict={x: lote_x, y: lote_y})
            # escribir logs en cada iteracion
            summary_writer.add_summary(summary, epoca * lote_total + i)
            # perdida promedio
            avg_cost += c / lote_total
        # imprimir información de entrenamiento
        if epoca % display_step == 0:
            print("Iteración: {0: 04d} costo = {1:.9f}".format(epoca + 1, avg_cost))
    print("Optimización Terminada!\n")

    #calcula precision con las imagenes de test
    print("Precisión: {0:.2f}".format(Precision.eval({x: mnist.test.images,
                                                y: mnist.test.labels})))

print('matriz de test')
print (mnist.test.images[0])
print('vector de test indica que es 7')
print (mnist.test.labels[0])
