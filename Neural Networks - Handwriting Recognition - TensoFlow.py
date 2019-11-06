from numpy.random import  seed
seed(888)
from tensorflow import  set_random_seed
set_random_seed(404)
import  os
import  numpy as np
import tensorflow as tf
from time import  strftime
import PIL
from PIL import Image

###  anaconda prompt :  >tensorboard --logdir=C:\Users\Artur\PycharmProjects\MLClass\Handwriting_Recognition\tensorboard_mnist_digit_logs
# or tensorboard --logdir=C:\Users\Artur\PycharmProjects\MLClass\Handwriting_Recognition\tensorboard_mnist_digit_logs --port 8080 --host 127.0.0.1

#Constants
X_Train_Path = 'MNIST/digit_xtrain.csv'
Y_Train_Path = 'MNIST/digit_ytrain.csv'
X_Test_Path = 'MNIST/digit_xtest.csv'
Y_Test_Path = 'MNIST/digit_ytest.csv'

LOGGING_PATH = 'tensorboard_mnist_digit_logs/'

Validation_Size = 10000
IMG_WIDTH = 28
IMG_HEIGHT = 28
CHANNELS = 1
Total_Inputs = IMG_HEIGHT*IMG_WIDTH*CHANNELS



# Getting Data
y_train_all = np.loadtxt(Y_Train_Path, delimiter=',',dtype= int)
y_test_all = np.loadtxt(Y_Test_Path, delimiter=',',dtype= int)
x_train_all = np.loadtxt(X_Train_Path, delimiter=',',dtype= int)
x_test_all = np.loadtxt(X_Test_Path, delimiter=',',dtype= int)
#Explore data
# print(x_test_all.shape)
# print(x_train_all.shape)   #  60 k examples of 28 x 28 px x `1 color channel
# print(y_train_all.shape)   # 60k labels
# print(y_test_all.shape)
# print(x_train_all[0])

        #Data preprocessing

# Rescale features from 0-1, now is 0-255.
x_train_all, x_test_all = x_train_all/ 255.0, x_test_all/255.0

#convert target values to  one-hot encoding
# eye - showing values in states 0 or 1, if there is eye of 4, and I have value= 3 then it will be " 0, 0 , 1 , 0 "
y_train_all = np.eye(10)[y_train_all]
y_test_all = np.eye(10)[y_test_all]

#Create validation data set
x_train = x_train_all[Validation_Size:]
x_val = x_train_all[:Validation_Size]

y_train = y_train_all[Validation_Size:]
y_val = y_train_all[:Validation_Size]
# print(x_train.shape)
# print(x_val.shape)

#  Setup TF Graph
# creating tensors
X = tf.placeholder(tf.float32, shape=[None, Total_Inputs], name='X')     #None for 1st dimension coz it will be later set, with batches
Y = tf.placeholder(tf.float32, shape= [None, 10], name='labels')

##      Neural Network Architecture
#hyperparameters - layers, nodes in layer, number of epochs itp
nr_epoch = 10
learning_rate = 0.01
n_hidden_1 = 512
n_hidden_2 = 64

#################### DONE BY METHOD
# with tf.name_scope('hidden1_layer'):
#
#     initial_w1 = tf.truncated_normal(shape=[Total_Inputs, n_hidden_1], stddev=0.1, seed= 42)
#     w1 = tf.Variable(initial_value=initial_w1, name='w1')
#     initia_bias1 = tf.constant(value=0.0, shape=[n_hidden_1])
#     bias1 = tf.Variable(initial_value=initia_bias1, name='b1')
#
#     layer1_input = tf.matmul(X, w1) + bias1
#     layer1_output = tf.nn.relu(layer1_input)
#
# # second  h layer
# with tf.name_scope('hidden2_layer'):
#
#     initial_w2 = tf.truncated_normal(shape=[n_hidden_1, n_hidden_2], stddev=0.1, seed= 42)
#     w2 = tf.Variable(initial_value=initial_w2, name='w2')
#     initia_bias2 = tf.constant(value=0.0, shape=[n_hidden_2])
#     bias2 = tf.Variable(initial_value=initia_bias2, name='b2')
#
#     layer2_input = tf.matmul(layer1_output, w2) + bias2
#     layer2_output=tf.nn.relu(layer2_input)
#
# # third  h layer
# with tf.name_scope('output_layer'):
#
#     initial_w3 = tf.truncated_normal(shape=[n_hidden_2, 10], stddev=0.1, seed= 42)
#     w3 = tf.Variable(initial_value=initial_w3, name='w3')
#     initia_bias3 = tf.constant(value=0.0, shape=[10])
#     bias3 = tf.Variable(initial_value=initia_bias3, name='b3')
#
#     layer3_input = tf.matmul(layer2_output, w3) + bias3
#     layer3_output=tf.nn.softmax(layer3_input)
#     output = layer3_output

####Function to seyup layers =======

def setup_layer(input, weight_dim, bias_dim,name):
    with tf.name_scope(name):
        initial_w = tf.truncated_normal(shape=weight_dim, stddev=0.1, seed=42)
        w = tf.Variable(initial_value=initial_w, name='W')
        initia_bias = tf.constant(value=0.0, shape=bias_dim)
        bias = tf.Variable(initial_value=initia_bias, name='B')

        layer_input = tf.matmul(input, w) + bias
        if name=='out':
            layer_out = tf.nn.log_softmax(layer_input)
        else:
            layer_out=tf.nn.relu(layer_input)

        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', bias)
        return layer_out

# Initializing layers
# layer_1= setup_layer(X,weight_dim=[Total_Inputs, n_hidden_1], bias_dim=[n_hidden_1],name='layer_1')
# layer_2= setup_layer(layer_1,weight_dim=[n_hidden_1, n_hidden_2], bias_dim=[n_hidden_2],name='layer_2')
# output= setup_layer(layer_2,weight_dim=[n_hidden_2, 10], bias_dim=[10],name='out')
# model_name = f'{n_hidden_1}-{n_hidden_2} LR{learning_rate} E{nr_epoch}'

# AND WITH DROPOUT layer
layer_1= setup_layer(X,weight_dim=[Total_Inputs, n_hidden_1], bias_dim=[n_hidden_1],name='layer_1')

layer_drop = tf.nn.dropout(layer_1, keep_prob=0.8, name='dropout_layer')

layer_2= setup_layer(layer_drop,weight_dim=[n_hidden_1, n_hidden_2], bias_dim=[n_hidden_2],name='layer_2')
output= setup_layer(layer_2,weight_dim=[n_hidden_2, 10], bias_dim=[10],name='out')
model_name = f'{n_hidden_1}-{n_hidden_2} LR{learning_rate} E{nr_epoch}'

#### TENSORBOARD SETUP
#folder for TBoard
folder_name = f'{model_name}_at_{strftime("%H.%M.%S")}'
print(folder_name)
directory = os.path.join(LOGGING_PATH, folder_name)

try:
    os.makedirs(directory)
except OSError as exception:
    print(exception.strerror)
else:
    print('Succesfully created folder')

## LOSS OPTIMALIZATION & METRICS
#defining loss function
with tf.name_scope('loss_calc'):
    loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=output))

#defining optimizer
with tf.name_scope('optimizer'):
    optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss)

#accuracy metrics
with tf.name_scope('accuracy_calc'):
    correct_prediction = tf.equal(tf.argmax(output, axis = 1),tf.argmax(Y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.name_scope('performance'):
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('cost',loss)

#Check input images in Tboard
with tf.name_scope('show_image'):
    x_image = tf.reshape(X, [-1,28,28,1])
    tf.summary.image('image_input',x_image,max_outputs=4)

#Running session
sess = tf.Session()
#Setup filewriter
merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(directory + '/train')
train_writer.add_graph(sess.graph)
validation_writer = tf.summary.FileWriter(directory + '/validation')
#initialize variables
init = tf.global_variables_initializer()
sess.run(init)
# w1.eval(sess)


#Batching Data

size_of_batch = 1000
num_of_examples = y_train.shape[0]
nr_iterations = int (num_of_examples/size_of_batch)
index_in_epoch = 0

def next_batch(batch_size, data, labels):
    global num_of_examples
    global index_in_epoch
    start = index_in_epoch
    index_in_epoch = index_in_epoch + batch_size

    if index_in_epoch> num_of_examples:
        start=0
        index_in_epoch = batch_size

    end = index_in_epoch

    return data[start:end], labels[start:end]

##training loop

for epochs in range(nr_epoch):

   # ~~~~~~~ === TRAINING DATA SET ============== ~~~~~~~~

    for i in range(nr_iterations):
            batch_x, batch_y = next_batch(batch_size=size_of_batch,data=x_train,labels= y_train)      ### maybe batch_size = size_of_batch....,
            feed_dictionary = {X:batch_x, Y:batch_y}
            sess.run(train_step, feed_dict = feed_dictionary)

    s, batch_accuracy = sess.run(fetches=[merged_summary,accuracy], feed_dict=feed_dictionary)
    train_writer.add_summary(s,epochs)
    print(f'Epoch {epochs} \t| Training Accuracy  = {batch_accuracy}')

   # ================ VALIDATION ============= #
summary = sess.run(fetches=merged_summary, feed_dict={X: x_val,Y: y_val})

print('Done!')

### MAKE A PREDICTION
img = Image.open('MNIST/test_img.png')
bw = img.convert('L')
img_array=np.invert(bw)
test_img = img_array.ravel()

prediction = sess.run(fetches=tf.argmax(output, axis=1),feed_dict={X: [test_img]})

### TESTIN AND EVALUATION
test_accuracy = sess.run(fetches=accuracy,feed_dict={X: x_test_all, Y: y_test_all})

#REset for next run
train_writer.close()
validation_writer.close()
sess.close()
tf.reset_default_graph()