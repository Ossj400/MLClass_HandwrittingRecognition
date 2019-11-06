from numpy.random import seed
from tensorflow import set_random_seed
set_random_seed(404)
import  os
import  numpy as np
import tensorflow as tf
import keras
from keras.datasets import cifar10
from IPython.display import  display
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
from    keras.models import  Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import TensorBoard
from time import strftime
from sklearn.metrics import confusion_matrix

# CONSTANTS
LABEL_Names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
IMG_Width = 32
IMG_Height = 32
IMG_Pix = IMG_Height*IMG_Width
COLOR_Channels = 3 #RGB
TOTAL_Inputs =IMG_Pix * COLOR_Channels
VALIDATION_Size = 10000
LOG_Dir = 'tensorboard_cifar_logs/' #there are folder with logs from learning

(x_train_all, y_train_all),(x_test, y_test) = cifar10.load_data()
pic = array_to_img(x_train_all[7])

#pic.show()
#print(y_train_all.shape)
#print(y_train_all[7][0])
#print(LABEL_NAMES[y_train_all[7][0]])
#plt.imshow(x_train_all[4])
#plt.xlabel(LABEL_NAMES[y_train_all[4][0]])
#plt.show()

#showing images
#for i in range (10):
    #print(LABEL_NAMES[y_train_all[i][0]])
    #plt.imshow(x_train_all[i])
    #plt.xlabel(LABEL_NAMES[y_train_all[i][0]])
    #plt.show()

#what contains train tuple
#nr_of_image, x, y, c = x_train_all.shape
#print(f' images= {nr_of_image} \t |width= {x} \t| height = {y} \t| channels = {c}')
# what is in test
#print(x_test.shape)

#     #PREPROCESS DATA
#divide rgb values by 255 to get smaller values (from 0-1)
x_train_all, x_test = x_train_all/255 , x_test/255
display(x_train_all[0][0][0][0])
#changing 4 dimensions to 1 row
x_train_all = x_train_all.reshape(x_train_all.shape[0], TOTAL_Inputs)
print( f'shape of x_train is {x_train_all.shape}')
x_test = x_test.reshape(x_test.shape[0], TOTAL_Inputs)
print(f'shape of x_test is {x_test.shape}')
#validation dataset
x_val = x_train_all[:VALIDATION_Size]
y_val = y_train_all[:VALIDATION_Size]
print(f'shape of x_val is {x_val.shape}')
x_train = x_train_all[10000:]   # or [VALIDATION_Size:]
y_train = y_train_all[10000:]
print('x train shape', x_train.shape)
print('y train shape', y_train.shape)
#Small dataset for illustration
x_train_small = x_train[:1000]
y_train_small = y_train[:1000]

#   DEFINE THE NEURAL NETWORK WITH KERAS
#  hidden layers = 3 and output
model_1 = Sequential([
    Dense(units=128, input_dim=TOTAL_Inputs, activation='relu', name='hidden_1'),
    Dense(units=64, activation='relu', name='hidden_2'),    #no need to input dimensions, keras can get it anyway
    Dense(16, activation='relu', name='hidden_3'),  #16 = units(neurons), its more often in docs. so good practice
    Dense(10,activation='softmax', name='output') #output
    ])

model_1.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
display(model_1.summary())

model_2 = Sequential()   # adding dropout - its randomly taking out a neuron
model_2.add(Dropout(0.2, seed=42, input_shape=(TOTAL_Inputs,)))
model_2.add(Dense(128, activation='relu', name='Model_2_hidden_1'))
model_2.add(Dense(64, activation='relu', name='Model_2_hidden_2'))
model_2.add(Dense(16, activation='relu', name='Model_2_hidden_3'))
model_2.add(Dense(10, activation='softmax', name='Model_2_output'))
model_2.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model_3 = Sequential()
model_3.add(Dropout(0.2, seed=42, input_shape=(TOTAL_Inputs,)))
model_3.add(Dense(128, activation='relu', name='Model_3_hidden_1'))
model_3.add(Dropout(0.25, seed=42))
model_3.add(Dense(64, activation='relu', name='Model_3_hidden_2'))
model_3.add(Dense(16, activation='relu', name='Model_3_hidden_3'))
model_3.add(Dense(10, activation='softmax', name='Model_3_output'))
model_3.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


#~~ !!!!!!!!!!!!! TENSORBOARD  (VISUALISING LEARNING)  !!!!!!!!!!!~~#
def get_tensorboard(model_name):
    #creating folder variable called: model_1 at (time)
    folder_name = f'{model_name} at {strftime("%H %M")}'
    # creating directory variable (need to use operation system)
    dir_paths = os.path.join(LOG_Dir, folder_name)
    # creating folder from paths, actually folder in folder :)
    #os.makedirs(dir_paths)
    # making try-catch if minutes and hours will be same, error will appear
    try:
        os.makedirs(dir_paths)
    except OSError as error:
        print(error.strerror)
    else:
        print('Successfully created directory')
    # Putin logs from TensorBoard
    return TensorBoard(log_dir= dir_paths)


# Fit the model   ~~ ~~ ~
samples_per_batch = 1000
nr_epochs = 150
model_1.fit(x_train_small, y_train_small,batch_size=samples_per_batch,epochs=nr_epochs,
            callbacks=[get_tensorboard('Model 1')], verbose=0, validation_data=(x_val,y_val))

# Fit model with dropout
samples_per_batch = 1000
nr_epochs = 150
model_2.fit(x_train_small, y_train_small,batch_size=samples_per_batch,epochs=nr_epochs,
            callbacks=[get_tensorboard('Model 2')], verbose=0, validation_data=(x_val,y_val))

samples_per_batch = 1000
nr_epochs = 150
model_3.fit(x_train_small, y_train_small,batch_size=samples_per_batch,epochs=nr_epochs,
            callbacks=[get_tensorboard('Model 3')], verbose=0, validation_data=(x_val,y_val))

# Fit models with MORE DATA!!!
samples_per_batch = 40000
nr_epochs = 100
model_1.fit(x_train_small, y_train_small,batch_size=samples_per_batch,epochs=nr_epochs,
            callbacks=[get_tensorboard('Model 1 XL')], verbose=0, validation_data=(x_val,y_val))

samples_per_batch = 40000
nr_epochs = 100
model_2.fit(x_train_small, y_train_small,batch_size=samples_per_batch,epochs=nr_epochs,
            callbacks=[get_tensorboard('Model 2 XL')], verbose=0, validation_data=(x_val,y_val))

samples_per_batch = 40000
nr_epochs = 100
model_3.fit(x_train_small, y_train_small,batch_size=samples_per_batch,epochs=nr_epochs,
            callbacks=[get_tensorboard('Model 3 XL')], verbose=0, validation_data=(x_val,y_val))


#Prediction on Individual IMAGES
display(x_val[0].shape) # only 1 dim. here
test = np.expand_dims(x_val[0], axis=0)   #expanding x_val dimension to 2 dimensions beacuse prediction in this case needs more than 1 dim.
display(test.shape) # 2 dims.
model_2.predict(test)  # nice but we want look it better, less decimals..
np.set_printoptions(precision=3)  #makes precision for decimals to 3.
model_2.predict(x_val)
model_2.predict_classes(test)

for iks in range(10):
    test_img = np.expand_dims(x_val[iks], axis=0)
    predicted_val = model_2.predict_classes(test_img)[0]
    print(f'actual value = {y_val[iks][0]} vs predicted = {predicted_val}')


# EVALUATION
test_loss, test_accuracy = model_2.evaluate(x_test,y_test)
print(f'Test loss is {test_loss:0.3} and test accuracy is {test_accuracy:0.1%}') # show loss with 3 number precision and test iwth percentage value and 1  number after dot.

#Confusion matrix
predictions = model_2.predict_classes(test)
conf_matrix = confusion_matrix(y_true= y_test, y_pred= predictions) #now conf_matrix have shape of (10,10) rows/columns
nr_rows = conf_matrix.shape[0]
nr_cols = conf_matrix.shape[1]

#plot
plt.figure(figsize=(7,7))
plt.imshow(conf_matrix, cmap=plt.cm.Greens)
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual Labels')
plt.xlabel('Prediction Labels')

tick_marks = np.arange(10)
plt.yticks(tick_marks, LABEL_Names)
plt.xticks(tick_marks, LABEL_Names)

plt.colorbar()  # color bar next to plot

plt.show()