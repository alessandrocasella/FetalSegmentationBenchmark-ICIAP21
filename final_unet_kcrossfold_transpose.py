# -*- coding: utf-8 -*-
"""NEW-UNET-KCROSSFOLD-UPSAMPLING.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AflRDYhqCOzJOtM8Mq5WoBj5FVcBO2S5

#Import Libraries
"""

import os
from os import listdir
from PIL import Image 
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K                                        #non viene mai usata???
from tensorflow.keras.layers import *                                            #need it for Input name in the UNET
from tensorflow.keras.models import Model, load_model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix 
from keras.models import load_model
from keras.utils import plot_model
from keras.utils import model_to_dot
#from IPython.display import SVG
import h5py



cross = [f.path for f in os.scandir(os.path.join(os.getcwd(), 'Try/NEW DATASET NUMPY 9paz')) if f.is_dir()]
print(cross)
cross = [f.path for f in os.scandir(cross[0]) if f.is_dir() ]
print(cross)
cross.sort()

directory = 'Graphs Transpose'
graph_path = os.path.join(os.path.join(os.getcwd(), 'Try/NEW DATASET NUMPY 9paz'), directory)
try:
    os.mkdir(graph_path)
except:
    pass

graph_path

mask_path = cross[1]
image_path = cross[0]
#predicted_path = cross[2]
print(mask_path)
print(image_path)
#print(predicted_path)


mask_subfolders = [f.path for f in os.scandir(mask_path) if f.is_dir() ]
mask_subfolders.sort()
print(mask_subfolders)

path_mask_list_new = []  #directories of the mask for every patient
for path in mask_subfolders:
    path_mask_list_new.append(path)
print(path_mask_list_new)

image_subfolders = [f.path for f in os.scandir(image_path) if f.is_dir() ]
image_subfolders.sort()
print(image_subfolders)

path_image_list_new = []    #directories of the image for every patient
for path in image_subfolders:
    path_image_list_new.append(path)
print(path_image_list_new)

n_patient = len(path_image_list_new)
print(n_patient)

predicted_subfolders_path = os.path.join(os.path.join(os.getcwd(), 'Try/NEW DATASET NUMPY 9paz/NEW DATASET NUMPY 9paz'), 'Prediction Transpose')
try:
    os.mkdir(predicted_subfolders_path)
except:
    pass
print(predicted_subfolders_path)

for k in range(0, n_patient):
    path = os.path.join(predicted_subfolders_path, 'Patient{0}'.format(k+1))
    try:
        os.mkdir(path)
    except:
        pass

predicted_subfolders = [f.path for f in os.scandir(predicted_subfolders_path) if f.is_dir() ]
predicted_subfolders.sort()
print(predicted_subfolders)

path_predicted_list_new = [] #directories of the prediction for every patient
for path in predicted_subfolders:
    path_predicted_list_new.append(path)
print(path_predicted_list_new)

n_patient = len(path_image_list_new)
print(n_patient)


"""#MY UNET 2D"""

class myUnet(object):
    def __init__(self, img_rows=256, img_cols=256):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def get_unet(self, training=True):
        inputs = Input((self.img_rows, self.img_cols, 3))

        c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = BatchNormalization()(c1)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        c1 = BatchNormalization()(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = BatchNormalization()(c2)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        c2 = BatchNormalization()(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = BatchNormalization()(c3)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        c3 = BatchNormalization()(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = BatchNormalization()(c4)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        c4 = BatchNormalization()(c4)
        p4 = MaxPooling2D((2, 2))(c4)

        c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = BatchNormalization()(c5)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        c5 = BatchNormalization()(c5)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = BatchNormalization()(c6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
        c6 = BatchNormalization()(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = BatchNormalization()(c7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
        c7 = BatchNormalization()(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = BatchNormalization()(c8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
        c8 = BatchNormalization()(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = BatchNormalization()(c9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        c9 = BatchNormalization()(c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        model = Model(inputs=inputs, outputs=outputs)

        return model

"""#MY DATA GENERATOR """

# Define ImageDataGenerator
#if you don't want to do data augmentation, set aug empty

aug = ImageDataGenerator(
    rotation_range=180,
    zoom_range=[0.5, 1.5],
    width_shift_range=-0.1,
    height_shift_range=-0.1,
    data_format='channels_last',
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
    )
#to obtain images distorted

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, list_np , list_label_np, batch_size, nChannelData, nChannelLabel, shuffle, aug):
        #print('__init__')
        self.batch_size = batch_size
        self.nChannelData = nChannelData
        self.nChannelLabel = nChannelLabel
        self.shuffle = shuffle
        self.list_np = list_np
        self.list_label_np = list_label_np
        #print('The lenght of the list np is {0}'.format(len(self.list_np)))
        self.aug = aug
        self.on_epoch_end()

    def __len__(self):
        #print('__len__')
        i = int(np.floor(len(self.list_np) / self.batch_size))
        #print('The lenght of list_np is {0}'.format(len(self.list_np)))
        #print('The division is {0}'.format(len(self.list_np) / self.batch_size))
        #print('The divison after the np.flor is {0}'.format(i))
        #print(i)
        return i

    def __getitem__(self, index):
        #print('__getitem__')
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        file_list_temp = [self.list_np[k] for k in indexes]
        #print('The lenght of the file temp is {0}'.format(len(file_list_temp)))
        X, y_new = self.__data_generation(file_list_temp)
        #print('The lenght of X is {0}'.format(len(X)))
        #print('The lenght of y_new is {0}'.format(len(y_new)))
        return X, y_new

    def on_epoch_end(self):
        #print('one epoch end')
        self.indexes = np.arange(len(self.list_np))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_list_temp):
        #print('data generation')
        y_new = np.empty([self.batch_size, 256, 256, 1], dtype=np.float32)
        x_new = np.empty([self.batch_size, 256, 256, 3], dtype=np.float32)
        max_z = 0
        max_y = 0

        for p, ID in enumerate(file_list_temp):

            transformInstance = self.aug.get_random_transform((256, 256, 3))
            X = np.array(ID/255.).astype('float32')
            
            Z = X

            y_old = self.list_label_np[p]
            y = np.array(y_old/np.max(y_old)).astype('float32') 
            y = np.asarray(np.dstack((y, y, y)), dtype=np.float32)

            Z[:, :, :] = self.aug.apply_transform(X[:, :, :], transformInstance)
            if (np.max(Z)!= 1.0):
                max_z = max_z + 1
                #print('The max in Z is not 1 but is {0}'.format(np.max(Z)))
            #print('The shape of Z is {0}'.format(Z.shape))
            x_new[p, :, :,:] = Z[:, :, :]
            #print('The shape of x_new is {0}'.format(x_new.shape))
            y[:, :, :] = self.aug.apply_transform(y[:, :, :], transformInstance)
            #print('The shape of y is {0}'.format(y.shape))
            if (np.max(y)!= 1.0):
                max_y = max_y + 1
                #print('The max in y is not 1 but {0}'.format(np.max(y)))
            y_new[p, :, :, 0] = y[:, :, 0]
            #print('The shape of y_new is {0}'.format(y_new.shape))
        if (max_z != 0):
          print('The max in Z is not 1 for {0} value'.format(max_z))
        if (max_y != 0):
          print('The max in y is not 1 for {0} value'.format(max_y))
        return x_new, y_new


"""## defining the metrics"""

#defining the metric Structural similarity index SSIM
def ssim(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

#defining the SSIM loss
def ssim_loss(y_true, y_pred):
  return 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


"""## Getting the Unet, visualizing it"""

########################################################
learning_rate = 0.001 #@param {type:"number"}
batchSize =  32 #@param {type:"number"}
#earlystop_patience = 50 #@param {type:"number"}  
#rule of thumb to make it 10% of number of epoch.

#GET THE UNET AND DISPLAY MY MODEL
MyModel = myUnet() # creo istanza dell'oggetto
myunet = MyModel.get_unet() # questa funzione, data l'istanza dell'oggetto, associa un modello a questa variabile
myunet.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss=ssim_loss, metrics=[ssim, 'acc', dice_coeff])
myunet.summary()

plot_model(myunet, to_file='myunet_transpose_plot.png', show_shapes=False, show_layer_names=False, rankdir='LR')

path = os.path.join(os.getcwd(),'myunet_transpose_plot.png')
im = Image.open(path)

im.save(graph_path+'/myunet_transpose_plot.png')

"""##Creating the Folds"""

size_test = 3
size_train = 4
size_val = 2

n_test = size_test
K_fold = int(n_patient / n_test)
print(K_fold)

from sklearn.model_selection import KFold
import random
kfold = KFold(K_fold)
print(kfold)


K_test = []
K_train = []
K_val = []
for train, test in kfold.split(path_image_list_new):
  print('    ')
  #print(train)
  print(test)
  np.random.seed(2)
  np.random.shuffle(train)
  #print(train)
  for train_2, val in kfold.split(train):
    new_train = []
    for x in train_2:
      new_train.append(train[x])
    new_val = []
    for x in val:
      new_val.append(train[x])
    new_train.sort()
    new_val.sort()
    print(new_train)
    print(new_val)
    break
  K_test.append(test)
  K_train.append(new_train)
  K_val.append(new_val)

print(K_test)

print(K_train)

print(K_val)

## TEST

#checking the correct selection of the patient directories
i = 0
for tr in K_test:
  i = i + 1
  print('FOLD {}'.format(i))
  print(tr)
  for tr_x in tr:
    path_image = path_image_list_new[tr_x]
    print(path_image)
    path_mask = path_mask_list_new[tr_x]
    print(path_mask)

#defining the testing set for each folds
K_fold_test_image = []
K_fold_test_mask = []
i = 0
for t in K_test:
  test_image_np = []
  test_mask_np = []
  patient_test_image = []
  patient_test_mask = []
  i = i + 1
  print('FOLD {}'.format(i))
  #print(t)
  for t_x in t:
    #print(t_x)
    path_image = path_image_list_new[t_x]
    p_i = os.listdir(path_image)
    patient_test_image.append(t_x+1)
    path_mask = path_mask_list_new[t_x]
    p_m = os.listdir(path_mask)
    patient_test_mask.append(t_x+1)
    for x,image in enumerate(p_i):
      if (x <45):
        test_image = np.load(path_image + '/' + image)   # the images are in uint8 and np.max = 255
        if ( test_image.shape != (256, 256, 3)):
          print('error')
        test_image_np.append(test_image)
        b = x + 1
    print('Test Image: Loading patient{0} and it contain {1} files'.format(t_x+1,b))
    for x,mask in enumerate(p_m):
      if (x < 45):
        test_mask = np.load(path_mask + '/' + mask)     # the masks are uint8 with different np.max and have shape (256, 256, 1)
        test_mask = np.asarray(test_mask/np.max(test_mask)).astype('float32') #binarizing the masks and transforming them into float32
        test_mask_np.append(test_mask)
        b = x + 1
    #print(test_mask_np[0].shape)
    #print(test_mask_np[0].dtype)
    #print(np.max(test_mask_np[0]))
    print('Test Mask: Loading patient{0} and it contain {1} files'.format(t_x+1,b))
  print('           The testing set image is lenght {} and it contains the patient: '.format(len(test_image_np)), patient_test_image)
  print('           The testing set mask is lenght {} and it contains the patient: '.format(len(test_mask_np)), patient_test_mask)
  K_fold_test_image.append(test_image_np)
  K_fold_test_mask.append(test_mask_np)
  print('   ')

#checking the creation of the folds
print(len(K_fold_test_image))
for x in K_fold_test_image:
  print(len(x))
print(len(K_fold_test_mask))
for x in K_fold_test_mask:
  print(len(x))

## TRAIN

#checking the correct selection of the patient directories
i = 0
for tr in K_train:
  i = i + 1
  print('FOLD {}'.format(i))
  print(tr)
  for tr_x in tr:
    path_image = path_image_list_new[tr_x]
    print(path_image)
    path_mask = path_mask_list_new[tr_x]
    print(path_mask)

#defining the training set for each folds
K_fold_train_image = []
K_fold_train_mask = []
i = 0

for tr in K_train:
  train_image_np = []
  train_mask_np = []

  patient_train_image = []
  patient_train_mask = []

  i = i + 1
  print('FOLD {}'.format(i))

  for tr_x in tr:
    path_image = path_image_list_new[tr_x]
    p_i = os.listdir(path_image)
    patient_train_image.append(tr_x+1)
    path_mask = path_mask_list_new[tr_x]
    p_m = os.listdir(path_mask)
    patient_train_mask.append(tr_x+1)
    for x, image in enumerate(p_i):
      if (x < 45):
        train_image = np.load(path_image + '/' + image)
        train_image_np.append(train_image)
        b = x + 1
    print('Train Image: Loading patient{0} and it contain {1} files'.format(tr_x+1,b))
    for x, mask in enumerate(p_m):
      if (x < 45):
        train_mask = np.load(path_mask + '/' + mask)
        train_mask_np.append(train_mask)
        b = x + 1
    print('Train Mask: Loading patient{0} and it contain {1} files'.format(tr_x+1,b))
  print('           The training set image is lenght {} and it contains the patient: '.format(len(train_image_np)), patient_train_image)
  print('           The training set mask is lenght {} and it contains the patient: '.format(len(train_mask_np)),  patient_train_mask )
  K_fold_train_image.append(train_image_np)
  K_fold_train_mask.append(train_mask_np)
  print('   ')

#checking the creation of the folds
print(len(K_fold_train_image))
for x in K_fold_train_image:
  print(len(x))
print(len(K_fold_train_mask))
for x in K_fold_train_mask:
  print(len(x))

## VALIDATION

#checking the correct selection of the patient directories
i = 0
for tr in K_val:
  i = i + 1
  print('FOLD {}'.format(i))
  print(tr)

  for tr_x in tr:
    path_image = path_image_list_new[tr_x]
    print(path_image)
    path_mask = path_mask_list_new[tr_x]
    print(path_mask)

#defining the validation set for each folds
K_fold_val_image = []
K_fold_val_mask = []
i = 0
for tr in K_val:
  val_image_np = []
  val_mask_np = []

  patient_train_image = []
  patient_train_mask = []
  patient_val_image = []
  patient_val_mask = []

  i = i + 1
  print('FOLD {}'.format(i))
  #print(tr)

  for tr_x in tr:
    path_image = path_image_list_new[tr_x]
    p_i = os.listdir(path_image)
    patient_val_image.append(tr_x+1)
    path_mask = path_mask_list_new[tr_x]
    p_m = os.listdir(path_mask)
    patient_val_mask.append(tr_x+1)
    for x, image in enumerate(p_i):
      if (x < 45):
        val_image = np.load(path_image + '/' + image)
        val_image_np.append(val_image)
        b = x + 1
    print('Val Image: Loading patient{0} and it contain {1} files'.format(tr_x+1,b))
    for x, mask in enumerate(p_m):
      if (x < 45):
        val_mask = np.load(path_mask + '/' + mask)
        val_mask_np.append(val_mask)
        b = x + 1
    print('Val Mask: Loading patient{0} and it contain {1} files'.format(tr_x+1,b))
  print('           The validation set image is lenght {} and it contains the patient: '.format(len(val_image_np)),  patient_val_image)
  print('           The validation set mask is lenght {} and it contains the patient: '.format(len(val_mask_np)),  patient_val_image)
  K_fold_val_image.append(val_image_np)
  K_fold_val_mask.append(val_mask_np)
  print('   ')


#checking the creation of the folds
print(len(K_fold_val_image))
for x in K_fold_val_image:
  print(len(x))
print(len(K_fold_val_mask))
for x in K_fold_val_mask:
  print(len(x))



"""## Trainig the model for each fold"""
K_ssim_history = []
K_val_ssim_history = []
K_acc_history = []
K_val_acc_history = []
K_dice_history = []
K_val_dice_history = []
K_path_model = []

for k in range(0,K_fold):
    print('FOLD {}'.format(k + 1))
    training_image = K_fold_train_image[k]
    training_mask = K_fold_train_mask[k]
    validation_image = K_fold_val_image[k]
    validation_mask = K_fold_val_mask[k]
    print('The image training set is lenght:', len(training_image))
    print('The mask training set is lenght:', len(training_mask))
    print('The image validation set is lenght:', len(validation_image))
    print('The mask validation set is lenght:', len(validation_mask))

    MyModel = myUnet()  # creo istanza dell'oggetto
    myunet = MyModel.get_unet()

    myunet.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss=dice_loss, metrics=[ssim, 'acc', dice_coeff])

    #earlystopper = EarlyStopping(patience=earlystop_patience, verbose=1)
    checkpointer = ModelCheckpoint(os.path.join(os.getcwd(), 'NEW DATASET NUMPY 9paz/NEW DATASET NUMPY 9paz/model_unet_transpose_checkpoint_{:02d}_fold.h5'.format(k+1)), verbose=1, save_best_only=True, monitor='val_loss', mode = 'min')

    fig, axs = plt.subplots(2, 2,  gridspec_kw={'hspace': 0.75, 'wspace': 0.25})
    axs = axs.ravel()

    axs[0].imshow(training_image[0])
    axs[1].imshow(validation_image[0])
    mask = training_mask[0]
    axs[2].imshow(mask[:,:,0], cmap='gray')
    val_mask =validation_mask[0]
    axs[3].imshow(val_mask[:,:,0], cmap='gray')

    plt.show()

    #it inter only inside the init and the on_epoch_end
    training_generator = DataGenerator(training_image, training_mask, batchSize, 3, 1, True, aug)
    validation_data = DataGenerator(validation_image, validation_mask, batchSize, 3, 1, True, aug)

    #it inter only inside the init and the on_epoch_end
    #it enter in len (2), get_item, len, get_item, data_generator(13), on_epoch_end(2)
    #change the epoch
    #len, get_item, len, get_item, data_generator(13), on_epoch_end(2)
    results = myunet.fit(training_generator, validation_data=validation_data, batch_size = batchSize, epochs=2000, callbacks=[checkpointer])
    ssim_history = results.history["ssim"]
    val_ssim_history = results.history["val_ssim"]
    acc_history = results.history["acc"]
    val_acc_history = results.history["val_acc"]
    dice_history = results.history["dice_coeff"]
    val_dice_history = results.history["val_dice_coeff"]
    K_ssim_history.append(ssim_history)
    K_val_ssim_history.append(val_ssim_history)
    K_acc_history.append(acc_history)
    K_val_acc_history.append(val_acc_history)
    K_dice_history.append(dice_history)
    K_val_dice_history.append(val_dice_history)
    K_path_model.append(os.path.join(os.getcwd(), 'NEW DATASET NUMPY 9paz/NEW DATASET NUMPY 9paz/model_unet_transpose_checkpoint_{:02d}_fold.h5'.format(k+1)))
    # saving the metrics' value in a dataset
    with h5py.File(os.path.join(os.getcwd(), 'NEW DATASET NUMPY 9paz/NEW DATASET NUMPY 9paz/FOLD{0}_Metrics_history_transpose.hdf5'.format(k + 1)), 'w') as f:
        f.create_dataset('ssim', data=ssim_history)
        f.create_dataset('val_ssim', data=val_ssim_history)
        f.create_dataset('acc', data=acc_history)
        f.create_dataset('val_acc', data=val_acc_history)
        f.create_dataset('dice', data=dice_history)
        f.create_dataset('val_dice', data=val_dice_history)
        f.close


print('The model created are: ', len(K_path_model))
if(len(K_path_model) == K_fold):
    print('One model is created for each fold')
    print('The models path are: ')

for path in K_path_model:
    print(path)


"""## Visualizing the plot of my metrics"""
## SSIM
fig, axs = plt.subplots(1, int(K_fold),sharex = True, sharey=True,  gridspec_kw={'hspace': 0.75, 'wspace': 0.25})
axs = axs.ravel()
fig.suptitle('Model1 - Structural Similarity Index')
plt.ylim(ymax = 1.0, ymin = 0)

SSIM_1 = K_ssim_history[0]
print(SSIM_1)
SSIM_2 = K_ssim_history[1]
print(SSIM_2)
SSIM_3 = K_ssim_history[2]
print(SSIM_3)
val_ssim_1 =  K_val_ssim_history[0]
print(val_ssim_1)
val_ssim_2 =  K_val_ssim_history[1]
print(val_ssim_2)
val_ssim_3 = K_val_ssim_history[2]
print(val_ssim_3)

axs[0].plot(SSIM_1)
axs[0].plot(val_ssim_1)
axs[0].set_title(' FOLD{0}'.format(1))
axs[0].set_ylabel('SSIM')
axs[1].plot(SSIM_2)
axs[1].plot(val_ssim_2)
axs[1].set_title(' FOLD{0}'.format(2))
axs[1].set_xlabel('Epochs')
axs[2].plot(SSIM_3)
axs[2].plot(val_ssim_3)
axs[2].set_title(' FOLD{0}'.format(3))

fig.legend(['Training', 'Validation'], loc='upper left')
plt.show()
fig.savefig(os.path.join(graph_path, 'folds_dataset(9paz)_transpose_SSIM.png'))

## DICE
fig, axs = plt.subplots(1, int(K_fold), sharex = True, sharey = True, gridspec_kw={'hspace': 0.75, 'wspace': 0.25})
fig.suptitle('Model1 - Dice coefficient')
axs = axs.ravel()
plt.ylim(ymax = 1.0, ymin = 0)

dice_1 = K_dice_history[0]
print(dice_1)
dice_2 = K_dice_history[1]
print(dice_2)
dice_3 = K_dice_history[2]
print(dice_3)
val_dice_1 =  K_val_dice_history[0]
print(val_dice_1)
val_dice_2 =  K_val_dice_history[1]
print(val_dice_2)
val_dice_3 =  K_val_dice_history[2]
print(val_dice_3)

axs[0].plot(dice_1)
axs[0].plot(val_dice_1)
axs[0].set_title(' FOLD{0}'.format(1))
axs[0].set_ylabel('DICE')
axs[1].plot(dice_2)
axs[1].plot(val_dice_2)
axs[1].set_title(' FOLD{0}'.format(2))
axs[1].set_xlabel('Epochs')
axs[2].plot(dice_3)
axs[2].plot(val_dice_3)
axs[2].set_title(' FOLD{0}'.format(3))

fig.legend(['Training', 'Validation'], loc='upper left')
plt.show()
fig.savefig(os.path.join(graph_path, 'folds_dataset(9paz)_transpose_DICE.png'))

## ACCURACY
fig, axs = plt.subplots(1, int(K_fold), sharex = True, gridspec_kw={'hspace': 0.75, 'wspace': 0.25})
fig.suptitle('Model1 - Accuracy')
axs = axs.ravel()
plt.ylim(ymax = 1.0, ymin = 0)

acc_1 = K_acc_history[0]
print(acc_1)
acc_2 = K_acc_history[1]
print(acc_2)
acc_3 = K_acc_history[2]
print(acc_3)
val_acc_1 = K_val_acc_history[0]
print(val_acc_1)
val_acc_2 = K_val_acc_history[1]
print(val_acc_2)
val_acc_3 = K_val_acc_history[2]
print(val_acc_3)

axs[0].plot(acc_1)
axs[0].plot(val_acc_1)
axs[0].set_title(' FOLD{0}'.format(1))
axs[0].set_ylabel('ACCURACY')
axs[1].plot(acc_2)
axs[1].plot(val_acc_2)
axs[1].set_title(' FOLD{0}'.format(2))
axs[1].set_xlabel('Epochs')
axs[2].plot(acc_3)
axs[2].plot(val_acc_3)
axs[2].set_title(' FOLD{0}'.format(3))

fig.legend(['Training', 'Validation'], loc='upper left')
plt.show()
fig.savefig(os.path.join(graph_path, 'folds_dataset(9paz)_transpose_ACC.png'))

"""# TESTING ALL THE FOLDS """

K_test_predicted_np = []
print(K_test_predicted_np)

for k in range(0, K_fold):
  print('FOLD {}'.format(k+1))
  predicted = []
  test_image_np = K_fold_test_image[k]
  print(len(test_image_np))
  patient_test_image = []
  patient_test_mask = []
  patient_test_image.append(K_test[k])
  patient_test_mask.append(K_test[k])
  myunet = load_model(os.path.join(os.getcwd(),'NEW DATASET NUMPY 9paz/NEW DATASET NUMPY 9paz/model_unet_transpose_checkpoint_{:02d}_fold.h5'.format(k + 1)), compile=False)

  for image in test_image_np:
    image = np.array(image / 255.).astype('float32')  # trasforming the image in float 32 with np.max 0 < 1
    prediction = myunet.predict(np.expand_dims(image, 0))
    if (prediction.shape != (1, 256, 256, 1)):
        print('error shape 1')
    if (np.max(prediction) < 0.5):
        print('error MAX! The max is {0}'.format(np.max(prediction)))
    predicted.append(prediction)
  print('The prediction in the fold{0} are {1}'.format(k+1, len(predicted)))

  # deleting the first dimension
  pred = []
  for image in predicted:  # image in float 32 with np.max 0< 1
    p = image[0]
    if (p.shape != (256, 256, 1)):
        print('error shape 2')
    pred.append(p)
  print('The reshaped prediction in the fold{0} are {1}'.format(i, len(pred)))

    # thresholding the prediction
  pred_thr = []
  print('Starting the threshold')
  for image in pred:
    a = np.where(image >= 0.5, 255, 0) # image in float 32 with np.max =  1
    print(np.min(a))
    pred_thr.append(a.astype('uint8'))  # image in float 32 with np.max =  1
  print('Ending the threshold')


  pred_int = pred_thr
  print(pred_int)
  print(np.max(pred_int[0]))

  K_test_predicted_np.append(pred_int)

  print('           The testing set image is lenght {} and it contains the patient: '.format(len(test_image_np)), patient_test_image)
  print('           The prediction set is lenght {} and it contains the patient: '.format(len(test_mask_np)), patient_test_mask)
  print('   ')


print(len(K_test_predicted_np))
print(len(K_test_predicted_np[0]))
print(len(K_test_predicted_np[1]))
print(len(K_test_predicted_np[2]))


#dividing the folds prediction in patients
K_test_predicted_patient = []
K_test_mask_patient = []
K_test_image_patient = []

for k in range(0, K_fold):
    test_pred = K_test_predicted_np[k]
    test_mask = K_fold_test_mask[k]
    test_image = K_fold_test_image[k]

    p1 = []
    p2 = []
    p3 = []
    p1_m = []
    p2_m = []
    p3_m = []
    p1_i = []
    p2_i = []
    p3_i = []
    for x, image in enumerate(test_pred):
        if (x < 45):
           p1.append(image)
        if (45 <= x < 90):
            p2.append(image)
        if (x >= 90):
            p3.append(image)
    for x, image in enumerate(test_mask):
        if (x < 45):
           p1_m.append(image)
        if (45 <= x < 90):
            p2_m.append(image)
        if (x >= 90):
            p3_m.append(image)
    for x, image in enumerate(test_image):
        if (x < 45):
            p1_i.append(image)
        if (45 <= x < 90):
            p2_i.append(image)
        if (x >= 90):
            p3_i.append(image)
    print('P1 ground truth lenght = {0}, prediction lenght = {1}'.format(len(p1_m), len(p1)))
    print('P2 ground truth lenght = {0}, prediction lenght = {1}'.format(len(p2_m), len(p2)))
    print('P3 ground truth lenght = {0}, prediction lenght = {1}'.format(len(p3_m), len(p3)))
    K_test_predicted_patient.append(p1)
    K_test_predicted_patient.append(p2)
    K_test_predicted_patient.append(p3)
    K_test_mask_patient.append(p1_m)
    K_test_mask_patient.append(p2_m)
    K_test_mask_patient.append(p3_m)
    K_test_image_patient.append(p1_i)
    K_test_image_patient.append(p2_i)
    K_test_image_patient.append(p3_i)

print(len(K_test_predicted_patient))
print(len(K_test_mask_patient))
print(len(K_test_image_patient))

for i in range(0,len(K_test_predicted_patient)):
    print(len(K_test_predicted_patient[i]))
    print(len(K_test_mask_patient[i]))
    print(len(K_test_image_patient[i]))

i = 0
for k in range(0,n_patient):
    i = 0
    if(k == 1):
        break
    while (i<len(K_test_predicted_patient[k])):
      image_list = K_test_image_patient[k]
      mask_list = K_test_mask_patient[k]
      predicted_list = K_test_predicted_patient[k]

      fig, axs = plt.subplots(1, 3,  gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(15,15))
      axs = axs.ravel()
      axs[0].imshow(image_list[i])
      axs[0].set_title('Image', fontsize = 15, loc = 'center')
      axs[0].set_yticklabels([])
      axs[0].set_xticklabels([])
      axs[1].imshow(np.squeeze(np.stack((mask_list[i], )*3,axis = -1)))
      axs[1].set_title('Ground Truth', fontsize = 15, loc = 'center')
      axs[1].set_yticklabels([])
      axs[1].set_xticklabels([])
      axs[2].imshow(np.squeeze(np.stack((predicted_list[i], )*3,axis = -1)))
      axs[2].set_title('Predicted', fontsize = 15, loc = 'center')
      axs[2].set_yticklabels([])
      axs[2].set_xticklabels([])
      i = i+1
      plt.show()
      #fig.savefig(os.path.join(figure_path, 'ImageVSMasks_{0}'.format(i-1)))


#Saving the prediction
i = 0
for t in K_test:
  patient_pred = []
  i = i + 1
  print('FOLD {}'.format(i))
  #print(t)
  for t_x in t:
    #print(t_x)
    path_pred = path_predicted_list_new[t_x]
    test_pred = K_test_predicted_patient[t_x]
    print(len(test_pred))
    patient_pred.append(t_x + 1)
    #for image in test_pred:
      #print(np.max(image))

    for x, image in enumerate(test_pred):
      print(image.shape)
      image = image[:,:,0]
      im = Image.fromarray(image, mode = 'L')
      im.save(os.path.join(path_pred, 'predicted_{:03d}.png'.format(x)))
    print('Prediction: Saving patient{0} and it contain {1} files'.format(t_x + 1, len(test_pred)))

  print('           In the Fold{0} the prediction list was lenght {1} and it contained the patient: '.format(i,len(test_pred)),
        patient_pred)



# Metrics to evaluate the prediction

# defining the metric Structural similarity index SSIM
def ssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

def dice_coef(y_true, y_pred):
    img1 = y_true[:, :, 0]
    # img1 = img1.flatten()
    img1 = img1.astype(int)

    img2 = y_pred[:, :, 0]
    # img2 = img2.flatten()
    img2 = img2.astype(int)

    #print(confusion_matrix(np.ravel(img1), np.ravel(img2)))

    tn, fp, fn, tp = np.ravel(confusion_matrix(np.ravel(img1), np.ravel(img2)))
    #print(tn, fp, fn, tp)
    dice = 2 * tp / (2 * tp + fp + fn)
    return dice

def accuracy(y_true, y_pred):
    img1 = y_true[:, :, 0]
    # img1 = img1.flatten()
    img1 = img1.astype(int)

    img2 = y_pred[:, :, 0]
    # img2 = img2.flatten()
    img2 = img2.astype(int)

    #print(confusion_matrix(np.ravel(img1), np.ravel(img2)))

    tn, fp, fn, tp = np.ravel(confusion_matrix(np.ravel(img1), np.ravel(img2)))
    #print(tn, fp, fn, tp)
    accur = (tp + tn) / (tp + fp + tn + fn)
    return accur


SSIM_PATIENTS = []
DICE_PATIENTS = []
ACC_PATIENTS = []

i = 0
for t in K_test:
  i = i + 1
  print('FOLD {}'.format(i))
  #print(t)
  SSIM = []
  DICE = []
  ACC = []
  for t_x in t:
    #print(t_x)
    test_mask = K_test_mask_patient[t_x]
    test_pred = K_test_predicted_patient[t_x]
    print('Evaluating the prediction of the patient{}'.format(t_x+1))
    for x, mask in enumerate(test_mask):
        y_true = mask
        print(y_true.dtype)
        if (y_true.shape != (256, 256, 1)):
            print('error')
        y_pred = test_pred[x]    #are int64 (when do they become int64??)
        y_pred = np.asarray(y_pred).astype('float32')
        print(y_pred.dtype)
        if (y_pred.shape != (256, 256, 1)):
            print('error')
        ssim_temp = ssim(y_true, y_pred)
        ssim_temp_np = ssim_temp.numpy()
        #print(ssim_temp_np)
        SSIM.append(ssim_temp_np)
        dice_temp = dice_coef(y_true, y_pred)
        DICE.append(dice_temp)
        acc_temp = accuracy(y_true, y_pred)
        ACC.append(acc_temp)
    with h5py.File(os.path.join(os.getcwd(), 'NEW DATASET NUMPY 9paz/NEW DATASET NUMPY 9paz/Patient{0}_Metrics_Ground.Truth.VS.Predicted_transpose.hdf5'.format(t_x + 1)), 'w') as f:
        f.create_dataset('SSIM', data=SSIM)
        f.create_dataset('dice', data=DICE)
        f.create_dataset('ACC', data=ACC)
        f.close
    SSIM_PATIENTS.append(SSIM)
    DICE_PATIENTS.append(DICE)
    ACC_PATIENTS.append(ACC)

print(len(SSIM_PATIENTS))
K_SSIM_MEAN = []
for x, ssim_list in enumerate(SSIM_PATIENTS):
    print('Patient {}'.format(x+1))
    print(len(ssim_list))
    print(ssim_list)
    a = 0
    for value in ssim_list:
        a = a + value
    SSIM_MEAN = a / len(ssim_list)
    print('Mean: ', SSIM_MEAN)
    K_SSIM_MEAN.append(SSIM_MEAN)
    print(' ')
print(len(K_SSIM_MEAN))

print(len(DICE_PATIENTS))
K_DICE_MEAN = []
for x, dice_list in enumerate(DICE_PATIENTS):
    print('Patient {}'.format(x+1))
    print(len(dice_list))
    print(dice_list)
    a = 0
    for value in dice_list:
        a = a + value
    DICE_MEAN = a / len(dice_list)
    print('Mean: ', DICE_MEAN)
    K_DICE_MEAN.append(DICE_MEAN)
    print(' ')
print(len(K_DICE_MEAN))

print(len(ACC_PATIENTS))
K_ACC_MEAN = []
for x, acc_list in enumerate(ACC_PATIENTS):
    print('Patient {}'.format(x+1))
    print(len(acc_list))
    print(acc_list)
    a = 0
    for value in acc_list:
        a = a + value
    ACC_MEAN = a / len(acc_list)
    print('Mean: ', ACC_MEAN)
    K_ACC_MEAN.append(ACC_MEAN)
    print(' ')
print(len(K_ACC_MEAN))