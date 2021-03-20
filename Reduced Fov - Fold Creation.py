#Import Libraries


import os
import cv2
from os import listdir
from PIL import Image, ImageEnhance, ImageOps
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


"""# LOADING ALL PATIENTS DIRECTORIES"""

#select my directory (inside there are all the images)
Reduced_fov_Polimi_dataset = os.path.join(os.getcwd(), 'NI - REDUCED FOV - POLIMI DATASET')
print(Reduced_fov_Polimi_dataset)

#visualize the folders inside my directory
Polimi_dataset_subfolders = [ f.path for f in os.scandir(Reduced_fov_Polimi_dataset) if f.is_dir() ]
Polimi_dataset_subfolders.sort()
print(Polimi_dataset_subfolders)

Numpy_subfolders = [ f.path for f in os.scandir(Polimi_dataset_subfolders[0]) if f.is_dir() ]
Numpy_subfolders.sort()
print(Numpy_subfolders )
print(' ')
Png_subfolders = [ f.path for f in os.scandir(Polimi_dataset_subfolders[1]) if f.is_dir() ]
Png_subfolders.sort()
print(Png_subfolders)

Image_numpy_subfolders = [f.path for f in os.scandir(Numpy_subfolders[0]) if f.is_dir() ]
Image_numpy_subfolders.sort()
print(Image_numpy_subfolders)
print(' ')
Mask_numpy_subfolders = [f.path for f in os.scandir(Numpy_subfolders[1]) if f.is_dir() ]
Mask_numpy_subfolders.sort()
print(Mask_numpy_subfolders)


path_image_list_new = []    #directories of the image for every patient
for path in Image_numpy_subfolders:
    path_image_list_new.append(path)
print(path_image_list_new)

path_mask_list_new = []    #directories of the image for every patient
for path in Mask_numpy_subfolders:
    path_mask_list_new.append(path)
print(path_mask_list_new)

n_patient = len(path_mask_list_new)
print(n_patient)

"""# FOLDERS TO SAVE MY DATASET  """

#creating the dataset subfolder
directory = 'K FOLD - NI - REDUCED FOV - POLIMI DATASET'
new_dataset_path = os.path.join(os.getcwd(), directory)
try:
  os.mkdir(new_dataset_path)
except:
  pass

numpy_path = os.path.join(new_dataset_path, 'Numpy')
png_path = os.path.join(new_dataset_path, 'PNG')
try:
  os.mkdir(numpy_path)
  os.mkdir(png_path)
except:
  pass

type_path_list = [numpy_path, png_path]
fold_path_list_numpy = []
fold_path_list_png = []
for l, path in enumerate(type_path_list):
    i = 1
    while(i<=n_patient):
        fold_path = os.path.join(path, 'FOLD_{:02d}'.format(i))
        i = i+1
        test_path = os.path.join(fold_path, 'TEST')
        train_path = os.path.join(fold_path, 'TRAIN')
        val_path = os.path.join(fold_path, 'VALIDATION')
        try:
            os.mkdir(fold_path)
            os.mkdir(train_path)
            os.mkdir(test_path)
            os.mkdir(val_path)
        except:
            pass
        train_image_path = os.path.join(train_path, 'image')
        train_mask_path = os.path.join(train_path, 'mask')
        val_image_path = os.path.join(val_path, 'image')
        val_mask_path = os.path.join(val_path, 'mask')
        test_image_path = os.path.join(test_path, 'image')
        test_mask_path = os.path.join(test_path, 'mask - ground truth')
        try:
            os.mkdir(train_image_path)
            os.mkdir(train_mask_path)
            os.mkdir(val_image_path)
            os.mkdir(val_mask_path)
            os.mkdir(test_image_path)
            os.mkdir(test_mask_path)
        except:
            pass
        if (l == 0):
            fold_path_list_numpy.append(fold_path)
        else:
            fold_path_list_png.append(fold_path)
print(fold_path_list_numpy)
print(' ')
print(fold_path_list_png)


"""##Creating the Folds"""
n_patient = len(Mask_numpy_subfolders)

size_test = 1
size_train = 9
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
x = 0
for train, test in kfold.split(path_image_list_new):  #genera 12
  #print('    ')
  #print(train)
  print(test)
  #np.random.shuffle(train)
  np.random.seed(x)
  np.random.shuffle(train)
  val = train[0:2]
  print(val)
  train = train[2:]
  print(train)
  print(' ')
  x = x + 1
  K_test.append(test)
  K_train.append(train)
  K_val.append(val)

print(K_test)

print(K_train)

print(K_val)

'''Test set = 1 ; 2 ; 3 ; 4 ; 6 ; 5; 7 ; 8 ; 9 ; 10 ; 11 ; 12
    Val set = 5 - 10 ;  3 - 4 ; 5 - 1 ; 6 - 5 ; 3 - 9 ; 6 - 9 ; 9 - 1 ; 9 - 5 ;  6 - 7 ;  7 - 4 ; 2 - 6 ; 6 - 10 
    Train set =  3, 11,  7,  2,  8,  9,  4,  1,  6 ;  , 10,  2,  7,  0,  8, 11,  9,  6 ;  10,  6,  0,  8,  3,  4,  7, 11,  9 ; 
            1,  2, 10,  7,  8,  0,  4,  9, 11 ;    5, 10,  2,  7,  0,  1,  6,  8, 11 ;    2, 11,  4,  8,  1,  0, 10,  7,  3; 
          8,  0,  7,  5,  2,  4,  3, 10, 11 ;  0,  2,  1, 11,  8,  3,  6, 10,  4 ; 10, 11,  0,  2,  5,  9,  1,  4,  3; 
           0,  2,  1, 11, 10,  3,  8,  6,  5 ;  8,  5,  7, 11,  3,  1,  0,  4,  9 ; 2, 3, 8, 4, 5, 7, 1, 0, 9 '''

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
        test_image = np.load(path_image + '/' + image)   # the images are in uint8 and np.max = 255
        if ( test_image.shape != (256, 256, 3)):
          print('error:', test_image.shape)
        test_image_np.append(test_image)
        b = x + 1
    print('Test Image: Loading patient{0} and it contain {1} files'.format(t_x+1,b))
    for x,mask in enumerate(p_m):
        test_mask = np.load(path_mask + '/' + mask)     # the masks are uint8 with different np.max and have shape (256, 256, 1)
        #test_mask = np.asarray(test_mask/np.max(test_mask)).astype('float32') #binarizing the masks and transforming them into float32 (we made it in the testing phase)
        if (test_mask.shape != (256, 256, 1)):
            print('error')
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

#Saving the numpy image
print(len(K_fold_test_image))
for x, test_image in enumerate(K_fold_test_image):
  k_fold_path_list_numpy_subfolders = [ f.path for f in os.scandir(fold_path_list_numpy[x]) if f.is_dir() ]
  k_fold_path_list_numpy_subfolders.sort()
  test_path_subfolders = [ f.path for f in os.scandir(k_fold_path_list_numpy_subfolders[0]) if f.is_dir() ]
  image_path = test_path_subfolders[0]
  print(image_path)

  k_fold_path_list_png_subfolders = [ f.path for f in os.scandir(fold_path_list_png[x]) if f.is_dir() ]
  k_fold_path_list_png_subfolders.sort()
  test_path_subfolders_png = [ f.path for f in os.scandir(k_fold_path_list_png_subfolders[0]) if f.is_dir() ]
  image_path_png = test_path_subfolders_png[0]
  print(image_path_png)

  print('FOLD{}'.format(x+1))
  print(len(test_image))
  for l, image in enumerate(test_image):
      np.save(os.path.join(image_path, '{:03d}.npy'.format(l)), np.asarray(image))
      im = Image.fromarray(image)
      im.save(os.path.join(image_path_png, '{:03d}.png'.format(l)))

#Saving the numpy mask
print(len(K_fold_test_mask))
for x, test_mask in enumerate(K_fold_test_mask):
  k_fold_path_list_numpy_subfolders = [ f.path for f in os.scandir(fold_path_list_numpy[x]) if f.is_dir() ]
  k_fold_path_list_numpy_subfolders.sort()
  test_path_subfolders = [ f.path for f in os.scandir(k_fold_path_list_numpy_subfolders[0]) if f.is_dir() ]
  mask_path = test_path_subfolders[1]
  print(mask_path)

  k_fold_path_list_png_subfolders = [ f.path for f in os.scandir(fold_path_list_png[x]) if f.is_dir() ]
  k_fold_path_list_png_subfolders.sort()
  test_path_subfolders_png = [ f.path for f in os.scandir(k_fold_path_list_png_subfolders[0]) if f.is_dir() ]
  mask_path_png = test_path_subfolders_png[1]
  print(mask_path_png)

  print('FOLD{}'.format(x+1))
  print(len(test_mask))
  for l, mask in enumerate(test_mask):
      np.save(os.path.join(mask_path, '{:03d}.npy'.format(l)), np.asarray(mask))
      print(mask.shape)
      mask = mask[:,:,0]
      im = Image.fromarray(mask)
      im.save(os.path.join(mask_path_png, '{:03d}.png'.format(l)))



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
        train_image = np.load(path_image + '/' + image)
        train_image_np.append(train_image)
        b = x + 1
    print('Train Image: Loading patient{0} and it contain {1} files'.format(tr_x+1,b))
    for x, mask in enumerate(p_m):
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


#Saving the numpy image
print(len(K_fold_train_mask))
for x, train_image in enumerate(K_fold_train_image):
    k_fold_path_list_numpy_subfolders = [f.path for f in os.scandir(fold_path_list_numpy[x]) if f.is_dir()]
    k_fold_path_list_numpy_subfolders.sort()
    train_path_subfolders = [f.path for f in os.scandir(k_fold_path_list_numpy_subfolders[1]) if f.is_dir()]
    image_path = train_path_subfolders[0]
    print(image_path)

    k_fold_path_list_png_subfolders = [f.path for f in os.scandir(fold_path_list_png[x]) if f.is_dir()]
    k_fold_path_list_png_subfolders.sort()
    train_path_subfolders_png = [f.path for f in os.scandir(k_fold_path_list_png_subfolders[1]) if f.is_dir()]
    image_path_png = train_path_subfolders_png[0]
    print(image_path_png)

    print('FOLD{}'.format(x + 1))
    print(len(train_image))
    for l, image in enumerate(train_image):
        np.save(os.path.join(image_path, '{:03d}.npy'.format(l)), np.asarray(image))
        im = Image.fromarray(image)
        im.save(os.path.join(image_path_png, '{:03d}.png'.format(l)))

# Saving the numpy mask
print(len(K_fold_train_mask))
for x, test_mask in enumerate(K_fold_train_mask):
    k_fold_path_list_numpy_subfolders = [f.path for f in os.scandir(fold_path_list_numpy[x]) if f.is_dir()]
    k_fold_path_list_numpy_subfolders.sort()
    train_path_subfolders = [f.path for f in os.scandir(k_fold_path_list_numpy_subfolders[1]) if f.is_dir()]
    mask_path = train_path_subfolders[1]
    print(mask_path)

    k_fold_path_list_png_subfolders = [f.path for f in os.scandir(fold_path_list_png[x]) if f.is_dir()]
    k_fold_path_list_png_subfolders.sort()
    train_path_subfolders_png = [f.path for f in os.scandir(k_fold_path_list_png_subfolders[1]) if f.is_dir()]
    mask_path_png = train_path_subfolders_png[1]
    print(mask_path_png)

    print('FOLD{}'.format(x + 1))
    print(len(test_mask))
    for l, mask in enumerate(test_mask):
        np.save(os.path.join(mask_path, '{:03d}.npy'.format(l)), np.asarray(mask))
        print(mask.shape)
        mask = mask[:,:,0]
        im = Image.fromarray(mask)
        im.save(os.path.join(mask_path_png, '{:03d}.png'.format(l)))



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
        val_image = np.load(path_image + '/' + image)
        val_image_np.append(val_image)
        b = x + 1
    print('Val Image: Loading patient{0} and it contain {1} files'.format(tr_x+1,b))
    for x, mask in enumerate(p_m):
        val_mask = np.load(path_mask + '/' + mask)
        val_mask_np.append(val_mask)
        b = x + 1
    print('Val Mask: Loading patient{0} and it contain {1} files'.format(tr_x+1,b))
  print('           The validation set image is lenght {} and it contains the patient: '.format(len(val_image_np)),  patient_val_image)
  print('           The validation set mask is lenght {} and it contains the patient: '.format(len(val_mask_np)),  patient_val_image)
  K_fold_val_image.append(val_image_np)
  K_fold_val_mask.append(val_mask_np)
  print('   ')

#Saving the numpy image
print(len(K_fold_val_image))
for x, test_image in enumerate(K_fold_val_image):
    k_fold_path_list_numpy_subfolders = [f.path for f in os.scandir(fold_path_list_numpy[x]) if f.is_dir()]
    k_fold_path_list_numpy_subfolders.sort()
    val_path_subfolders = [f.path for f in os.scandir(k_fold_path_list_numpy_subfolders[2]) if f.is_dir()]
    image_path = val_path_subfolders[0]
    print(image_path)

    k_fold_path_list_png_subfolders = [f.path for f in os.scandir(fold_path_list_png[x]) if f.is_dir()]
    k_fold_path_list_png_subfolders.sort()
    val_path_subfolders_png = [f.path for f in os.scandir(k_fold_path_list_png_subfolders[2]) if f.is_dir()]
    image_path_png = val_path_subfolders_png[0]
    print(image_path_png)

    print('FOLD{}'.format(x + 1))
    print(len(test_image))
    for l, image in enumerate(test_image):
        np.save(os.path.join(image_path, '{:03d}.npy'.format(l)), np.asarray(image))
        im = Image.fromarray(image)
        im.save(os.path.join(image_path_png, '{:03d}.png'.format(l)))

# Saving the numpy mask
print(len(K_fold_val_mask))
for x, test_mask in enumerate(K_fold_val_mask):
    k_fold_path_list_numpy_subfolders = [f.path for f in os.scandir(fold_path_list_numpy[x]) if f.is_dir()]
    k_fold_path_list_numpy_subfolders.sort()
    val_path_subfolders = [f.path for f in os.scandir(k_fold_path_list_numpy_subfolders[2]) if f.is_dir()]
    mask_path = val_path_subfolders[1]
    print(mask_path)

    k_fold_path_list_png_subfolders = [f.path for f in os.scandir(fold_path_list_png[x]) if f.is_dir()]
    k_fold_path_list_png_subfolders.sort()
    val_path_subfolders_png = [f.path for f in os.scandir(k_fold_path_list_png_subfolders[2]) if f.is_dir()]
    mask_path_png = val_path_subfolders_png[1]
    print(mask_path_png)

    print('FOLD{}'.format(x + 1))
    print(len(test_mask))
    for l, mask in enumerate(test_mask):
        np.save(os.path.join(mask_path, '{:03d}.npy'.format(l)), np.asarray(mask))
        print(mask.shape)
        mask = mask[:,:,0]
        im = Image.fromarray(mask)
        im.save(os.path.join(mask_path_png, '{:03d}.png'.format(l)))

#checking the creation of the folds
print(len(K_fold_val_image))
for x in K_fold_val_image:
  print(len(x))
print(len(K_fold_val_mask))
for x in K_fold_val_mask:
  print(len(x))
