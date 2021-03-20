'''RESULT OF THE CHECKIN PROCESS:
'K FOLD - CONTRAST -  NI - REDUCED FOV - POLIMI DATASET':
    IMAGE SHAPE = (256, 256, 3) ; TYPE = uint8
    MASK SHAPE = (256, 256) ; TYPE = uint8

'K FOLD - NI - REDUCED FOV - POLIMI DATASET':
    IMAGE SHAPE = (256, 256, 3) ; TYPE = uint8
    MASK SHAPE = (256, 256, 1) ; TYPE = uint8
'''



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
fold_dataset_path = os.path.join(os.getcwd(), 'K FOLD - 2 - CONTRAST -  NI - REDUCED FOV - POLIMI DATASET')
print(fold_dataset_path)

fold_dataset_path_subfolders = [ f.path for f in os.scandir(fold_dataset_path) if f.is_dir() ]
fold_dataset_path_subfolders.sort()
numpy_path = fold_dataset_path_subfolders[0]
print(numpy_path)

numpy_path_subfolders = [ f.path for f in os.scandir(numpy_path) if f.is_dir() ]
numpy_path_subfolders.sort()
print(numpy_path_subfolders)

k_fold_test_path_subfolders = []
k_fold_train_path_subfolders = []
k_fold_val_path_subfolders = []
for path in numpy_path_subfolders:
    subfolders = [ f.path for f in os.scandir(path) if f.is_dir() ]
    subfolders.sort()
    fold_test_path = subfolders[0]
    fold_train_path = subfolders[1]
    fold_val_path = subfolders[2]
    k_fold_test_path_subfolders.append(fold_test_path)
    k_fold_train_path_subfolders.append(fold_train_path)
    k_fold_val_path_subfolders.append(fold_val_path)

print(k_fold_test_path_subfolders)
print(' ')
print(k_fold_train_path_subfolders)
print(' ')
print(k_fold_val_path_subfolders)


#TEST CHECKING
for x, path in enumerate(k_fold_test_path_subfolders):
    print(' ')
    print('Test')
    image_list_np = []
    mask_list_np = []
    subfolders = [ f.path for f in os.scandir(path) if f.is_dir() ]
    subfolders.sort()
    Image_path = subfolders[0]
    Mask_path = subfolders[1]
    image_list = os.listdir(Image_path)
    mask_list = os.listdir(Mask_path)
    if (len(image_list) != len(mask_list)):
        print('Error! Image and Mask lists have different lenght')
    for image in image_list:
        im = np.load(Image_path + '/' + image)
        image_list_np.append(im)
    s = image_list_np[0].shape
    t = image_list_np[0].dtype
    print('IMAGE: FOLD {0} - Shape {1} - Type {2} - {3}'.format(x+1, s, t, len(image_list_np)))
    for image in image_list_np:
        if(image.shape!= s):
            print('error shape')
        if(image.dtype != t):
            print('error type')
    for mask in mask_list:
        im = np.load(Mask_path + '/' + mask)
        mask_list_np.append(im)
    s = mask_list_np[0].shape
    t = mask_list_np[0].dtype
    print('MASK: FOLD {0} - Shape {1} - Typr {2} - {3}'.format(x+1, s, t, len(mask_list_np)))
    for image in mask_list_np:
        if(image.shape!= s):
            print('error shape')
        if(image.dtype != t):
            print('error type')
    '''l = 0
    while (l < len(image_list_np)):
        plt.close("all")
        fig, axs = plt.subplots(4, 5, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(10, 10))
        fig.suptitle('FOLD {0} - Testing Set'.format(x+1), fontsize=20)
        axs = axs.ravel()
        axs[0].imshow(image_list_np[l])
        axs[0].set_yticklabels([])
        axs[0].set_xticklabels([])
        axs[1].imshow(image_list_np[l+1])
        axs[1].set_yticklabels([])
        axs[1].set_xticklabels([])
        axs[2].imshow(image_list_np[l+2])
        axs[2].set_yticklabels([])
        axs[2].set_xticklabels([])
        axs[3].imshow(image_list_np[l+3])
        axs[3].set_yticklabels([])
        axs[3].set_xticklabels([])
        axs[4].imshow(image_list_np[l+4])
        axs[4].set_yticklabels([])
        axs[4].set_xticklabels([])
        axs[5].imshow(np.squeeze(np.stack((mask_list_np[l],) * 3, axis=-1)))
        axs[5].set_yticklabels([])
        axs[5].set_xticklabels([])
        axs[6].imshow(np.squeeze(np.stack((mask_list_np[l+1],) * 3, axis=-1)))
        axs[6].set_yticklabels([])
        axs[6].set_xticklabels([])
        axs[7].imshow(np.squeeze(np.stack((mask_list_np[l+2],) * 3, axis=-1)))
        axs[7].set_yticklabels([])
        axs[7].set_xticklabels([])
        axs[8].imshow(np.squeeze(np.stack((mask_list_np[l + 3],) * 3, axis=-1)))
        axs[8].set_yticklabels([])
        axs[8].set_xticklabels([])
        axs[9].imshow(np.squeeze(np.stack((mask_list_np[l + 4],) * 3, axis=-1)))
        axs[9].set_yticklabels([])
        axs[9].set_xticklabels([])
        axs[10].imshow(image_list_np[l+5])
        axs[10].set_yticklabels([])
        axs[10].set_xticklabels([])
        axs[11].imshow(image_list_np[l + 6])
        axs[11].set_yticklabels([])
        axs[11].set_xticklabels([])
        axs[12].imshow(image_list_np[l + 7])
        axs[12].set_yticklabels([])
        axs[12].set_xticklabels([])
        axs[13].imshow(image_list_np[l + 8])
        axs[13].set_yticklabels([])
        axs[13].set_xticklabels([])
        axs[14].imshow(image_list_np[l + 9])
        axs[14].set_yticklabels([])
        axs[14].set_xticklabels([])
        axs[15].imshow(np.squeeze(np.stack((mask_list_np[l+5],) * 3, axis=-1)))
        axs[15].set_yticklabels([])
        axs[15].set_xticklabels([])
        axs[16].imshow(np.squeeze(np.stack((mask_list_np[l + 6],) * 3, axis=-1)))
        axs[16].set_yticklabels([])
        axs[16].set_xticklabels([])
        axs[17].imshow(np.squeeze(np.stack((mask_list_np[l + 7],) * 3, axis=-1)))
        axs[17].set_yticklabels([])
        axs[17].set_xticklabels([])
        axs[18].imshow(np.squeeze(np.stack((mask_list_np[l + 8],) * 3, axis=-1)))
        axs[18].set_yticklabels([])
        axs[18].set_xticklabels([])
        axs[19].imshow(np.squeeze(np.stack((mask_list_np[l + 9],) * 3, axis=-1)))
        axs[19].set_yticklabels([])
        axs[19].set_xticklabels([])
        l = l+10
        plt.show(block=False)'''



#TRAIN CHECKING
for x, path in enumerate(k_fold_train_path_subfolders):
    print(' ')
    print('Train')
    image_list_np = []
    mask_list_np = []
    subfolders = [ f.path for f in os.scandir(path) if f.is_dir() ]
    subfolders.sort()
    Image_path = subfolders[0]
    Mask_path = subfolders[1]
    image_list = os.listdir(Image_path)
    mask_list = os.listdir(Mask_path)
    if (len(image_list) != len(mask_list)):
        print('Error! Image and Mask lists have different lenght')
    for image in image_list:
        im = np.load(Image_path + '/' + image)
        image_list_np.append(im)
    s = image_list_np[0].shape
    t = image_list_np[0].dtype
    print('IMAGE: FOLD {0} - Shape {1} - Type {2} - {3}'.format(x+1, s, t, len(image_list_np)))
    for image in image_list_np:
        if (image.shape != s):
            print('error shape')
        if (image.dtype != t):
            print('error type')
    for mask in mask_list:
        im = np.load(Mask_path + '/' + mask)
        mask_list_np.append(im)
    s = mask_list_np[0].shape
    t = mask_list_np[0].dtype
    print('MASK: FOLD {0} - Shape {1} - Typr {2} - {3}'.format(x+1, s, t, len(mask_list_np)))
    for image in mask_list_np:
        if (image.shape != s):
            print('error shape')
        if (image.dtype != t):
            print('error type')
    '''l = 0
    while (l < len(image_list_np)):
        fig, axs = plt.subplots(4, 5, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(10, 10))
        fig.suptitle('FOLD {0} - Training Set'.format(x+1), fontsize=20)
        axs = axs.ravel()
        axs[0].imshow(image_list_np[l])
        axs[0].set_yticklabels([])
        axs[0].set_xticklabels([])
        axs[1].imshow(image_list_np[l+1])
        axs[1].set_yticklabels([])
        axs[1].set_xticklabels([])
        axs[2].imshow(image_list_np[l+2])
        axs[2].set_yticklabels([])
        axs[2].set_xticklabels([])
        axs[3].imshow(image_list_np[l+3])
        axs[3].set_yticklabels([])
        axs[3].set_xticklabels([])
        axs[4].imshow(image_list_np[l+4])
        axs[4].set_yticklabels([])
        axs[4].set_xticklabels([])
        axs[5].imshow(np.squeeze(np.stack((mask_list_np[l],) * 3, axis=-1)))
        axs[5].set_yticklabels([])
        axs[5].set_xticklabels([])
        axs[6].imshow(np.squeeze(np.stack((mask_list_np[l+1],) * 3, axis=-1)))
        axs[6].set_yticklabels([])
        axs[6].set_xticklabels([])
        axs[7].imshow(np.squeeze(np.stack((mask_list_np[l+2],) * 3, axis=-1)))
        axs[7].set_yticklabels([])
        axs[7].set_xticklabels([])
        axs[8].imshow(np.squeeze(np.stack((mask_list_np[l + 3],) * 3, axis=-1)))
        axs[8].set_yticklabels([])
        axs[8].set_xticklabels([])
        axs[9].imshow(np.squeeze(np.stack((mask_list_np[l + 4],) * 3, axis=-1)))
        axs[9].set_yticklabels([])
        axs[9].set_xticklabels([])
        axs[10].imshow(image_list_np[l+5])
        axs[10].set_yticklabels([])
        axs[10].set_xticklabels([])
        axs[11].imshow(image_list_np[l + 6])
        axs[11].set_yticklabels([])
        axs[11].set_xticklabels([])
        axs[12].imshow(image_list_np[l + 7])
        axs[12].set_yticklabels([])
        axs[12].set_xticklabels([])
        axs[13].imshow(image_list_np[l + 8])
        axs[13].set_yticklabels([])
        axs[13].set_xticklabels([])
        axs[14].imshow(image_list_np[l + 9])
        axs[14].set_yticklabels([])
        axs[14].set_xticklabels([])
        axs[15].imshow(np.squeeze(np.stack((mask_list_np[l+5],) * 3, axis=-1)))
        axs[15].set_yticklabels([])
        axs[15].set_xticklabels([])
        axs[16].imshow(np.squeeze(np.stack((mask_list_np[l + 6],) * 3, axis=-1)))
        axs[16].set_yticklabels([])
        axs[16].set_xticklabels([])
        axs[17].imshow(np.squeeze(np.stack((mask_list_np[l + 7],) * 3, axis=-1)))
        axs[17].set_yticklabels([])
        axs[17].set_xticklabels([])
        axs[18].imshow(np.squeeze(np.stack((mask_list_np[l + 8],) * 3, axis=-1)))
        axs[18].set_yticklabels([])
        axs[18].set_xticklabels([])
        axs[19].imshow(np.squeeze(np.stack((mask_list_np[l + 9],) * 3, axis=-1)))
        axs[19].set_yticklabels([])
        axs[19].set_xticklabels([])
        l = l+10
        plt.show(block=False)
        plt.close()'''


#VALIDATION CHECKING
for x, path in enumerate(k_fold_val_path_subfolders):
    print(' ')
    print('Val')
    image_list_np = []
    mask_list_np = []
    subfolders = [ f.path for f in os.scandir(path) if f.is_dir() ]
    subfolders.sort()
    Image_path = subfolders[0]
    Mask_path = subfolders[1]
    image_list = os.listdir(Image_path)
    mask_list = os.listdir(Mask_path)
    if (len(image_list) != len(mask_list)):
        print('Error! Image and Mask lists have different lenght')
    for image in image_list:
        im = np.load(Image_path + '/' + image)
        image_list_np.append(im)
    s = image_list_np[0].shape
    t = image_list_np[0].dtype
    print('IMAGE: FOLD {0} - Shape {1} - Type {2} - {3}'.format(x+1, s, t, len(image_list_np)))
    for image in image_list_np:
        if (image.shape != s):
            print('error shape')
        if (image.dtype != t):
            print('error type')
    for mask in mask_list:
        im = np.load(Mask_path + '/' + mask)
        mask_list_np.append(im)
    s = mask_list_np[0].shape
    t = mask_list_np[0].dtype
    print('MASK: FOLD {0} - Shape {1} - Typr {2} - {3}'.format(x+1, s, t, len(mask_list_np)))
    for image in mask_list_np:
        if (image.shape != s):
            print('error shape')
        if (image.dtype != t):
            print('error type')
    '''l = 0
    while (l < len(image_list_np)):
        fig, axs = plt.subplots(4, 5, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(10, 10))
        fig.suptitle('FOLD {0} - Training Set'.format(x+1), fontsize=20)
        axs = axs.ravel()
        axs[0].imshow(image_list_np[l])
        axs[0].set_yticklabels([])
        axs[0].set_xticklabels([])
        axs[1].imshow(image_list_np[l+1])
        axs[1].set_yticklabels([])
        axs[1].set_xticklabels([])
        axs[2].imshow(image_list_np[l+2])
        axs[2].set_yticklabels([])
        axs[2].set_xticklabels([])
        axs[3].imshow(image_list_np[l+3])
        axs[3].set_yticklabels([])
        axs[3].set_xticklabels([])
        axs[4].imshow(image_list_np[l+4])
        axs[4].set_yticklabels([])
        axs[4].set_xticklabels([])
        axs[5].imshow(np.squeeze(np.stack((mask_list_np[l],) * 3, axis=-1)))
        axs[5].set_yticklabels([])
        axs[5].set_xticklabels([])
        axs[6].imshow(np.squeeze(np.stack((mask_list_np[l+1],) * 3, axis=-1)))
        axs[6].set_yticklabels([])
        axs[6].set_xticklabels([])
        axs[7].imshow(np.squeeze(np.stack((mask_list_np[l+2],) * 3, axis=-1)))
        axs[7].set_yticklabels([])
        axs[7].set_xticklabels([])
        axs[8].imshow(np.squeeze(np.stack((mask_list_np[l + 3],) * 3, axis=-1)))
        axs[8].set_yticklabels([])
        axs[8].set_xticklabels([])
        axs[9].imshow(np.squeeze(np.stack((mask_list_np[l + 4],) * 3, axis=-1)))
        axs[9].set_yticklabels([])
        axs[9].set_xticklabels([])
        axs[10].imshow(image_list_np[l+5])
        axs[10].set_yticklabels([])
        axs[10].set_xticklabels([])
        axs[11].imshow(image_list_np[l + 6])
        axs[11].set_yticklabels([])
        axs[11].set_xticklabels([])
        axs[12].imshow(image_list_np[l + 7])
        axs[12].set_yticklabels([])
        axs[12].set_xticklabels([])
        axs[13].imshow(image_list_np[l + 8])
        axs[13].set_yticklabels([])
        axs[13].set_xticklabels([])
        axs[14].imshow(image_list_np[l + 9])
        axs[14].set_yticklabels([])
        axs[14].set_xticklabels([])
        axs[15].imshow(np.squeeze(np.stack((mask_list_np[l+5],) * 3, axis=-1)))
        axs[15].set_yticklabels([])
        axs[15].set_xticklabels([])
        axs[16].imshow(np.squeeze(np.stack((mask_list_np[l + 6],) * 3, axis=-1)))
        axs[16].set_yticklabels([])
        axs[16].set_xticklabels([])
        axs[17].imshow(np.squeeze(np.stack((mask_list_np[l + 7],) * 3, axis=-1)))
        axs[17].set_yticklabels([])
        axs[17].set_xticklabels([])
        axs[18].imshow(np.squeeze(np.stack((mask_list_np[l + 8],) * 3, axis=-1)))
        axs[18].set_yticklabels([])
        axs[18].set_xticklabels([])
        axs[19].imshow(np.squeeze(np.stack((mask_list_np[l + 9],) * 3, axis=-1)))
        axs[19].set_yticklabels([])
        axs[19].set_xticklabels([])
        l = l+10
        plt.show(block=False)
        plt.close()'''