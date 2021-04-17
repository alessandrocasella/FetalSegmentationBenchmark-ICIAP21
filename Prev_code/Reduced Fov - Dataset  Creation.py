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
Polimi_dataset = os.path.join(os.getcwd(), 'POLIMI DATASET final')
print(Polimi_dataset)

#visualize the folders inside my directory
Polimi_dataset_subfolders = [ f.path for f in os.scandir(Polimi_dataset) if f.is_dir() ]
Polimi_dataset_subfolders.sort()
print(Polimi_dataset_subfolders)

#visualize the folders inside my directory
Polimi_dataset_subfolders = [ f.path for f in os.scandir(Polimi_dataset) if f.is_dir() ]
Polimi_dataset_subfolders.sort()
print(Polimi_dataset_subfolders)

#defining the directories inside the folder Polimi Dataset founded in the previous line
patients_directories = []
for x, path in enumerate(Polimi_dataset_subfolders):
   newpath = [ f.path for f in os.scandir(path) if f.is_dir() ]
   newpath.sort()
   patients_directories.append(newpath)
patients_directories.sort()
print(patients_directories)

#defining the Patient direcoties
path_image_list_new = []
path_mask_list_new = []
for x, path in enumerate(patients_directories):
  print('Patient{}'.format(x+1))
  paziente = path
  paziente_image = paziente[0]
  paziente_mask = paziente[1]
  print(paziente_image)
  print(paziente_mask)
  path_image_list_new.append(paziente_image)
  path_mask_list_new.append(paziente_mask)

print(path_image_list_new)
print(path_mask_list_new)

n_patient = len(path_image_list_new)
print(n_patient)

if (len(path_mask_list_new) != n_patient):
  print('Error in the directories')

"""# FOLDERS TO SAVE MY DATASET  """

#creating the dataset subfolder
directory = 'REDUCED FOV - POLIMI DATASET'
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
patient_image_path_list_numpy = []
patient_mask_path_list_numpy = []
patient_image_path_list_png = []
patient_mask_path_list_png = []
for l, path in enumerate(type_path_list):
    image_path = os.path.join(path, 'Image_Reduced_fov')
    mask_path = os.path.join(path, 'Mask_Reduced_fov')
    try:
        os.mkdir(image_path)
        os.mkdir(mask_path)
    except:
        pass
    i = 1
    while(i<=n_patient):
        patient_image_path = os.path.join(image_path, 'Patient_{:02d}'.format(i))
        patient_mask_path = os.path.join(mask_path, 'Patient_{:02d}'.format(i))
        i = i+1
        try:
            os.mkdir(patient_image_path)
            os.mkdir(patient_mask_path)
        except:
            pass
        if (l == 0):
            patient_image_path_list_numpy.append(patient_image_path)
            patient_mask_path_list_numpy.append(patient_mask_path)
        else:
            patient_image_path_list_png.append(patient_image_path)
            patient_mask_path_list_png.append(patient_mask_path)
print(patient_image_path_list_numpy)
print(patient_mask_path_list_numpy)
print(' ')
print(patient_image_path_list_png)
print(patient_mask_path_list_png)


"""# PREPROCESSING - REDUCING FOV"""
patient_cropped_image_list = []
patient_cropped_mask_list = []
for x, path in enumerate(path_image_list_new):
    image_list = os.listdir(path)
    image_list.sort()
    path_mask = path_mask_list_new[x]
    mask_list = os.listdir(path_mask)
    mask_list.sort()
    print('Patient{0}: {1} images and {2}'.format(x+1, len(image_list), len(mask_list)))
    print(path)
    print(path_mask)
    image_list_cropped = []
    mask_list_cropped = []
    if (x == 0):
        (left, upper, right, lower) = (85, 55, 560, 530)
    if (x == 1):
        (left, upper, right, lower) = (95, 15, 605, 525)
    if (x == 2):
        (left, upper, right, lower) = (120, 50, 600, 530)
    if (x == 3):
        (left, upper, right, lower) = (184, 120, 520, 456)
    if (x == 4):
        (left, upper, right, lower) = (150, 80, 570, 500)
    if (x == 5):
        (left, upper, right, lower) = (200, 120, 520, 440)
    if (x == 6):
        (left, upper, right, lower) = (150, 80, 570, 496)
    if (x == 7):
        (left, upper, right, lower) = (110, 60, 540, 490)
    if (x == 8):
        (left, upper, right, lower) = (110, 60, 550, 500)
    if (x == 9):
        (left, upper, right, lower) = (155, 75, 575, 495)
    if (x == 10):
        (left, upper, right, lower) = (195, 125, 515, 445)
    if (x == 11):
        (left, upper, right, lower) = (160, 125, 450, 415) #
    for image in image_list:
        real = Image.open(path + '/' + image)
        # Here the image "im" is cropped and assigned to new variable im_crop
        im_crop = real.crop((left, upper, right, lower))
        # print(im_crop)
        fig, axs = plt.subplots(1, 2)
        fig.suptitle('Image - Patient {0}'.format(x + 1), fontsize=20)
        axs = axs.ravel()
        axs[0].imshow(np.asarray(real))
        axs[0].set_yticklabels([])
        axs[0].set_xticklabels([])
        axs[1].imshow(np.asarray(im_crop))
        axs[1].set_yticklabels([])
        axs[1].set_xticklabels([])
        plt.show()
        image_list_cropped.append(im_crop)
    patient_cropped_image_list.append(image_list_cropped)
    for l, image in enumerate(image_list_cropped):
        image.save(os.path.join(patient_image_path_list_png[x], 'image_reduced_fov_{:03d}.png'.format(l)), format="png")
    for mask in mask_list:
        real = Image.open(path_mask + '/' + mask)
        im_crop = real.crop((left, upper, right, lower))
        # print(im_crop)
        fig, axs = plt.subplots(1, 2)
        fig.suptitle('Mask - Patient {0}'.format(x + 1), fontsize=20)
        axs = axs.ravel()
        axs[0].imshow(np.asarray(real))
        axs[0].set_yticklabels([])
        axs[0].set_xticklabels([])
        axs[1].imshow(np.asarray(im_crop))
        axs[1].set_yticklabels([])
        axs[1].set_xticklabels([])
        plt.show()
        mask_list_cropped.append(im_crop)
    patient_cropped_mask_list.append(mask_list_cropped)
    for l, mask in enumerate(mask_list_cropped):
        mask.save(os.path.join(patient_mask_path_list_png[x], 'mask_reduced_fov_{:03d}.png'.format(l)), format = "png")

#checking the correct importation in the folder
for path in patient_image_path_list_png:
    image_list = os.listdir(path)
    image_list.sort()
    if (len(image_list) != 100):
        print('Error in Png importation')
for path in patient_mask_path_list_png:
    mask_list = os.listdir(path)
    mask_list.sort()
    if (len(mask_list) != 100):
        print('Error in Png importation')


#saving the numpy imaage
for x, path in enumerate(patient_image_path_list_png):
    image_list = os.listdir(path)
    image_list.sort()
    if (len(image_list) != 100):
        print('Error in Png importation')
    image_list_pil = []
    for image in image_list:
        image = Image.open(path + '/' + image)
        image_list_pil.append(image)
    # resize the image  and the mask
    image_resized = []
    (width, height) = (256, 256)
    for image in image_list_pil:
        im = image.resize((width, height), Image.BILINEAR)
        image_resized.append(im)
    # transforming in np the image
    image_np = []
    for image in image_resized:
        image = np.asarray(image)
        image_np.append(image)
    s = image_np[0].shape
    for l, image in enumerate(image_np):
        if (image.shape != s):
            print('The images have different shape')
        np.save(os.path.join(patient_image_path_list_numpy[x], '{:03d}.npy'.format(l)), np.asarray(image))

for x, path in enumerate(patient_mask_path_list_png):
    mask_list = os.listdir(path)
    mask_list.sort()
    if (len(mask_list) != 100):
        print('Error in Png importation')
    mask_list_pil = []
    for mask in mask_list:
        image = Image.open(path + '/' + mask)
        mask_list_pil.append(image)
    # resize the mask
    mask_resized = []
    (width, height) = (256, 256)
    for mask in mask_list_pil:
        im = mask.resize((width, height), Image.NEAREST)
        mask_resized.append(im)
        print(im.size)
    # transforming in np the image
    mask_np = []
    for mask in mask_resized:
        mask = np.asarray(mask)
        mask_np.append(mask)
    s = mask_np[0].shape
    print(s)
    for l, mask in enumerate(mask_np):
        if (mask.shape != s):
            print('The mask have different shape')
        np.save(os.path.join(patient_mask_path_list_numpy[x], '{:03d}.npy'.format(l)), np.asarray(image))


for path in patient_mask_path_list_numpy:
    mask_list = os.listdir(path)
    mask_list.sort()
    if (len(mask_list) != 100):
        print('Error in Numpy importation')

#renaame the folder
final_dataset_path = os.path.join(os.getcwd(), 'REDUCED FOV - POLIMI DATASET')
print(final_dataset_path)

os.renames(new_dataset_path, final_dataset_path)