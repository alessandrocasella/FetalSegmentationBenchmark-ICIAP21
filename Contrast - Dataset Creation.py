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
Polimi_dataset = os.path.join(os.getcwd(), 'NI - REDUCED FOV - POLIMI DATASET')
print(Polimi_dataset)

#visualize the folders inside my directory
Polimi_dataset_subfolders = [ f.path for f in os.scandir(Polimi_dataset) if f.is_dir() ]
Polimi_dataset_subfolders.sort()
print(Polimi_dataset_subfolders)


Png_subfolders = [ f.path for f in os.scandir(Polimi_dataset_subfolders[1]) if f.is_dir() ]
Png_subfolders.sort()
print(Png_subfolders)

Image_png_subfolders = [f.path for f in os.scandir(Png_subfolders[0]) if f.is_dir() ]
Image_png_subfolders.sort()
print(Image_png_subfolders)
print(' ')
Mask_png_subfolders = [f.path for f in os.scandir(Png_subfolders[1]) if f.is_dir() ]
Mask_png_subfolders.sort()
print(Mask_png_subfolders)

#PATH PATIENTS PNG
path_image_list_new = []    #directories of the image for every patient
for path in Image_png_subfolders:
    path_image_list_new.append(path)
print(path_image_list_new)

path_mask_list_new = []    #directories of the mask for every patient
for path in Mask_png_subfolders:
    path_mask_list_new.append(path)
print(path_mask_list_new)

n_patient = len(path_image_list_new)
print(n_patient)

"""# FOLDERS TO SAVE MY DATASET  """

#creating the dataset subfolder
directory = 'CONTRAST -  NI -  REDUCED FOV - POLIMI DATASET'
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
    image_path = os.path.join(path, 'Image_Contrast')
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

#IMAGES

patient_contrast_image_list = []
for x, path in enumerate(path_image_list_new):
    image_list = os.listdir(path)
    image_list.sort()
    print('Patient{0}:images {1}'.format(x+1, len(image_list)))
    print(path)
    image_list_contrast = []
    path_image_list = [] #lista dei path delle immagini, perchè imread vuole il path
    image_list_pil = []
    for image in image_list:
        dir = path + "/" + image
        #print('PATH IMAGE IS:', dir)
        path_image_list.append(dir)
    for image in path_image_list:
        im = Image.open(image)
        image_list_pil.append(im)
        # resize the image  and the mask
    #image_resized = []
    (width, height) = (256, 256)
    for image in image_list_pil:
        im = image.resize((width, height))
        #image_resized.append(im)
        #image_cv = cv2.imread(image)  # imread vuole il path
        # image_list_cv.append(image_cv)
        # print(image_cv)
        lab = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2LAB)
        #print('prima')
        #cv2.imshow("image",image_cv)
        #cv2.waitKey()
        # cv2_imshow(lab)
        # -----Splitting the LAB image to different channels-------------------------
        l, a, b = cv2.split(lab)
        # cv2_imshow(l)
        # cv2_imshow(a)
        # cv2_imshow(b)

        # -----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        # cv2_imshow(cl)
        # print('cl')

        # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl, a, b))
        # cv2_imshow(limg)
        # print('limg')

        # -----Converting image from LAB Color model to RGB model--------------------
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        #print('dopo')
        #cv2.imshow(final)
        # print('final')
        if(final.shape != (256,256,3)):
            print('error shape')
        image_list_contrast.append(final)
    if(len(image_list_contrast) != 100):
        print('error len')
    patient_contrast_image_list.append(image_list_contrast)

    #saving in PNG
    for l, image in enumerate(image_list_contrast):
        cv2.imwrite(os.path.join(patient_image_path_list_png[x], 'image_contrast_{:03d}.png'.format(l)), image)

# checking the correct importation in the folder
for path in patient_image_path_list_png:
    image_list = os.listdir(path)
    image_list.sort()
    if (len(image_list) != 100):
        print('Error in Png importation')

#saving the numpy image
for patient, image_list in enumerate(patient_contrast_image_list):
    image_np = []
    for image in image_list:
        im = np.asarray(image)
        if(im.shape != (256,256,3)):
            print('error shape')
        image_np.append(im)

    for l, image in enumerate(image_np):
        np.save(os.path.join(patient_image_path_list_numpy[patient], '{:03d}.npy'.format(l)), np.asarray(image))

#checking the correct importation in the folder
for path in patient_image_path_list_numpy:
    image_list = os.listdir(path)
    image_list.sort()
    if (len(image_list) != 100):
        print('Error in Numpy importation')





"""## MASK"""
#loading
for patient, path in enumerate(path_mask_list_new):
  p = os.listdir(path)
  if (len(p) != 100):
    print('Error Png in the Patient{}'.format(x+1))
    print(len(p))

#loading my saved data
for patient, path in enumerate(path_mask_list_new):
    max = 0
    shape = 0
    type_ = 0
    p = os.listdir(path)
    p.sort()
    (width, height) = (256, 256)
    mask_np = []
    if (len(p) > 0):
      for l, mask in enumerate(p):
        mask = Image.open(os.path.join(path, mask))
        if (mask.size != (256, 256)):
            print(mask.size)
            mask = mask.resize((width, height))
        mask.save(os.path.join(patient_mask_path_list_png[patient], '{:03d}.png'.format(l)))
        im = np.asarray(mask)
        if (im.shape != (256, 256)):
          print('error shape')
        mask_np.append(im)
    max = np.max(mask_np[0])
    shape = mask_np[0].shape
    type_ = mask_np[0].dtype
    for l, i in enumerate(mask_np):
        if (np.max(i) != max):
         print('different np max')
         print(np.max(i))
        if (i.shape != shape):
        print('different shape')
        if (i.dtype != type_):
        print('differnt type')
        np.save(os.path.join(patient_mask_path_list_numpy[patient], '{:03d}.npy'.format(l)), np.asarray(i))
    a = len(p)
    if (a > 0):
        print('There are {0} masks in the patient {1} folder'.format(a,patient+1))

#checking the correct importation in the folder
for path in patient_mask_path_list_numpy:
    print('Cheching numpy')
    mask_list = os.listdir(path)
    mask_list.sort()
    if (len(mask_list) != 100):
        print('Error in Numpy importation')

for path in patient_mask_path_list_png:
    print('Cheching png')
    mask_list = os.listdir(path)
    mask_list.sort()
    if (len(mask_list) != 100):
        print('Error in PNG importation')