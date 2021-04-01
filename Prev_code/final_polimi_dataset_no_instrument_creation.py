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
Reduced_fov_Polimi_dataset = os.path.join(os.getcwd(), 'REDUCED FOV - POLIMI DATASET')
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

Image_png_subfolders = [f.path for f in os.scandir(Png_subfolders[0]) if f.is_dir() ]
Image_png_subfolders.sort()
print(Image_png_subfolders)
print(' ')
Mask_png_subfolders = [f.path for f in os.scandir(Png_subfolders[1]) if f.is_dir() ]
Mask_png_subfolders.sort()
print(Mask_png_subfolders)


path_image_list_new = []    #directories of the image for every patient
for path in Image_png_subfolders:
    path_image_list_new.append(path)
print(path_image_list_new)

path_mask_list_new = []    #directories of the image for every patient
for path in Mask_png_subfolders:
    path_mask_list_new.append(path)
print(path_mask_list_new)

n_patient = len(path_mask_list_new)
print(n_patient)

path_image_list = path_image_list_new
path_mask_list = path_mask_list_new
print(path_image_list)
print(path_mask_list)

n_patient = len(path_image_list)
print(n_patient)

if (len(path_mask_list) != n_patient):
  print('Error in the directories')

"""# FOLDERS TO SAVE MY DATASET  """

#creating the dataset subfolder 
directory = 'NI - REDUCED FOV - POLIMI DATASET'
new_dataset_path = os.path.join(os.getcwd(), directory)
try:
  os.mkdir(new_dataset_path)
except:
  pass

numpy_path = os.path.join(new_dataset_path, 'Numpy')
png_path = os.path.join(new_dataset_path, 'PNG')
type_path_list = [numpy_path, png_path]
path_mask_list_new_numpy = []
path_image_list_new_numpy = []
path_mask_list_new_png = []
path_image_list_new_png = []
for l, path in enumerate(type_path_list):
  try:
    os.mkdir(path)
  except:
    pass
  mask_path = os.path.join(path, 'Mask')
  image_path = os.path.join(path, 'Image')
  try:
    os.mkdir(mask_path)
    os.mkdir(image_path)
  except:
    pass
  x = 1
  while (x <= n_patient):
    patient_mask_path = os.path.join(mask_path, 'patient_{:02d}'.format(x))
    try:
      os.mkdir(patient_mask_path)
    except:
      pass
    if (l == 0):
      path_mask_list_new_numpy.append(patient_mask_path)
    else:
      path_mask_list_new_png.append(patient_mask_path)

    patient_image_path = os.path.join(image_path, 'patient_{:02d}'.format(x))
    try:
      os.mkdir(patient_image_path)
    except:
      pass
    if (l==0):
      path_image_list_new_numpy.append(patient_image_path)
    else:
      path_image_list_new_png.append(patient_image_path)
    x = x+1
  print('{} Patient folders were created'.format(x))

print(len(path_mask_list_new_numpy), path_mask_list_new_numpy)
print(len(path_image_list_new_numpy), path_image_list_new_numpy)
print(len(path_mask_list_new_png), path_mask_list_new_png)
print(len(path_image_list_new_png), path_image_list_new_png)

new_dataset_path_subfolders = [f.path for f in os.scandir(new_dataset_path) if f.is_dir()]
new_dataset_path_subfolders.sort()
print(new_dataset_path_subfolders)

PNG = [f.path for f in os.scandir(new_dataset_path_subfolders[1]) if f.is_dir()]
PNG.sort()
Numpy = [f.path for f in os.scandir(new_dataset_path_subfolders[0]) if f.is_dir()]
Numpy.sort()

print(' ')
IMAGE_PNG = [f.path for f in os.scandir(PNG[0]) if f.is_dir()]
IMAGE_PNG.sort()
print(IMAGE_PNG)
MASK_PNG = [f.path for f in os.scandir(PNG[1]) if f.is_dir()]
MASK_PNG.sort()
print(MASK_PNG)
print(' ')
IMAGE_numpy = [f.path for f in os.scandir(Numpy[0]) if f.is_dir()]
IMAGE_numpy.sort()
print(IMAGE_numpy)
MASK_numpy = [f.path for f in os.scandir(Numpy[1]) if f.is_dir()]
MASK_numpy.sort()
print(MASK_numpy)


"""# LOADING ALL PATIENT DATA AND SAVING THEM IN CROSS VALIDATION FOLDER"""

for i in range(n_patient): 
    d = 0
    print(' ')
    print('Loading the image and mask in the Patient{0}, resizing them, transforming them in numpy image, saving them in the new dataset subolders'.format(i+1))

    #import all the image directory from the list images in the list imaged
    imaged = []                                #create the list with the png image 
    image_list = []                            #create the list with the PIL image
    path = path_image_list[i]                  #taking the right path
    image_imaged = os.listdir(path)            #create the list with the png image in the video0* in the iteration *
    image_imaged.sort()
    print(image_imaged) 

    for image in image_imaged:
      path_new = path + "/" + image
      o = Image.open(path_new)
      image_list.append(o)
      imaged.append(image)
    
    print('The path list is:', path)    
    print('The list of my opened image.png is:', imaged)
    print('The number of images in the path folder is {0} and the lenght in the list of my opened image.png is {1}'.format(len(image_list), len(imaged)))

    #import all the mask directory from the list masks in the list masked  
    masked = []                     #create the list with the png image 
    mask_list = []                  #create the list with the PIL image
    path = path_mask_list[i]        #taking the right path
    image_masked = os.listdir(path) #create the list with the png image in the video0* in the iteration *
    image_masked.sort()
    print(image_masked)  
    
    for mask in image_masked:
      path_new = path + "/" + mask
      mask_list.append(Image.open(path_new).convert('L'))
      masked.append(mask)

    print('The path list is:', path)
    print('The list of my opened mask.png is:', masked)
    print('The number of masks in the path folder is {0} and the lenght in the list of my opened image.png is {1}'.format(len(mask_list), len(masked)))

    #resize the image  and the mask 
    image_resized = []
    mask_resized = []
    (width, height) = (256,256)
    for image in image_list:
        print(image.size)
        im = image.resize((width, height), Image.NEAREST)
        image_resized.append(im)
    for image in mask_list:
        im = image.resize((width, height), Image.NEAREST)
        mask_resized.append(im)

    #transforming in np the image
    image_np = []
    for x in image_resized: 
      image = np.asarray(x)
      image_np.append(image)

    #try:
    print('The max value in the numpy image is {0}'.format(np.max(image_np)))

    s = image_np[0].shape
    for image in image_np:
      if (image.shape != s):
        print('The images have different shape')
    print('The shape of the images is {0}'.format(s))
    print('The number of the image in numpy array form is  {0}'.format(len(image_np)))

    #transforming in np the mask
    mask_np = []
    mask_instr_np = []
    for x,m in enumerate(mask_resized):
      print('The shape of the images is {0}'.format(m.size))
      mask_ = np.asarray(m).copy()
      mask_instr_np.append(mask_)
      mask = mask_
      print(np.max(m))
      mask[mask == 19] = 0
      mask[mask == 25] = 34
      if (np.max(mask) == 0):
        print('Deleting one mask. The number is {0}'.format(x))
        d = x+1
      else:
        print(np.max(mask))
        mask_np.append(mask)

    if (d > 0):
      d_new = d-1
      new_image_np = []
      for k, image in enumerate(image_np):
          if (k == d_new):
            print('Delating one image')
          else:
            new_image_np.append(image)
      print('The number of the image in numpy array form is  {0}'.format(len(new_image_np)))
      image_np = new_image_np

    print('The max value in the numpy mask is {0}'.format(np.max(mask_np)))
    p = mask_np[0].shape
    s = np.squeeze(np.stack((mask_np[0], )*3,axis = -1)).shape
    t = np.squeeze(np.stack((mask_instr_np[0], )*3,axis = -1)).shape
    for mask in mask_np:
      if (mask.shape != p):
        print('The mask have different shape')
    print('The shape of the masks is {0}'.format(s))
    print('The shape of the masks with the instrument is {0}'.format(t))

    print('The number of the masks in numpy array form is {0}'.format(len(mask_np)))

    #saving the image data in Image
    print('Saving the numpy images in the Patient{0} in the new folder'.format(i+1))
    path = path_image_list_new_numpy[i]
    for x, image in enumerate(image_np):
      np.save(os.path.join(path,'{:03d}.npy'.format(x)), np.asarray(image))
      im = Image.fromarray(image)
      im.save(os.path.join(path_image_list_new_png[i], '{:03d}.png'.format(x)))

    #saving the mask data in Mask
    print('Saving the numpy masks in the Patient{0} in the new folder'.format(i+1))
    path = path_mask_list_new_numpy[i]
    for x, mask in enumerate(mask_np):
      mask_3d = np.expand_dims(mask, 2)
      np.save(os.path.join(path,'{:03d}.npy'.format(x)), np.asarray(mask_3d))
      #mask = mask[:, :, 0]
      im = Image.fromarray(mask, mode='L')
      im.save(os.path.join(path_mask_list_new_png[i], '{:03d}.png'.format(x)))

    fig, axs = plt.subplots(3, 3,  gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize = (10,10))
    axs = axs.ravel()

    axs[0].imshow(image_np[34])
    axs[1].imshow(image_np[35])
    axs[2].imshow(image_np[36])
    axs[3].imshow(np.squeeze(np.stack((mask_instr_np[34], )*3,axis = -1)))
    axs[4].imshow(np.squeeze(np.stack((mask_instr_np[35], )*3,axis = -1)))
    axs[5].imshow(np.squeeze(np.stack((mask_instr_np[36], )*3,axis = -1)))
    axs[6].imshow(np.squeeze(np.stack((mask_np[34], )*3,axis = -1)))
    axs[7].imshow(np.squeeze(np.stack((mask_np[35], )*3,axis = -1)))
    axs[8].imshow(np.squeeze(np.stack((mask_np[36], )*3,axis = -1)))
    plt.show()


"""#CHECKIN THE PROCESS

## IMAGE
"""

#loading my saved data
for x, path in enumerate(path_image_list_new_png):
  p = os.listdir(path)
  if (len(p) != 100):
    print('Error Png in the Patient{}'.format(x+1))

#loading my saved data
for x, path in enumerate(path_image_list_new_numpy):
  shape = 0
  type_ = 0
  count = 0
  p = os.listdir(path) 
  if (len(p) > 0):
    shape = (np.load(os.path.join(path, p[0]))).shape
    type_ = (np.load(os.path.join(path, p[0]))).dtype
    for image in p:
      i = np.load(os.path.join(path, image))
      if (i.shape != shape):
        print('different shape')
      if (i.dtype != type_):
        print('differnt type')
  a = len(p)
  if (a > 0):
    print('There are {0} images in the patient {1} folder'.format(a,x+1))
  #print(p)
  '''if (shape != 0):
    print('Patient {0}: '.format(x+1), shape, type_)'''

#visualizing my saved data
for x, path in enumerate(path_image_list_new_numpy):
  image_list = []
  p = os.listdir(path) 
  p.sort()
  for image in p:
    path_new = path + '/' + image
    p_new = np.load(path_new)
    image_list.append(p_new)
  a = len(p)
  print('There are {0} images in the patient {1} folder'.format(a,x+1))
  b = int (len(image_list))
  k = 0
  while (k<b):
    fig, axs = plt.subplots(2, 5,  gridspec_kw={'hspace': 0.75, 'wspace': 0.25})
    axs = axs.ravel()
    axs[0].imshow(image_list[k])
    axs[1].imshow(image_list[k+1])
    axs[2].imshow(image_list[k+2])
    axs[3].imshow(image_list[k+3])
    axs[4].imshow(image_list[k+4])
    axs[5].imshow(image_list[k+5])
    axs[6].imshow(image_list[k+6])
    axs[7].imshow(image_list[k+7])
    axs[8].imshow(image_list[k+8])
    axs[9].imshow(image_list[k+9])
    k = k + 10
  plt.show()

"""## MASK"""

#loading my saved data
for x, path in enumerate(path_mask_list_new_png):
  p = os.listdir(path)
  if (len(p) != 100):
    print('Error Png in the Patient{}'.format(x+1))
    print(len(p))

#loading my saved data
for x, path in enumerate(path_mask_list_new_numpy):
  max = 0
  shape = 0
  type_ = 0
  p = os.listdir(path) 
  if (len(p) > 0):
    max = np.max(np.load(os.path.join(path, p[0])))
    print(max)
    shape = (np.load(os.path.join(path, p[0]))).shape
    type_ = (np.load(os.path.join(path, p[0]))).dtype
    for image in p:
      i = np.load(os.path.join(path, image))
      if (np.max(i) != max):
         print('different np max')
         print(image)
         print(np.max(i))
      if (i.shape != shape):
        print('different shape')
      if (i.dtype != type_):
        print('differnt type')
  a = len(p)
  if (a > 0):
      print('There are {0} masks in the patient {1} folder'.format(a,x+1))
  #print(p)
  '''if (shape != 0):
    print('Patient {0}: '.format(x+1), max, shape, type_)'''

#visualizing my saved data
for x, path in enumerate(path_mask_list_new_numpy):
  image_list = []
  p = os.listdir(path) 
  p.sort()
  for image in p:
    path_new = path + '/' + image
    p_new = np.load(path_new)
    p_new = p_new[:,:,0]
    image_list.append(p_new)
  a = len(p)
  print('There are {0} images in the patient {1} folder'.format(a,x+1))
  b = int (len(image_list))
  k = 0
  while (k<b):
    fig, axs = plt.subplots(2, 5,  gridspec_kw={'hspace': 0.75, 'wspace': 0.25})
    axs = axs.ravel()
    axs[0].imshow(image_list[k])
    axs[1].imshow(image_list[k+1])
    axs[2].imshow(image_list[k+2])
    axs[3].imshow(image_list[k+3])
    axs[4].imshow(image_list[k+4])
    axs[5].imshow(image_list[k+5])
    axs[6].imshow(image_list[k+6])
    axs[7].imshow(image_list[k+7])
    axs[8].imshow(image_list[k+8])
    axs[9].imshow(image_list[k+9])
    k = k + 10
  plt.show()
