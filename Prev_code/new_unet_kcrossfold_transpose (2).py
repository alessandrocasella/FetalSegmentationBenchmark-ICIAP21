import os
from os import listdir
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K  # non viene mai usata???
from tensorflow.keras.layers import *  # need it for Input name in the UNET
from tensorflow.keras.models import Model, load_model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras.utils import plot_model
from keras.utils import model_to_dot
# from IPython.display import SVG
import h5py

cross = [f.path for f in os.scandir(os.path.join(os.getcwd(), 'Try/NEW DATASET NUMPY 9paz')) if f.is_dir()]
print(cross)
cross = [f.path for f in os.scandir(cross[0]) if f.is_dir()]
print(cross)
cross.sort()

directory = 'Graphs Transpose'
graph_path = os.path.join(os.path.join(os.getcwd(), 'Try/NEW DATASET NUMPY 9paz'), directory)
try:
    os.mkdir(graph_path)
except:
    pass

print(graph_path)

mask_path = cross[1]
image_path = cross[0]
predicted_path = cross[2]
print(mask_path)
print(image_path)
print(predicted_path)


mask_subfolders = [f.path for f in os.scandir(mask_path) if f.is_dir()]
mask_subfolders.sort()
print(mask_subfolders)

path_mask_list_new = []  # directories of the mask for every patient
for path in mask_subfolders:
    path_mask_list_new.append(path)
print(path_mask_list_new)

image_subfolders = [f.path for f in os.scandir(image_path) if f.is_dir()]
image_subfolders.sort()
print(image_subfolders)

path_image_list_new = []  # directories of the image for every patient
for path in image_subfolders:
    path_image_list_new.append(path)
print(path_image_list_new)

n_patient = len(path_image_list_new)
print(n_patient)

predicted_subfolders = [f.path for f in os.scandir(predicted_path) if f.is_dir()]
predicted_subfolders.sort()
print(predicted_subfolders)

path_predicted_list_new = []  # directories of the prediction for every patient
for path in predicted_subfolders:
    path_predicted_list_new.append(path)
print(path_predicted_list_new)

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
# if you don't want to do data augmentation, set aug empty

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


# to obtain images distorted

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, list_np, list_label_np, batch_size, nChannelData, nChannelLabel, shuffle, aug):
        # print('__init__')
        self.batch_size = batch_size
        self.nChannelData = nChannelData
        self.nChannelLabel = nChannelLabel
        self.shuffle = shuffle
        self.list_np = list_np
        self.list_label_np = list_label_np
        # print('The lenght of the list np is {0}'.format(len(self.list_np)))
        self.aug = aug
        self.on_epoch_end()

    def __len__(self):
        # print('__len__')
        i = int(np.floor(len(self.list_np) / self.batch_size))
        # print('The lenght of list_np is {0}'.format(len(self.list_np)))
        # print('The division is {0}'.format(len(self.list_np) / self.batch_size))
        # print('The divison after the np.flor is {0}'.format(i))
        # print(i)
        return i

    def __getitem__(self, index):
        # print('__getitem__')
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        file_list_temp = [self.list_np[k] for k in indexes]
        # print('The lenght of the file temp is {0}'.format(len(file_list_temp)))
        X, y_new = self.__data_generation(file_list_temp)
        # print('The lenght of X is {0}'.format(len(X)))
        # print('The lenght of y_new is {0}'.format(len(y_new)))
        return X, y_new

    def on_epoch_end(self):
        # print('one epoch end')
        self.indexes = np.arange(len(self.list_np))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_list_temp):
        # print('data generation')
        y_new = np.empty([self.batch_size, 256, 256, 1], dtype=np.float32)
        x_new = np.empty([self.batch_size, 256, 256, 3], dtype=np.float32)
        max_z = 0
        max_y = 0

        for p, ID in enumerate(file_list_temp):

            transformInstance = self.aug.get_random_transform((256, 256, 3))
            X = np.array(ID / 255.).astype('float32')

            Z = X

            y_old = self.list_label_np[p]
            y = np.array(y_old / np.max(y_old)).astype('float32')
            y = np.asarray(np.dstack((y, y, y)), dtype=np.float32)

            Z[:, :, :] = self.aug.apply_transform(X[:, :, :], transformInstance)
            if (np.max(Z) != 1.0):
                max_z = max_z + 1
                # print('The max in Z is not 1 but is {0}'.format(np.max(Z)))
            # print('The shape of Z is {0}'.format(Z.shape))
            x_new[p, :, :, :] = Z[:, :, :]
            # print('The shape of x_new is {0}'.format(x_new.shape))
            y[:, :, :] = self.aug.apply_transform(y[:, :, :], transformInstance)
            # print('The shape of y is {0}'.format(y.shape))
            if (np.max(y) != 1.0):
                max_y = max_y + 1
                # print('The max in y is not 1 but {0}'.format(np.max(y)))
            y_new[p, :, :, 0] = y[:, :, 0]
            # print('The shape of y_new is {0}'.format(y_new.shape))
        '''if (max_z != 0):
            print('The max in Z is not 1 for {0} value'.format(max_z))'''
        if (max_y != 0):
            print('The max in y is not 1 for {0} value'.format(max_y))
        return x_new, y_new


"""## defining the metrics"""


# defining the metric Structural similarity index SSIM
def ssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))


# defining the SSIM loss
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))


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
learning_rate = 0.001  # @param {type:"number"}
batchSize = 32  # @param {type:"number"}
# earlystop_patience = 50 #@param {type:"number"}
# rule of thumb to make it 10% of number of epoch.

# GET THE UNET AND DISPLAY MY MODEL
MyModel = myUnet()  # creo istanza dell'oggetto
myunet = MyModel.get_unet()  # questa funzione, data l'istanza dell'oggetto, associa un modello a questa variabile
myunet.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='binary_crossentropy', metrics=[ssim, 'acc', dice_coeff])
myunet.summary()

plot_model(myunet, to_file='myunet_transpose_plot.png', show_shapes=False, show_layer_names=False, rankdir='LR')

path = os.path.join(os.getcwd(), 'myunet_transpose_plot.png')
im = Image.open(path)

im.save(graph_path + '/myunet_transpose_plot.png')

"""##Creating the Folds"""

size_test = 3
size_train = 4
size_val = 2

n_test = size_test
K_fold = int(n_patient / n_test)
print(K_fold)

K_test = []
K_train = []
K_val = []
for train, test in kfold.split(path_image_list_new):
    print('    ')
    # print(train)
    print(test)
    np.random.seed(2)
    np.random.shuffle(train)
    # print(train)
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

# checking the correct selection of the patient directories
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

# defining the testing set for each folds
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
    # print(t)
    for t_x in t:
        # print(t_x)
        path_image = path_image_list_new[t_x]
        p_i = os.listdir(path_image)
        patient_test_image.append(t_x + 1)
        path_mask = path_mask_list_new[t_x]
        p_m = os.listdir(path_mask)
        patient_test_mask.append(t_x + 1)
        for x, image in enumerate(p_i):
            if (x < 45):
                test_image = np.load(path_image + '/' + image)  # the images are in uint8 and np.max = 255
                if (test_image.shape != (256, 256, 3)):
                    print('error')
                test_image_np.append(test_image)
                b = x + 1
        print('Test Image: Loading patient{0} and it contain {1} files'.format(t_x + 1, b))
        for x, mask in enumerate(p_m):
            if (x < 45):
                test_mask = np.load(
                    path_mask + '/' + mask)  # the masks are uint8 with different np.max and have shape (256, 256, 1)
                test_mask = np.asarray(test_mask / np.max(test_mask)).astype(
                    'float32')  # binarizing the masks and transforming them into float32
                test_mask_np.append(test_mask)
                b = x + 1
        # print(test_mask_np[0].shape)
        # print(test_mask_np[0].dtype)
        # print(np.max(test_mask_np[0]))
        print('Test Mask: Loading patient{0} and it contain {1} files'.format(t_x + 1, b))
    print('           The testing set image is lenght {} and it contains the patient: '.format(len(test_image_np)),
          patient_test_image)
    print('           The testing set mask is lenght {} and it contains the patient: '.format(len(test_mask_np)),
          patient_test_mask)
    K_fold_test_image.append(test_image_np)
    K_fold_test_mask.append(test_mask_np)
    print('   ')

# checking the creation of the folds
print(len(K_fold_test_image))
for x in K_fold_test_image:
    print(len(x))
print(len(K_fold_test_mask))
for x in K_fold_test_mask:
    print(len(x))

## TRAIN

# checking the correct selection of the patient directories
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

# defining the training set for each folds
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
        patient_train_image.append(tr_x + 1)
        path_mask = path_mask_list_new[tr_x]
        p_m = os.listdir(path_mask)
        patient_train_mask.append(tr_x + 1)
        for x, image in enumerate(p_i):
            if (x < 45):
                train_image = np.load(path_image + '/' + image)
                train_image_np.append(train_image)
                b = x + 1
        print('Train Image: Loading patient{0} and it contain {1} files'.format(tr_x + 1, b))
        for x, mask in enumerate(p_m):
            if (x < 45):
                train_mask = np.load(path_mask + '/' + mask)
                train_mask_np.append(train_mask)
                b = x + 1
        print('Train Mask: Loading patient{0} and it contain {1} files'.format(tr_x + 1, b))
    print('           The training set image is lenght {} and it contains the patient: '.format(len(train_image_np)),
          patient_train_image)
    print('           The training set mask is lenght {} and it contains the patient: '.format(len(train_mask_np)),
          patient_train_mask)
    K_fold_train_image.append(train_image_np)
    K_fold_train_mask.append(train_mask_np)
    print('   ')

# checking the creation of the folds
print(len(K_fold_train_image))
for x in K_fold_train_image:
    print(len(x))
print(len(K_fold_train_mask))
for x in K_fold_train_mask:
    print(len(x))

## VALIDATION

# checking the correct selection of the patient directories
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

# defining the validation set for each folds
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
    # print(tr)

    for tr_x in tr:
        path_image = path_image_list_new[tr_x]
        p_i = os.listdir(path_image)
        patient_val_image.append(tr_x + 1)
        path_mask = path_mask_list_new[tr_x]
        p_m = os.listdir(path_mask)
        patient_val_mask.append(tr_x + 1)
        for x, image in enumerate(p_i):
            if (x < 45):
                val_image = np.load(path_image + '/' + image)
                val_image_np.append(val_image)
                b = x + 1
        print('Val Image: Loading patient{0} and it contain {1} files'.format(tr_x + 1, b))
        for x, mask in enumerate(p_m):
            if (x < 45):
                val_mask = np.load(path_mask + '/' + mask)
                val_mask_np.append(val_mask)
                b = x + 1
        print('Val Mask: Loading patient{0} and it contain {1} files'.format(tr_x + 1, b))
    print('           The validation set image is lenght {} and it contains the patient: '.format(len(val_image_np)),
          patient_val_image)
    print('           The validation set mask is lenght {} and it contains the patient: '.format(len(val_mask_np)),
          patient_val_image)
    K_fold_val_image.append(val_image_np)
    K_fold_val_mask.append(val_mask_np)
    print('   ')

# checking the creation of the folds
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
K_test_predicted_np = []

#FOLD 1
k = 0
print('FOLD {}'.format(k + 1))
training_image = K_fold_train_image[k]
training_mask = K_fold_train_mask[k]
validation_image = K_fold_val_image[k]
validation_mask = K_fold_val_mask[k]
for l in range(0, len(training_image)):
    shape = training_image[l].shape
    if (shape != (256,256,3)):
        print(shape)
    m_shape = training_mask[l].shape
    if (m_shape != (256, 256, 1)):
        print(m_shape)
    '''fig, axs = plt.subplots(1, 2, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(10, 10))
    axs = axs.ravel()
    fig.suptitle('FOLD 1 - Training: number {}'.format(l), fontsize=20)
    axs[0].imshow(training_image[l])
    axs[0].set_title('Image', fontsize=15, loc='center')
    axs[0].set_yticklabels([])
    axs[0].set_xticklabels([])
    axs[1].imshow(np.squeeze(np.stack((training_mask[l],) * 3, axis=-1)))
    axs[1].set_title('Ground Truth', fontsize=15, loc='center')
    axs[1].set_yticklabels([])
    axs[1].set_xticklabels([])
    plt.show()'''

for l in range(0, len(validation_image)):
    shape = validation_image[l].shape
    if (shape != (256, 256, 3)):
        print(shape)
    m_shape = validation_mask[l].shape
    if (m_shape != (256, 256, 1)):
        print(m_shape)
    '''fig, axs = plt.subplots(1, 2, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(10, 10))
    axs = axs.ravel()
    fig.suptitle('FOLD 1 - Validation: number {}'.format(l), fontsize=20)
    axs[0].imshow(validation_image[l])
    axs[0].set_title('Image', fontsize=15, loc='center')
    axs[0].set_yticklabels([])
    axs[0].set_xticklabels([])
    axs[1].imshow(np.squeeze(np.stack((validation_mask[l],) * 3, axis=-1)))
    axs[1].set_title('Ground Truth', fontsize=15, loc='center')
    axs[1].set_yticklabels([])
    axs[1].set_xticklabels([])
    plt.show()'''
print('The image training set is lenght:', len(training_image))
print('The mask training set is lenght:', len(training_mask))
print('The image validation set is lenght:', len(validation_image))
print('The mask validation set is lenght:', len(validation_mask))

MyModel1 = myUnet()  # creo istanza dell'oggetto
myunet1 = MyModel1.get_unet()

myunet1.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='binary_crossentropy', metrics=[ssim, 'acc', dice_coeff])

#earlystopper = EarlyStopping(patience=earlystop_patience, verbose=1)
checkpointer = ModelCheckpoint(os.path.join(os.getcwd(), 'NEW DATASET NUMPY 9paz/NEW DATASET NUMPY 9paz/new_model_unet_transpose_checkpoint_{:02d}_fold.h5'.format(k+1)), verbose=1, save_best_only=True, monitor='val_loss', mode = 'min')

training_generator = DataGenerator(training_image, training_mask, batchSize, 3, 1, True, aug)
validation_data = DataGenerator(validation_image, validation_mask, batchSize, 3, 1, True, aug)

results_1 = myunet1.fit(training_generator, validation_data=validation_data, batch_size=batchSize, epochs=2000, callbacks=[checkpointer])
ssim_history = results_1.history["ssim"]
val_ssim_history = results_1.history["val_ssim"]
acc_history = results_1.history["acc"]
val_acc_history = results_1.history["val_acc"]
dice_history = results_1.history["dice_coeff"]
val_dice_history = results_1.history["val_dice_coeff"]
K_ssim_history.append(ssim_history)
K_val_ssim_history.append(val_ssim_history)
K_acc_history.append(acc_history)
K_val_acc_history.append(val_acc_history)
K_dice_history.append(dice_history)
K_val_dice_history.append(val_dice_history)
K_path_model.append(os.path.join(os.getcwd(),'NEW DATASET NUMPY 9paz/NEW DATASET NUMPY 9paz/new1_model_unet_transpose_checkpoint_{:02d}_fold.h5'.format(k + 1)))
# saving the metrics' value in a dataset
with h5py.File(os.path.join(os.getcwd(), 'NEW DATASET NUMPY 9paz/NEW DATASET NUMPY 9paz/new1_FOLD{0}_Metrics_history_transpose.hdf5'.format(k + 1)), 'w') as f:
    f.create_dataset('ssim', data=ssim_history)
    f.create_dataset('val_ssim', data=val_ssim_history)
    f.create_dataset('acc', data=acc_history)
    f.create_dataset('val_acc', data=val_acc_history)
    f.create_dataset('dice', data=dice_history)
    f.create_dataset('val_dice', data=val_dice_history)
    f.close


predicted = []
test_image_np = K_fold_test_image[k]
print(len(test_image_np))
patient_test_image = []
patient_test_mask = []
patient_test_image.append(K_test[k])
patient_test_mask.append(K_test[k])
myunet = load_model(os.path.join(os.getcwd(),'NEW DATASET NUMPY 9paz/NEW DATASET NUMPY 9paz/new_model_unet_transpose_checkpoint_{:02d}_fold.h5'.format(k+1)), compile=False)
for image in test_image_np:
    image = np.array(image / 255.).astype('float32')  # trasforming the image in float 32 with np.max 0 < 1
    prediction = myunet1.predict(np.expand_dims(image, 0))
    if (prediction.shape != (1, 256, 256, 1)):
        print('error shape 1')
    if (np.max(prediction) < 0.5):
        print('error MAX! The max is {0}'.format(np.max(prediction)))
    predicted.append(prediction)
    plt.imshow(prediction[0])
    plt.show()
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
    #print(np.min(a))
    pred_thr.append(a.astype('uint8'))  # image in float 32 with np.max =  1
print('Ending the threshold')


pred_int = pred_thr
print(len(pred_int))
print(np.max(pred_int[0]))

K_test_predicted_np.append(pred_int)

print('           The testing set image is lenght {} and it contains the patient: '.format(len(test_image_np)), patient_test_image)
print('           The prediction set is lenght {} and it contains the patient: '.format(len(test_mask_np)), patient_test_mask)
print('   ')

print(len(K_test_predicted_np))
print(len(K_test_predicted_np[k]))

test_image = K_fold_test_image[k]
ground_truth = K_fold_test_mask[k]
predicted = K_test_predicted_np[k]
print('  ')
print('Visualizing FOLD{} prediction'.format(k + 1))

print(len(test_image))
for l in range(0, len(test_image)):
    fig, axs = plt.subplots(1, 4, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(10, 10))
    fig.suptitle('FOLD 1 - Testinng: number {}'.format(l), fontsize=20)
    axs = axs.ravel()
    axs[0].imshow(test_image[l])
    axs[0].set_title('Image', fontsize=15, loc='center')
    axs[0].set_yticklabels([])
    axs[0].set_xticklabels([])
    axs[1].imshow(np.squeeze(np.stack((ground_truth[l],) * 3, axis=-1)))
    axs[1].set_title('Ground Truth', fontsize=15, loc='center')
    axs[1].set_yticklabels([])
    axs[1].set_xticklabels([])
    axs[2].imshow(np.squeeze(np.stack((pred[l],) * 3, axis=-1)))
    axs[2].set_title('Predicted', fontsize=15, loc='center')
    axs[2].set_yticklabels([])
    axs[2].set_xticklabels([])
    axs[3].imshow(np.squeeze(np.stack((predicted[l],) * 3, axis=-1)))
    axs[3].set_title('Predicted THR', fontsize=15, loc='center')
    axs[3].set_yticklabels([])
    axs[3].set_xticklabels([])
    plt.show()
    # fig.savefig(os.path.join(figure_path, 'ImageVSMasks_{0}'.format(i-1)))

#FOLD 2
k = 1
print('FOLD {}'.format(k + 1))
training_image = K_fold_train_image[k]
training_mask = K_fold_train_mask[k]
validation_image = K_fold_val_image[k]
validation_mask = K_fold_val_mask[k]
print('The image training set is lenght:', len(training_image))
print('The mask training set is lenght:', len(training_mask))
print('The image validation set is lenght:', len(validation_image))
print('The mask validation set is lenght:', len(validation_mask))

MyModel2 = myUnet()  # creo istanza dell'oggetto
myunet2 = MyModel2.get_unet()

myunet2.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss=dice_loss, metrics=[ssim, 'acc', dice_coeff])

#earlystopper = EarlyStopping(patience=earlystop_patience, verbose=1)
checkpointer = ModelCheckpoint(os.path.join(os.getcwd(), 'NEW DATASET NUMPY 9paz/NEW DATASET NUMPY 9paz/new_model_unet_transpose_checkpoint_{:02d}_fold.h5'.format(k+1)), verbose=1, save_best_only=True, monitor='val_loss', mode = 'min')

training_generator = DataGenerator(training_image, training_mask, batchSize, 3, 1, True, aug)
validation_data = DataGenerator(validation_image, validation_mask, batchSize, 3, 1, True, aug)

results_2 = myunet2.fit(training_generator, validation_data=validation_data, batch_size=batchSize, epochs=2000, callbacks=[checkpointer])
ssim_history = results_2.history["ssim"]
val_ssim_history = results_2.history["val_ssim"]
acc_history = results_2.history["acc"]
val_acc_history = results_2.history["val_acc"]
dice_history = results_2.history["dice_coeff"]
val_dice_history = results_2.history["val_dice_coeff"]
K_ssim_history.append(ssim_history)
K_val_ssim_history.append(val_ssim_history)
K_acc_history.append(acc_history)
K_val_acc_history.append(val_acc_history)
K_dice_history.append(dice_history)
K_val_dice_history.append(val_dice_history)
K_path_model.append(os.path.join(os.getcwd(),'NEW DATASET NUMPY 9paz/NEW DATASET NUMPY 9paz/new_model_unet_transpose_checkpoint_{:02d}_fold.h5'.format(k + 1)))
# saving the metrics' value in a dataset
with h5py.File(os.path.join(os.getcwd(), 'NEW DATASET NUMPY 9paz/NEW DATASET NUMPY 9paz/new_FOLD{0}_Metrics_history_transpose.hdf5'.format(k + 1)), 'w') as f:
    f.create_dataset('ssim', data=ssim_history)
    f.create_dataset('val_ssim', data=val_ssim_history)
    f.create_dataset('acc', data=acc_history)
    f.create_dataset('val_acc', data=val_acc_history)
    f.create_dataset('dice', data=dice_history)
    f.create_dataset('val_dice', data=val_dice_history)
    f.close


predicted = []
test_image_np = K_fold_test_image[k]
print(len(test_image_np))
patient_test_image = []
patient_test_mask = []
patient_test_image.append(K_test[k])
patient_test_mask.append(K_test[k])

for image in test_image_np:
    image = np.array(image / 255.).astype('float32')  # trasforming the image in float 32 with np.max 0 < 1
    prediction = myunet2.predict(np.expand_dims(image, 0))
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
    #print(np.min(a))
    pred_thr.append(a.astype('uint8'))  # image in float 32 with np.max =  1
print('Ending the threshold')


pred_int = pred_thr
print(len(pred_int))
print(np.max(pred_int[0]))

K_test_predicted_np.append(pred_int)

print('           The testing set image is lenght {} and it contains the patient: '.format(len(test_image_np)), patient_test_image)
print('           The prediction set is lenght {} and it contains the patient: '.format(len(test_mask_np)), patient_test_mask)
print('   ')

print(len(K_test_predicted_np))
print(len(K_test_predicted_np[k]))

test_image = K_fold_test_image[k]
ground_truth = K_fold_test_mask[k]
predicted = K_test_predicted_np[k]
print('  ')
print('Visualizing FOLD{} prediction'.format(k + 1))
print(len(test_image))
for l in range(0, len(test_image)):
    fig, axs = plt.subplots(1, 4, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(10, 10))
    fig.suptitle('FOLD 2 - Testinng: number {}'.format(l), fontsize=20)
    axs = axs.ravel()
    axs[0].imshow(test_image[l])
    axs[0].set_title('Image', fontsize=15, loc='center')
    axs[0].set_yticklabels([])
    axs[0].set_xticklabels([])
    axs[1].imshow(np.squeeze(np.stack((ground_truth[l],) * 3, axis=-1)))
    axs[1].set_title('Ground Truth', fontsize=15, loc='center')
    axs[1].set_yticklabels([])
    axs[1].set_xticklabels([])
    axs[2].imshow(np.squeeze(np.stack((pred[l],) * 3, axis=-1)))
    axs[2].set_title('Predicted', fontsize=15, loc='center')
    axs[2].set_yticklabels([])
    axs[2].set_xticklabels([])
    axs[3].imshow(np.squeeze(np.stack((predicted[l],) * 3, axis=-1)))
    axs[3].set_title('Predicted THR', fontsize=15, loc='center')
    axs[3].set_yticklabels([])
    axs[3].set_xticklabels([])
    plt.show()
    # fig.savefig(os.path.join(figure_path, 'ImageVSMasks_{0}'.format(i-1)))

#FOLD 3
k = 2
print('FOLD {}'.format(k + 1))
training_image = K_fold_train_image[k]
training_mask = K_fold_train_mask[k]
validation_image = K_fold_val_image[k]
validation_mask = K_fold_val_mask[k]
print('The image training set is lenght:', len(training_image))
print('The mask training set is lenght:', len(training_mask))
print('The image validation set is lenght:', len(validation_image))
print('The mask validation set is lenght:', len(validation_mask))

MyModel3 = myUnet()  # creo istanza dell'oggetto
myunet3 = MyModel3.get_unet()

myunet3.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss=dice_loss, metrics=[ssim, 'acc', dice_coeff])

#earlystopper = EarlyStopping(patience=earlystop_patience, verbose=1)
checkpointer = ModelCheckpoint(os.path.join(os.getcwd(), 'NEW DATASET NUMPY 9paz/NEW DATASET NUMPY 9paz/new_model_unet_transpose_checkpoint_{:02d}_fold.h5'.format(k+1)), verbose=1, save_best_only=True, monitor='val_loss', mode = 'min')

training_generator = DataGenerator(training_image, training_mask, batchSize, 3, 1, True, aug)
validation_data = DataGenerator(validation_image, validation_mask, batchSize, 3, 1, True, aug)

results_3 = myunet3.fit(training_generator, validation_data=validation_data, batch_size=batchSize, epochs=2000, callbacks=[checkpointer])
ssim_history = results_3.history["ssim"]
val_ssim_history = results_3.history["val_ssim"]
acc_history = results_3.history["acc"]
val_acc_history = results_3.history["val_acc"]
dice_history = results_3.history["dice_coeff"]
val_dice_history = results_3.history["val_dice_coeff"]
K_ssim_history.append(ssim_history)
K_val_ssim_history.append(val_ssim_history)
K_acc_history.append(acc_history)
K_val_acc_history.append(val_acc_history)
K_dice_history.append(dice_history)
K_val_dice_history.append(val_dice_history)
K_path_model.append(os.path.join(os.getcwd(),'NEW DATASET NUMPY 9paz/NEW DATASET NUMPY 9paz/new_model_unet_transpose_checkpoint_{:02d}_fold.h5'.format(k + 1)))
# saving the metrics' value in a dataset
with h5py.File(os.path.join(os.getcwd(), 'NEW DATASET NUMPY 9paz/NEW DATASET NUMPY 9paz/new_FOLD{0}_Metrics_history_transpose.hdf5'.format(k + 1)), 'w') as f:
    f.create_dataset('ssim', data=ssim_history)
    f.create_dataset('val_ssim', data=val_ssim_history)
    f.create_dataset('acc', data=acc_history)
    f.create_dataset('val_acc', data=val_acc_history)
    f.create_dataset('dice', data=dice_history)
    f.create_dataset('val_dice', data=val_dice_history)
    f.close


predicted = []
test_image_np = K_fold_test_image[k]
print(len(test_image_np))
patient_test_image = []
patient_test_mask = []
patient_test_image.append(K_test[k])
patient_test_mask.append(K_test[k])

for image in test_image_np:
    image = np.array(image / 255.).astype('float32')  # trasforming the image in float 32 with np.max 0 < 1
    prediction = myunet3.predict(np.expand_dims(image, 0))
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
    #print(np.min(a))
    pred_thr.append(a.astype('uint8'))  # image in float 32 with np.max =  1
print('Ending the threshold')


pred_int = pred_thr
print(len(pred_int))
print(np.max(pred_int[0]))

K_test_predicted_np.append(pred_int)

print('           The testing set image is lenght {} and it contains the patient: '.format(len(test_image_np)), patient_test_image)
print('           The prediction set is lenght {} and it contains the patient: '.format(len(test_mask_np)), patient_test_mask)
print('   ')

print(len(K_test_predicted_np))
print(len(K_test_predicted_np[k]))

test_image = K_fold_test_image[k]
ground_truth = K_fold_test_mask[k]
predicted = K_test_predicted_np[k]
print('  ')
print('Visualizing FOLD{} prediction'.format(k + 1))
print(len(test_image))
for l in range(0, len(test_image)):
    fig, axs = plt.subplots(1, 4, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(10, 10))
    fig.suptitle('FOLD 3 - Testinng: number {}'.format(l), fontsize=20)
    axs = axs.ravel()
    axs[0].imshow(test_image[l])
    axs[0].set_title('Image', fontsize=15, loc='center')
    axs[0].set_yticklabels([])
    axs[0].set_xticklabels([])
    axs[1].imshow(np.squeeze(np.stack((ground_truth[l],) * 3, axis=-1)))
    axs[1].set_title('Ground Truth', fontsize=15, loc='center')
    axs[1].set_yticklabels([])
    axs[1].set_xticklabels([])
    axs[2].imshow(np.squeeze(np.stack((pred[l],) * 3, axis=-1)))
    axs[2].set_title('Predicted', fontsize=15, loc='center')
    axs[2].set_yticklabels([])
    axs[2].set_xticklabels([])
    axs[3].imshow(np.squeeze(np.stack((predicted[l],) * 3, axis=-1)))
    axs[3].set_title('Predicted THR', fontsize=15, loc='center')
    axs[3].set_yticklabels([])
    axs[3].set_xticklabels([])
    plt.show()
    # fig.savefig(os.path.join(figure_path, 'ImageVSMasks_{0}'.format(i-1)))
