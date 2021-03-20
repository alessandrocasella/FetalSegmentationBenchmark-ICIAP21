''' CHANGE THE NAME OF:
                        THE K FOLD DATASET IN LINE 22
                        THE NAME OF THE RESULT DATASET PATH IN LINE 90'''


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

K_FOLD = os.path.join(os.getcwd(), 'K FOLD - 2 - CONTRAST -  NI - REDUCED FOV - POLIMI DATASET')
print(K_FOLD)
K_FOLD_subfolders = [f.path for f in os.scandir(K_FOLD) if f.is_dir()]
K_FOLD_subfolders.sort()
print(K_FOLD_subfolders)
Numpy_subfolders = [f.path for f in os.scandir(K_FOLD_subfolders[0]) if f.is_dir()]
Numpy_subfolders.sort()
print(Numpy_subfolders)

k_fold = len(Numpy_subfolders)
print(k_fold)

train_path_image_list = []
val_path_image_list = []
test_path_image_list = []
train_path_mask_list = []
val_path_mask_list = []
test_path_mask_list = []

for k in range(0,k_fold):
    print('Fold{}'.format(k+1))
    FOLD = Numpy_subfolders[k]
    FOLD_subfolders = [ f.path for f in os.scandir(FOLD) if f.is_dir() ]
    FOLD_subfolders.sort()
    print(FOLD_subfolders)
    FOLD_test_subfolders = [ f.path for f in os.scandir(FOLD_subfolders[0]) if f.is_dir() ]
    FOLD_test_subfolders.sort()
    FOLD_train_subfolders = [ f.path for f in os.scandir(FOLD_subfolders[1]) if f.is_dir() ]
    FOLD_train_subfolders.sort()
    FOLD_val_subfolders = [ f.path for f in os.scandir(FOLD_subfolders[2]) if f.is_dir() ]
    FOLD_val_subfolders.sort()
    print(FOLD_test_subfolders)
    print(FOLD_train_subfolders)
    print(FOLD_val_subfolders)
    test_path_image_list.append(FOLD_test_subfolders[0])
    train_path_image_list.append(FOLD_train_subfolders[0])
    val_path_image_list.append(FOLD_val_subfolders[0])
    test_path_mask_list.append(FOLD_test_subfolders[1])
    train_path_mask_list.append(FOLD_train_subfolders[1])
    val_path_mask_list.append(FOLD_val_subfolders[1])

if (len(test_path_image_list) != 6):
    print('Error in image test path')
if (len(train_path_image_list) != 6):
    print('Error in image train path')
if (len(val_path_image_list) != 6):
    print('Error in image val path')

if (len(test_path_mask_list) != 6):
    print('Error in mask test path')
if (len(train_path_mask_list) != 6):
    print('Error in mask train path')
if (len(val_path_mask_list) != 6):
    print('Error in mask val path')

print(test_path_image_list)
print(train_path_image_list)
print(val_path_image_list)
print(test_path_mask_list)
print(train_path_mask_list)
print(val_path_mask_list)


Result_path = os.path.join(os.getcwd(),'RESULT_GENOVA')
result_model_path = os.path.join(Result_path, 'Model 2')
result_dataset_path = os.path.join(result_model_path, 'CONTRAST -  NI - REDUCED FOV')
try:
    os.mkdir(Result_path)
    os.mkdir(result_model_path)
    os.mkdir(result_dataset_path)
except:
    pass
try:
    os.mkdir(result_dataset_path)
except:
    pass
print(Result_path)
print(result_model_path)
print(result_dataset_path)

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

        u6 = UpSampling2D((2, 2), interpolation = 'nearest')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = BatchNormalization()(c6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
        c6 = BatchNormalization()(c6)

        u7 = UpSampling2D((2, 2), interpolation = 'nearest')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = BatchNormalization()(c7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
        c7 = BatchNormalization()(c7)

        u8 = UpSampling2D((2, 2), interpolation = 'nearest')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = BatchNormalization()(c8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
        c8 = BatchNormalization()(c8)

        u9 = UpSampling2D((2, 2), interpolation = 'nearest')(c8)
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

noaug = ImageDataGenerator(rescale=1.)
# to obtain images distorted
class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataDir, labelDir, batch_size, nChannelData, nChannelLabel, shuffle, aug):
        # print('__init__')
        self.batch_size = batch_size
        self.dataDir = dataDir
        self.labelDir = labelDir
        self.nChannelData = nChannelData
        self.nChannelLabel = nChannelLabel
        self.shuffle = shuffle
        self.list_IDs = os.listdir(self.dataDir)
        # print('The list IDS is {0}'.format(self.list_IDs))
        self.aug = aug
        self.on_epoch_end()

    def __len__(self):
        # print('__len__')
        # return int(np.floor(len(self.list_IDs) / self.batch_size))
        i = int(np.floor(len(self.list_IDs) / self.batch_size))
        # print('The lenght of list_ids is {0}'.format(len(self.list_IDs)))
        # print('The division is {0}'.format(len(self.list_IDs) / self.batch_size))
        # print('The divison after the np.flor is {0}'.format(i))
        return i

    def __getitem__(self, index):
        # print('__getitem__')
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        file_list_temp = [self.list_IDs[k] for k in indexes]
        # print('The lenght of the file temp is {0}'.format(len(file_list_temp)))
        X, y_new = self.__data_generation(file_list_temp)
        # print('The lenght of X is {0}'.format(len(X)))
        # print('The lenght of y_new is {0}'.format(len(y_new)))
        return X, y_new

    def on_epoch_end(self):
        # print('on_epoch_end')
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_list_temp):
        # print('__data_generation')
        y_new = np.empty([self.batch_size, 256, 256, 1], dtype=np.float32)
        x_new = np.empty([self.batch_size, 256, 256, 3], dtype=np.float32)
        for p, ID in enumerate(file_list_temp):
            x_file_path = os.path.join(self.dataDir, ID)
            y_file_path = os.path.join(self.labelDir, ID)

            transformInstance = self.aug.get_random_transform((256, 256, 3))
            # X = np.array(np.load(x_file_path)).astype('float32')
            X = np.array(np.load(x_file_path) / 255.).astype('float32')

            Z = X
            # y = np.array(np.load(y_file_path)).astype('float32')
            y_old = np.load(y_file_path)
            y = np.array(y_old / np.max(y_old)).astype('float32')
            y = np.asarray(np.dstack((y, y, y)), dtype=np.float32)

            Z[:, :, :] = self.aug.apply_transform(X[:, :, :], transformInstance)
            '''if (np.max(Z)!= 1.0):
                print('The max in Z is not 1 but is {0}'.format(np.max(Z)))
            print('The shape of Z is {0}'.format(Z.shape))'''
            x_new[p, :, :, :] = Z[:, :, :]
            # print('The shape of x_new is {0}'.format(x_new.shape))
            y[:, :, :] = self.aug.apply_transform(y[:, :, :], transformInstance)
            '''if (np.max(y)!= 1.0):
                print('The max in y is not 1 but {0}'.format(np.max(y)))
            print('The max in y is {0}'.format(y.shape))'''
            y_new[p, :, :, 0] = y[:, :, 0]
            # print('The shape of y_new is {0}'.format(y_new.shape))
        return x_new, y_new



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

def combo_loss(y_true, y_pred):
    return 0.5*dice_loss(y_true, y_pred) + 0.5*ssim_loss(y_true, y_pred)

"""## Getting the Unet, visualizing it"""

########################################################
learning_rate = 0.001  # @param {type:"number"}
batchSize = 32  # @param {type:"number"}
# earlystop_patience = 50 #@param {type:"number"}
# rule of thumb to make it 10% of number of epoch.

# GET THE UNET AND DISPLAY MY MODEL
MyModel = myUnet()  # creo istanza dell'oggetto
myunet = MyModel.get_unet()  # questa funzione, data l'istanza dell'oggetto, associa un modello a questa variabile
myunet.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss=combo_loss, metrics=[ssim, 'acc', dice_coeff])
myunet.summary()

#plot_model(myunet, to_file='Try/K FOLD/Model-Unet-Transpose/myunet_transpose_plot.png', show_shapes=False, show_layer_names=False, rankdir='LR')

#path = os.path.join(os.getcwd(), 'K FOLD/Model-Unet-Transpose/myunet_transpose_plot.png')
#im = Image.open(path)

#im.save(model_path + '/myunet_transpose_plot.png')

'''
K_ssim_history = []
K_val_ssim_history = []
K_acc_history = []
K_val_acc_history = []
K_dice_history = []
K_val_dice_history = []
K_path_model = []

for k in range(5,k_fold):

    MyModel = myUnet()  # creo istanza dell'oggetto
    myunet = MyModel.get_unet()

    myunet.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss=combo_loss, metrics=[ssim, 'acc', dice_coeff])

    #earlystopper = EarlyStopping(patience=earlystop_patience, verbose=1)
    checkpointer = ModelCheckpoint(os.path.join(os.getcwd(), 'RESULT_GENOVA/Model 2/CONTRAST - NI - REDUCED FOV/model_unet_upsampling_checkpoint_{:02d}_fold.h5'.format(k+1)), verbose=1, save_best_only=True, monitor='val_loss', mode = 'min')

    training_generator = DataGenerator(train_path_image_list[k], train_path_mask_list[k], batchSize, 3, 1, True, aug)
    validation_data = DataGenerator(val_path_image_list[k], val_path_mask_list[k], batchSize, 3, 1, True, noaug)

    results = myunet.fit(training_generator, validation_data=validation_data, batch_size=batchSize, epochs=700, callbacks=[checkpointer])
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
    K_path_model.append(os.path.join(os.getcwd(),'RESULT_GENOVA//Model 2/CONTRAST - NI - REDUCED FOV/model_unet_upsampling_checkpoint_{:02d}_fold.h5'.format(k + 1)))
    # saving the metrics' value in a dataset
    with h5py.File(os.path.join(os.getcwd(), 'RESULT_GENOVA//Model 2/CONTRAST - NI - REDUCED FOV/FOLD{0}_Metrics_history_upsampling.hdf5'.format(k + 1)), 'w') as f:
        f.create_dataset('ssim', data=ssim_history)
        f.create_dataset('val_ssim', data=val_ssim_history)
        f.create_dataset('acc', data=acc_history)
        f.create_dataset('val_acc', data=val_acc_history)
        f.create_dataset('dice', data=dice_history)
        f.create_dataset('val_dice', data=val_dice_history)
        f.close

print('The model created are: ', len(K_path_model))
if (len(K_path_model) == k_fold):
    print('One model is created for each fold')
    print('The models path are: ')

for path in K_path_model:
    print(path)
'''
#TESTING
K_test_predicted = []
K_test_thr_predicted = []
K_test_image = []
K_test_ground_truth= []

for k in range(0, k_fold):
    print('Fold{}'.format(k+1))
    #path_model = K_path_model[k]
    myunet = load_model(os.path.join(os.getcwd(), 'RESULT_GENOVA/Model 2/CONTRAST - NI - REDUCED FOV/model_unet_upsampling_checkpoint_{:02d}_fold.h5'.format(k+1)), compile=False)
    test_image = []
    predicted_4d = []
    predicted_3d = []
    pred_thr = []
    test_image_list = os.listdir(test_path_image_list[k])
    test_image_list.sort()
    for image in test_image_list:
        t_file_path = os.path.join(test_path_image_list[k], image)
        im = np.array(np.load(t_file_path) / 255.).astype('float32')  # trasforming the image in float 32 with np.max 0< 1
        test_image.append(im)
        prediction = myunet.predict(np.expand_dims(im, 0))
        if (prediction.shape != (1, 256, 256, 1)):
            print('error shape 1')
        if (np.max(prediction) < 0.5):
            print('error MAX! The max is {0}'.format(np.max(prediction)))
        predicted_4d.append(prediction)
        # deleting the first dimension
        p = prediction[0]        # image in float 32 with np.max 0< 1
        if (p.shape != (256, 256, 1)):
            print('error shape 2')
        predicted_3d.append(p)
        # thresholding the prediction
        a = np.where(p >= 0.5, 255, 0) # image in float 32 with np.max =  1
        pred_thr.append(a.astype('uint8'))  # image in float 32 with np.max =  1

    print('The Image tested are {1} and the prediction are {0}'.format(len(predicted_4d), len(test_image)))
    print('The reshaped prediction are {0}'.format(len(predicted_3d)))
    print('The thresholded prediction are {0}'.format(len(pred_thr)))
    K_test_thr_predicted.append(pred_thr)
    K_test_predicted.append(predicted_3d)
    K_test_image.append(test_image)

if (len(K_test_predicted)!=3):
    print('error prediction 3d')
if (len(K_test_thr_predicted) != 3):
    print('error prediction thr')

for k in range(0, k_fold):
    ground_truth = []
    ground_truth_list = os.listdir(test_path_mask_list[k])
    ground_truth_list.sort()
    for image in ground_truth_list:
        t_file_path = os.path.join(test_path_mask_list[k], image)
        im = np.array(np.load(t_file_path) / 255.).astype('float32')  # trasforming the image in float 32 with np.max 0< 1
        ground_truth.append(im)
    K_test_ground_truth.append(ground_truth)

directory2 = 'Numpy Prediction'
prediction_path_numpy = os.path.join(result_dataset_path, directory2)
directory3 = 'PNG Prediction'
prediction_path_png = os.path.join(result_dataset_path, directory3)
directory4 = 'THR'
thr_prediction_path_numpy = os.path.join(prediction_path_numpy, directory4)
directory6 = 'THR'
thr_prediction_path_png = os.path.join(prediction_path_png, directory6)
directory5 = 'Comparison_Image_GroundTruth_Prediction'
figure_path = os.path.join(result_dataset_path, directory5)
try:
    os.mkdir(prediction_path_numpy)
    os.mkdir(prediction_path_png)
    os.mkdir(thr_prediction_path_numpy)
    os.mkdir(thr_prediction_path_png)
    os.mkdir(figure_path)
except:
    pass
print(figure_path)
print(prediction_path_numpy)
print(prediction_path_png)
print(thr_prediction_path_png)
print(thr_prediction_path_numpy)

for k in range(0, k_fold):
    test_image = K_test_image[k]
    ground_truth = K_test_ground_truth[k]
    predicted = K_test_predicted[k]
    thr = K_test_thr_predicted[k]
    fold = 'Fold{}_'.format(k+1)
    print('  ')
    print('Visualizing FOLD{} prediction'.format(k + 1))

    print(len(test_image))
    l = 0
    while (l < len(test_image)):
        fig, axs = plt.subplots(4, 4, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(10, 10))
        fig.suptitle('FOLD {0} - Testing'.format(k+1), fontsize=20)
        axs = axs.ravel()
        axs[0].imshow(test_image[l])
        axs[0].set_title('Image', fontsize=15, loc='center')
        axs[0].set_yticklabels([])
        axs[0].set_xticklabels([])
        axs[1].imshow(np.squeeze(np.stack((ground_truth[l],) * 3, axis=-1)))
        axs[1].set_title('Ground Truth', fontsize=15, loc='center')
        axs[1].set_yticklabels([])
        axs[1].set_xticklabels([])
        axs[2].imshow(np.squeeze(np.stack((predicted[l],) * 3, axis=-1)))
        axs[2].set_title('Predicted', fontsize=15, loc='center')
        axs[2].set_yticklabels([])
        axs[2].set_xticklabels([])
        axs[3].imshow(np.squeeze(np.stack((thr[l],) * 3, axis=-1)))
        axs[3].set_title('Predicted THR', fontsize=15, loc='center')
        axs[3].set_yticklabels([])
        axs[3].set_xticklabels([])
        axs[4].imshow(test_image[l+1])
        axs[4].set_yticklabels([])
        axs[4].set_xticklabels([])
        axs[5].imshow(np.squeeze(np.stack((ground_truth[l+1],) * 3, axis=-1)))
        axs[5].set_yticklabels([])
        axs[5].set_xticklabels([])
        axs[6].imshow(np.squeeze(np.stack((predicted[l+1],) * 3, axis=-1)))
        axs[6].set_yticklabels([])
        axs[6].set_xticklabels([])
        axs[7].imshow(np.squeeze(np.stack((thr[l+1],) * 3, axis=-1)))
        axs[7].set_yticklabels([])
        axs[7].set_xticklabels([])
        axs[8].imshow(test_image[l + 2])
        axs[8].set_yticklabels([])
        axs[8].set_xticklabels([])
        axs[9].imshow(np.squeeze(np.stack((ground_truth[l + 2],) * 3, axis=-1)))
        axs[9].set_yticklabels([])
        axs[9].set_xticklabels([])
        axs[10].imshow(np.squeeze(np.stack((predicted[l + 2],) * 3, axis=-1)))
        axs[10].set_yticklabels([])
        axs[10].set_xticklabels([])
        axs[11].imshow(np.squeeze(np.stack((thr[l + 2],) * 3, axis=-1)))
        axs[11].set_yticklabels([])
        axs[11].set_xticklabels([])
        axs[12].imshow(test_image[l + 3])
        axs[12].set_yticklabels([])
        axs[12].set_xticklabels([])
        axs[13].imshow(np.squeeze(np.stack((ground_truth[l + 3],) * 3, axis=-1)))
        axs[13].set_yticklabels([])
        axs[13].set_xticklabels([])
        axs[14].imshow(np.squeeze(np.stack((predicted[l + 3],) * 3, axis=-1)))
        axs[14].set_yticklabels([])
        axs[14].set_xticklabels([])
        axs[15].imshow(np.squeeze(np.stack((thr[l + 3],) * 3, axis=-1)))
        axs[15].set_yticklabels([])
        axs[15].set_xticklabels([])
        l = l+4
        plt.show()
        fig.savefig(os.path.join(figure_path, fold + 'ImageVSMasks_{0}'.format(l)))

#saving the prediction in png and numpy
for k in range(0, k_fold):
    predicted = K_test_predicted[k]
    thr = K_test_thr_predicted[k]
    fold = 'Fold{}_'.format(k+1)
    path_fold_numpy = os.path.join(prediction_path_numpy, fold)
    thr_path_fold_numpy = os.path.join(thr_prediction_path_numpy, fold)
    path_fold_png = os.path.join(prediction_path_png, fold)
    thr_path_fold_png = os.path.join(thr_prediction_path_png, fold)
    try:
        os.mkdir(path_fold_numpy)
        os.mkdir(thr_path_fold_numpy)
        os.mkdir(path_fold_png)
        os.mkdir(thr_path_fold_png)
    except:
        pass
    print(path_fold_numpy)
    print(thr_path_fold_numpy)
    print(path_fold_png)
    print(thr_path_fold_png)
    for l, image in enumerate(predicted):
        np.save(os.path.join(path_fold_numpy, '{:03d}.npy'.format(l)), np.asarray(image))
        image = image[:, :, 0]
        im = Image.fromarray(image, mode='L')
        im.save(os.path.join(path_fold_png, 'predicted_{:03d}.png'.format(l)))
    for l, image in enumerate(thr):
        np.save(os.path.join(thr_path_fold_numpy, '{:03d}.npy'.format(l)), np.asarray(image))
        image = image[:, :, 0]
        im = Image.fromarray(image, mode='L')
        im.save(os.path.join(thr_path_fold_png, 'predicted_{:03d}.png'.format(l)))