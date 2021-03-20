''' CHANGE THE NAME OF:
                        THE K FOLD DATASET IN LINE 22
                        THE NAME OF THE RESULT DATASET PATH IN LINE 90'''


import os
import random
from os import listdir
from PIL import Image
from torchsummary import summary
from torchvision.transforms import functional as F
from torch.nn import functional as Fn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
#import pytorch_ssim
from sklearn.metrics import confusion_matrix
import h5py
from tqdm import tqdm

scaler = torch.cuda.amp.GradScaler()

K_FOLD = os.path.join(os.getcwd(), 'K FOLD- 2 - NI - REDUCED FOV - POLIMI DATASET')
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


Result_path = os.path.join(os.getcwd(),'RESULTS')
result_model_path = os.path.join(Result_path, 'UNET_IN')
result_dataset_path = os.path.join(result_model_path, 'K FOLD- 2 - NI - REDUCED FOV - POLIMI DATASET')
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
class DConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = DConv(3, 64)
        self.dconv_down2 = DConv(64, 128)
        self.dconv_down3 = DConv(128, 256)
        self.dconv_down4 = DConv(256, 512)
        self.dconv_down5 = DConv(512, 1024)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.trans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DConv(1024, 512)

        self.trans2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DConv(512, 256)

        self.trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DConv(256, 128)

        self.trans4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DConv(128, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)

        x = self.trans1(conv5)
        x = self.up_conv1(torch.cat([x, conv4], dim=1))

        x = self.trans2(conv4)
        x = self.up_conv2(torch.cat([x, conv3], dim=1))

        x = self.trans3(x)
        x = self.up_conv3(torch.cat([x, conv2], dim=1))

        x = self.trans4(x)
        x = self.up_conv4(torch.cat([x, conv1], dim=1))

        out = self.conv_last(x)

        return out

class DataGenerator(torch.utils.data.Dataset):

    def __init__(self, dataDir, labelDir, batch_size, aug):
        # print('__init__')
        self.batch_size = batch_size
        self.dataDir = dataDir
        self.labelDir = labelDir
        self.list_IDs = os.listdir(self.dataDir)
        self.aug = aug
        #self.on_epoch_end()
        self.X = ()
        self.Y = ()
        for i in self.list_IDs:
            x,y = self.__data_generation(i)
            self.X = self.X + (x,)
            self.Y = self.Y + (y,)

    def __len__(self):
        # print('__len__')
        # return int(np.floor(len(self.list_IDs) / self.batch_size))
        i = len(self.list_IDs)
        # print('The lenght of list_ids is {0}'.format(len(self.list_IDs)))
        # print('The division is {0}'.format(len(self.list_IDs) / self.batch_size))
        # print('The divison after the np.flor is {0}'.format(i))
        return i

    def __getitem__(self, index):
        # print('__getitem__')
        if torch.is_tensor(index):
            index = index.tolist()
        #print(index)
        #indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        #file_list_temp = [self.list_IDs[k] for k in indexes]
        # print('The lenght of the file temp is {0}'.format(len(file_list_temp)))

        # print('The lenght of X is {0}'.format(len(X)))
        # print('The lenght of y_new is {0}'.format(len(y_new)))
        return self.X[index], self.Y[index]

    def __data_generation(self, file_list_temp):

        x_file_path = os.path.join(self.dataDir, file_list_temp)
        y_file_path = os.path.join(self.labelDir, file_list_temp)

        X = np.array(np.load(x_file_path))
        y_old = np.load(y_file_path)
        y = np.array((y_old / np.max(y_old)) * 255).astype('uint8')
        X, y = F.to_pil_image(X), F.to_pil_image(y)
        if self.aug:
        #y = np.asarray(np.dstack((y, y, y)), dtype=np.float32)
            deg, trasl, scal, shr = torchvision.transforms.RandomAffine((-180, 180), (0, 0.2), (0.5, 1.5))\
                .get_params((-180, 180), (0, 0.2), (0.5, 1.5), shears=None, img_size=(256, 256))

            if random.random()>0.5:
                X,y = F.hflip(X), F.hflip(y)

            if random.random()>0.5:
                X, y = F.vflip(X), F.vflip(y)

            X, y = F.affine(X, deg, trasl, scal, shr), F.affine(y, deg, trasl, scal, shr)
        #X.save(os.path.join(os.getcwd(),file_list_temp+'.jpg'))
        #y.save(os.path.join(os.getcwd(), file_list_temp + 'm.jpg'))
        x_new = np.array(X, dtype=np.float32) / 255.
        x_new = torch.from_numpy(x_new)
        x_new = x_new.permute(2,0,1).contiguous()
        y_new = np.array(y, dtype=np.float32) / 255.
        y_new = torch.from_numpy(y_new)
        y_new = y_new.unsqueeze_(2)
        y_new = y_new.permute(2,0,1).contiguous()
        return x_new, y_new


def diceCoeff(pred, gt, smooth=1e-5):
    pred = torch.sigmoid(pred)

    # flatten label and prediction tensors
    inputs = pred.view(-1)
    targets = gt.view(-1)
    intersection = (inputs * targets).sum()
    dice = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return dice

class ComboLOSS(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLOSS, self).__init__()
        self.ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        # flatten label and prediction tensors
        inputs_f = inputs.view(-1)
        targets_f = targets.view(-1)

        intersection = (inputs_f * targets_f).sum()
        dice_loss = (2. * intersection + smooth) / (inputs_f.sum() + targets_f.sum() + smooth)
        ssim_loss = self.ssim_loss(inputs, targets)
        Combo_loss = 1. - ( (dice_loss + ssim_loss) / 2. )
        return Combo_loss, dice_loss, ssim_loss


"""## Getting the Unet, visualizing it"""

########################################################
learning_rate = 0.001  # @param {type:"number"}
batchSize = 16  # @param {type:"number"}
epochs = 400
# earlystop_patience = 50 #@param {type:"number"}
# rule of thumb to make it 10% of number of epoch.

# GET THE UNET AND DISPLAY MY MODEL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#plot_model(myunet, to_file='Try/K FOLD/Model-Unet-Transpose/myunet_transpose_plot.png', show_shapes=False, show_layer_names=False, rankdir='LR')

#path = os.path.join(os.getcwd(), 'K FOLD/Model-Unet-Transpose/myunet_transpose_plot.png')
#im = Image.open(path)

#im.save(model_path + '/myunet_transpose_plot.png')


K_ssim_history = []
K_val_ssim_history = []
K_acc_history = []
K_val_acc_history = []
K_dice_history = []
K_val_dice_history = []
K_path_model = []
torch.autograd.set_detect_anomaly(True)
for k in range(0,k_fold):

    model = UNet(1)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9)
    criterion = ComboLOSS()

    best_loss = np.inf
    summary(model, input_size=(3, 256, 256))

    cell_dataset = DataGenerator(train_path_image_list[k], train_path_mask_list[k], batchSize, True)
    cell_val_dataset = DataGenerator(val_path_image_list[k], val_path_mask_list[k], batchSize, False)
    dataloader = DataLoader(cell_dataset, batch_size=batchSize, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(cell_val_dataset, batch_size=batchSize, shuffle=False, num_workers=0, pin_memory=True)
    for epoch in range(epochs):

        avg_loss = []
        avg_loss_train = []
        #i = random.randint(0, len(cell_dataset) - 1)
        model.train()
        for iteration, (input_train, target_train) in enumerate(tqdm(dataloader)):
            # input, target = next(iter(dataloader))
            # input_train, target_train = Variable(batch[0]), Variable(batch[1])
            input_train = input_train.to(device, torch.float32)
            target_train = target_train.to(device, torch.float32)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                output = model(input_train)
                loss, tdice, tssim = criterion(output, target_train)
                loss.backward()
                optimizer.step()

            # loss_item = loss.item()
            avg_loss_train.append(loss.item())

        model.eval()

        for iteration, (input_val, target_val) in enumerate(val_dataloader):
            input_val = input_val.to(device)
            target_val = target_val.to(device)

            with torch.no_grad():
                scores = model.forward(input_val)
                #test = torch.sigmoid(scores)
                #test = test.to('cpu')
                #test = torchvision.transforms.ToPILImage()(test[0])
                #test.save(os.path.join(os.getcwd(), '{}.png'.format(iteration)))
                vloss, vdice, vssim = criterion(scores, target_val)
                vloss = vloss.item()
                vdice = vdice.item()
                vssim = vssim.item()
                avg_loss.append(vloss)

        K_ssim_history.append(tssim)
        K_val_ssim_history.append(vssim)
        K_dice_history.append(tdice)
        K_val_dice_history.append(vdice)
        print('Epoch {}, Loss: {:4f}, Val Loss: {:4f}'.format(epoch, np.mean(avg_loss_train), np.mean(avg_loss)))

        if np.mean(avg_loss) < best_loss:
            best_loss = np.mean(avg_loss)
            print('Saving model')
            torch.save(model, os.path.join(os.getcwd(), 'RESULTS/UNET_IN/K FOLD- 2 - NI - REDUCED FOV - POLIMI DATASET/model_unet_checkpoint_{:02d}_fold.pth'.format(k+1)))


print('The model created are: ', len(K_path_model))
if (len(K_path_model) == k_fold):
    print('One model is created for each fold')
    print('The models path are: ')

for path in K_path_model:
    print(path)


#TESTING
K_test_predicted = []
K_test_thr_predicted = []
K_test_image = []
K_test_ground_truth= []

for k in range(0, k_fold):
    model = UNet(1)
    model = model.to(device)
    print('Fold{}'.format(k+1))
    #path_model = K_path_model[k]
    model = torch.load(os.path.join(os.getcwd(), 'RESULTS/UNET_IN/K FOLD- 2 - NI - REDUCED FOV - POLIMI DATASET/model_unet_checkpoint_{:02d}_fold.pth'.format(k+1)))
    model.eval()
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
        im = torch.from_numpy(np.expand_dims(im,0))
        im = im.to(device, torch.float32)
        im = im.permute(0,3,1,2).float()
        with torch.no_grad():
            prediction = model(im)
        prediction = torch.sigmoid(prediction)
        prediction = prediction.to('cpu').numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        #prediction = torchvision.transforms.ToPILImage()(prediction[0])
        #prediction = np.expand_dims(np.array(prediction),0)
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

if (len(K_test_predicted) != 3):
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
        #plt.show()
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
        image = np.array(image[:, :, 0]*255., dtype=np.uint8)
        im = Image.fromarray(image, 'L')
        im.save(os.path.join(path_fold_png, 'predicted_{:03d}.png'.format(l)))
    for l, image in enumerate(thr):
        np.save(os.path.join(thr_path_fold_numpy, '{:03d}.npy'.format(l)), np.asarray(image))
        image = np.array(image[:, :, 0], dtype=np.uint8)
        im = Image.fromarray(image, mode='L')
        im.save(os.path.join(thr_path_fold_png, 'predicted_{:03d}.png'.format(l)))