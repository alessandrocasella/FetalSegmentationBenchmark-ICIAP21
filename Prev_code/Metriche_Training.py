import os
import numpy as np
from matplotlib import pyplot as plt
import h5py
import matplotlib.lines as mlines

K_acc_history = []
K_val_acc_history = []
K_dice_history = []
K_val_dice_history = []
K_ssim_history = []
K_val_ssim_history = []
#reading the dataset
for k in range(0,1):
    d = h5py.File(os.path.join(os.getcwd(), 'RESULT_DENSEUNET/Model 3/NI - REDUCED FOV/FOLD4_Metrics_history_dense_unet.hdf5'), 'r')
    print(list(d.keys()))

''' acc = d['acc']
    dice = d['dice']
    ssim_ = d['ssim']
    val_acc = d['val_acc']
    val_dice = d['val_dice']
    val_ssim_ = d['val_ssim']
    print(acc.shape, dice.shape, ssim_.shape)
    print(acc.dtype, dice.dtype, ssim_.dtype )
    print(val_acc.shape, val_dice.shape, val_ssim_.shape)
    print(val_acc.dtype, val_dice.dtype, val_ssim_.dtype )

    i = len(acc)
    K_acc_history.append(acc[:i])
    K_val_acc_history.append(val_acc[:i])
    K_dice_history.append(dice[:i])
    K_val_dice_history.append(val_dice[:i])
    K_ssim_history.append(ssim_[:i])
    K_val_ssim_history.append(val_ssim_[:i])

for x in range(0,len(K_val_ssim_history)):
    print('  ')
    print('Fold{}'.format(x+1))
    print('Accuracy:{} values'.format(len(K_acc_history[x])))
    print('Validation Accuracy:{} values'.format(len(K_val_acc_history[x])))
    print('Dice:{} values'.format(len(K_dice_history[x])))
    print('Validation Dice:{} values'.format(len(K_val_dice_history[x])))
    print('SSIM:{} values'.format(len(K_ssim_history[x])))
    print('Validation SSIM:{} values'.format(len(K_val_ssim_history[x])))

#VISUALIZIING SSIM
fig, axs = plt.subplots(1, 3, sharex = True, gridspec_kw={'hspace': 0.75, 'wspace': 0.25}, figsize=(40,10))
axs = axs.ravel()
fig.suptitle('Model 1 - SSIM - Structural Similarity Index History', fontsize = 20)
blue_line = mlines.Line2D([], [], color='blue')
orange_line = mlines.Line2D([], [], color='orange')
fig.legend(handles=[blue_line, orange_line], labels = ['Training', 'Validation'], loc = 'upper left', fontsize = 15, borderaxespad=1.5)

SSIM_1 = K_ssim_history[0]
SSIM_2 = K_ssim_history[1]
SSIM_3 = K_ssim_history[2]

val_ssim_1 =  K_val_ssim_history[0]
val_ssim_2 =  K_val_ssim_history[1]
val_ssim_3 =  K_val_ssim_history[2]

axs[0].plot(SSIM_1)
axs[0].plot(val_ssim_1)
axs[0].set_title('FOLD {0}'.format(1), fontsize = 15, loc = 'center')
axs[0].set_ylim(0,1)
axs[1].plot(SSIM_2)
axs[1].plot(val_ssim_2)
axs[1].set_title('FOLD {0}'.format(2), fontsize = 15, loc = 'center')
axs[1].set_ylim(0,1)
axs[2].plot(SSIM_3)
axs[2].plot(val_ssim_3)
axs[2].set_title('FOLD {0}'.format(3), fontsize = 15, loc = 'center')
axs[2].set_ylim(0,1)
plt.show()
#fig.savefig(os.path.join(K_FOLD, 'Model2-500_SSIM_hystory.png'))

#VISUALIZING DICE
fig, axs = plt.subplots(1, 3, sharex = True, gridspec_kw={'hspace': 0.75, 'wspace': 0.25}, figsize=(40,10))
fig.suptitle('Model 1: Dice Coefficient History', fontsize = 20)
blue_line = mlines.Line2D([], [], color='blue')
orange_line = mlines.Line2D([], [], color='orange')
fig.legend(handles=[blue_line, orange_line], labels = ['Training', 'Validation'], loc = 'upper left', fontsize = 15, borderaxespad=1.5)
axs = axs.ravel()

dice_1 = K_dice_history[0]
dice_2 = K_dice_history[1]
dice_3 = K_dice_history[2]

val_dice_1 =  K_val_dice_history[0]
val_dice_2 =  K_val_dice_history[1]
val_dice_3 =  K_val_dice_history[2]

axs[0].plot(dice_1)
axs[0].plot(val_dice_1)
axs[0].set_title('DICE FOLD {0}'.format(1), fontsize = 15, loc = 'center')
axs[0].set_ylim(0,1)
axs[1].plot(dice_2)
axs[1].plot(val_dice_2)
axs[1].set_title('DICE LOSS FOLD {0}'.format(2), fontsize = 15, loc = 'center')
axs[1].set_ylim(0,1)
axs[2].plot(dice_3)
axs[2].plot(val_dice_3)
axs[2].set_title('DICE LOSS FOLD {0}'.format(3), fontsize = 15, loc = 'center')
axs[2].set_ylim(0,1)

plt.show()
#fig.savefig(os.path.join(K_FOLD, 'Model2-500_DICE_hystory.png'))


#VISUALIZING ACCURACY
fig, axs = plt.subplots(1, 3, sharex = True, gridspec_kw={'hspace': 0.75, 'wspace': 0.25}, figsize=(40,10))
fig.suptitle('Model 1: Accuracy History', fontsize = 20)
blue_line = mlines.Line2D([], [], color='blue')
orange_line = mlines.Line2D([], [], color='orange')
fig.legend(handles=[blue_line, orange_line], labels = ['Training', 'Validation'], loc = 'upper left', fontsize = 15, borderaxespad=1.5)
axs = axs.ravel()

acc_1 = K_acc_history[0]
acc_2 = K_acc_history[1]
acc_3 = K_acc_history[2]

val_acc_1 =  K_val_acc_history[0]
val_acc_2 =  K_val_acc_history[1]
val_acc_3 =  K_val_acc_history[2]

axs[0].plot(acc_1)
axs[0].plot(val_acc_1)
axs[0].set_title('ACCURACY FOLD {0}'.format(1), fontsize = 15, loc = 'center')
axs[0].set_ylim(0,1)
axs[1].plot(acc_2)
axs[1].plot(val_acc_2)
axs[1].set_title('ACCURACY FOLD {0}'.format(2), fontsize = 15, loc = 'center')
axs[1].set_ylim(0,1)
axs[2].plot(acc_3)
axs[2].plot(val_acc_3)
axs[2].set_title('ACCURACY FOLD {0}'.format(3), fontsize = 15, loc = 'center')
axs[2].set_ylim(0,1)

plt.show()
#fig.savefig(os.path.join(K_FOLD, 'Model2-500_Acc_hystory.png'))'''