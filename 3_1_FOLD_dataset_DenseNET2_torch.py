''' CHANGE THE NAME OF:
                        THE K FOLD DATASET IN LINE 22
                        THE NAME OF THE RESULT DATASET PATH IN LINE 90'''
import math
import os
import random
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.transforms import functional as F
from tqdm import tqdm

K_FOLD = os.path.join(os.getcwd(), 'KFOLD')
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
result_model_path = os.path.join(Result_path, 'DENSE')
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

class TransitionDown(nn.Sequential):
    r"""
    Transition Down Block as described in [FCDenseNet](https://arxiv.org/abs/1611.09326),
    plus compression from [DenseNet](https://arxiv.org/abs/1608.06993)
    Consists of:
    - Batch Normalization
    - ReLU
    - 1x1 Convolution (with optional compression of the number of channels)
    - (Dropout)
    - 2x2 Max Pooling
    """

    def __init__(self, in_channels: int, compression: float = 1.0, dropout: float = 0.0):
        super(TransitionDown, self).__init__()

        if not 0.0 < compression <= 1.0:
            raise ValueError(f'Compression must be in (0, 1] range, got {compression}')

        self.in_channels = in_channels
        self.dropout = dropout
        self.compression = compression
        self.out_channels = int(math.ceil(compression * in_channels))

        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False))

        if dropout > 0:
            self.add_module('drop', nn.Dropout2d(dropout))

        self.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2))

class TransitionUp(nn.Module):
    r"""
    Transition Up Block as described in [FCDenseNet](https://arxiv.org/abs/1611.09326)
    The block upsamples the feature map and concatenates it with the feature map coming from the skip connection.
    If the two maps don't overlap perfectly they are first aligened centrally and cropped to match.
    """

    def __init__(self, upsample_channels: int, skip_channels: Optional[int] = None):
        r"""
        :param upsample_channels: number of channels from the upsampling path
        :param skip_channels: number of channels from the skip connection, it is not required,
                              but if specified allows to statically compute the number of output channels
        """
        super(TransitionUp, self).__init__()

        self.upsample_channels = upsample_channels
        self.skip_channels = skip_channels
        self.out_channels = upsample_channels + skip_channels if skip_channels is not None else None

        self.add_module('upconv', nn.ConvTranspose2d(self.upsample_channels, self.upsample_channels,
                                                  kernel_size=3, stride=2, padding=0, bias=True))
        self.add_module('concat', CenterCropConcat())

    def forward(self, upsample, skip):
        if self.skip_channels is not None and skip.shape[1] != self.skip_channels:
            raise ValueError(f'Number of channels in the skip connection input ({skip.shape[1]}) '
                             f'is different from the expected number of channels ({self.skip_channels})')
        res = self.upconv(upsample)
        res = self.concat(res, skip)
        return res


class CenterCropConcat(nn.Module):
    def forward(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError(f'x and y inputs contain a different number of samples')
        height = min(x.size(2), y.size(2))
        width = min(x.size(3), y.size(3))

        x = self.center_crop(x, height, width)
        y = self.center_crop(y, height, width)

        res = torch.cat([x, y], dim=1)
        return res

    @staticmethod
    def center_crop(x, target_height, target_width):
        current_height = x.size(2)
        current_width = x.size(3)
        min_h = (current_width - target_width) // 2
        min_w = (current_height - target_height) // 2
        return x[:, :, min_w:(min_w + target_height), min_h:(min_h + target_width)]

class Bottleneck(nn.Sequential):
    r"""
    A 1x1 convolutional layer, followed by Batch Normalization and ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module('norm', nn.BatchNorm2d(num_features=out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

class DenseLayer(nn.Sequential):
    r"""
    Dense Layer as described in [DenseNet](https://arxiv.org/abs/1608.06993)
    and implemented in https://github.com/liuzhuang13/DenseNet
    Consists of:
    - Batch Normalization
    - ReLU
    - (Bottleneck)
    - 3x3 Convolution
    - (Dropout)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 bottleneck_ratio: Optional[int] = None, dropout: float = 0.0):
        super(DenseLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        if bottleneck_ratio is not None:
            self.add_module('bottleneck', Bottleneck(in_channels, bottleneck_ratio * out_channels))
            in_channels = bottleneck_ratio * out_channels

        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))

        if dropout > 0:
            self.add_module('drop', nn.Dropout2d(dropout, inplace=True))

class DenseBlock(nn.Module):
    r"""
    Dense Block as described in [DenseNet](https://arxiv.org/abs/1608.06993)
    and implemented in https://github.com/liuzhuang13/DenseNet
    - Consists of several DenseLayer (possibly using a Bottleneck and Dropout) with the same output shape
    - The first DenseLayer is fed with the block input
    - Each subsequent DenseLayer is fed with a tensor obtained by concatenating the input and the output
      of the previous DenseLayer on the channel axis
    - The block output is the concatenation of the output of every DenseLayer, and optionally the block input,
      so it will have a channel depth of (growth_rate * num_layers) or (growth_rate * num_layers + in_channels)
    """

    def __init__(self, in_channels: int, growth_rate: int, num_layers: int,
                 concat_input: bool = False, dense_layer_params: Optional[dict] = None):
        super(DenseBlock, self).__init__()

        self.concat_input = concat_input
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.out_channels = growth_rate * num_layers
        if self.concat_input:
            self.out_channels += self.in_channels

        if dense_layer_params is None:
            dense_layer_params = {}

        for i in range(num_layers):
            self.add_module(
                f'layer_{i}',
                DenseLayer(in_channels=in_channels + i * growth_rate, out_channels=growth_rate, **dense_layer_params)
            )

    def forward(self, block_input):
        layer_input = block_input
        # empty tensor (not initialized) + shape=(0,)
        layer_output = block_input.new_empty(0)

        all_outputs = [block_input] if self.concat_input else []
        for layer in self._modules.values():
            layer_input = torch.cat([layer_input, layer_output], dim=1)
            layer_output = layer(layer_input)
            all_outputs.append(layer_output)

        return torch.cat(all_outputs, dim=1)


class FCDenseNet(nn.Module):
    r"""
    The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation
    https://arxiv.org/abs/1611.09326
    In this paper, we extend DenseNets to deal with the problem of semantic segmentation. We achieve state-of-the-art
    results on urban scene benchmark datasets such as CamVid and Gatech, without any further post-processing module nor
    pretraining. Moreover, due to smart construction of the model, our approach has much less parameters than currently
    published best entries for these datasets.
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1,
                 initial_num_features: int = 48,
                 dropout: float = 0.2,

                 down_dense_growth_rates: Union[int, Sequence[int]] = 16,
                 down_dense_bottleneck_ratios: Union[Optional[int], Sequence[Optional[int]]] = None,
                 down_dense_num_layers: Union[int, Sequence[int]] = (4, 5, 7, 10, 12),
                 down_transition_compression_factors: Union[float, Sequence[float]] = 1.0,

                 middle_dense_growth_rate: int = 16,
                 middle_dense_bottleneck: Optional[int] = None,
                 middle_dense_num_layers: int = 15,

                 up_dense_growth_rates: Union[int, Sequence[int]] = 16,
                 up_dense_bottleneck_ratios: Union[Optional[int], Sequence[Optional[int]]] = None,
                 up_dense_num_layers: Union[int, Sequence[int]] = (12, 10, 7, 5, 4)):
        super(FCDenseNet, self).__init__()

        # region Parameters handling
        self.in_channels = in_channels
        self.out_channels = out_channels

        if type(down_dense_growth_rates) == int:
            down_dense_growth_rates = (down_dense_growth_rates,) * 5
        if down_dense_bottleneck_ratios is None or type(down_dense_bottleneck_ratios) == int:
            down_dense_bottleneck_ratios = (down_dense_bottleneck_ratios,) * 5
        if type(down_dense_num_layers) == int:
            down_dense_num_layers = (down_dense_num_layers,) * 5
        if type(down_transition_compression_factors) == float:
            down_transition_compression_factors = (down_transition_compression_factors,) * 5

        if type(up_dense_growth_rates) == int:
            up_dense_growth_rates = (up_dense_growth_rates,) * 5
        if up_dense_bottleneck_ratios is None or type(up_dense_bottleneck_ratios) == int:
            up_dense_bottleneck_ratios = (up_dense_bottleneck_ratios,) * 5
        if type(up_dense_num_layers) == int:
            up_dense_num_layers = (up_dense_num_layers,) * 5
        # endregion

        # region First convolution
        # The Lasagne implementation uses convolution with 'same' padding, the PyTorch equivalent is padding=1
        self.features = nn.Conv2d(in_channels, initial_num_features, kernel_size=3, padding=1, bias=False)
        current_channels = self.features.out_channels
        # endregion

        # region Downward path
        # Pairs of Dense Blocks with input concatenation and TransitionDown layers
        down_dense_params = [
            {
                'concat_input': True,
                'growth_rate': gr,
                'num_layers': nl,
                'dense_layer_params': {
                    'dropout': dropout,
                    'bottleneck_ratio': br
                }
            }
            for gr, nl, br in
            zip(down_dense_growth_rates, down_dense_num_layers, down_dense_bottleneck_ratios)
        ]
        down_transition_params = [
            {
                'dropout': dropout,
                'compression': c
            } for c in down_transition_compression_factors
        ]
        skip_connections_channels = []

        self.down_dense = nn.Module()
        self.down_trans = nn.Module()
        down_pairs_params = zip(down_dense_params, down_transition_params)
        for i, (dense_params, transition_params) in enumerate(down_pairs_params):
            block = DenseBlock(current_channels, **dense_params)
            current_channels = block.out_channels
            self.down_dense.add_module(f'block_{i}', block)

            skip_connections_channels.append(block.out_channels)

            transition = TransitionDown(current_channels, **transition_params)
            current_channels = transition.out_channels
            self.down_trans.add_module(f'trans_{i}', transition)
        # endregion

        # region Middle block
        # Renamed from "bottleneck" in the paper, to avoid confusion with the Bottleneck of DenseLayers
        self.middle = DenseBlock(
            current_channels,
            middle_dense_growth_rate,
            middle_dense_num_layers,
            concat_input=True,
            dense_layer_params={
                'dropout': dropout,
                'bottleneck_ratio': middle_dense_bottleneck
            })
        current_channels = self.middle.out_channels
        # endregion

        # region Upward path
        # Pairs of TransitionUp layers and Dense Blocks without input concatenation
        up_transition_params = [
            {
                'skip_channels': sc,
            } for sc in reversed(skip_connections_channels)
        ]
        up_dense_params = [
            {
                'concat_input': False,
                'growth_rate': gr,
                'num_layers': nl,
                'dense_layer_params': {
                    'dropout': dropout,
                    'bottleneck_ratio': br
                }
            }
            for gr, nl, br in
            zip(up_dense_growth_rates, up_dense_num_layers, up_dense_bottleneck_ratios)
        ]

        self.up_dense = nn.Module()
        self.up_trans = nn.Module()
        up_pairs_params = zip(up_transition_params, up_dense_params)
        for i, (transition_params_up, dense_params_up) in enumerate(up_pairs_params):
            transition = TransitionUp(current_channels, **transition_params_up)
            current_channels = transition.out_channels
            self.up_trans.add_module(f'trans_{i}', transition)

            block = DenseBlock(current_channels, **dense_params_up)
            current_channels = block.out_channels
            self.up_dense.add_module(f'block_{i}', block)
        # endregion

        # region Final convolution
        self.final = nn.Conv2d(current_channels, out_channels, kernel_size=1, bias=False)
        # endregion

        # region Weight initialization
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                module.reset_parameters()
            elif isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.constant_(module.bias, 0)
        # endregion

    def forward(self, x):
        res = self.features(x)

        skip_tensors = []
        for dense, trans in zip(self.down_dense.children(), self.down_trans.children()):
            res = dense(res)
            skip_tensors.append(res)
            res = trans(res)

        res = self.middle(res)

        for skip, trans, dense in zip(reversed(skip_tensors), self.up_trans.children(), self.up_dense.children()):
            res = trans(res, skip)
            res = dense(res)

        res = self.final(res)

        return res

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
        y_old[y_old!=34] = 0
        y = np.array((y_old / np.max(y_old)) * 255.).astype('uint8')
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
        #x_new = torch.from_numpy(x_new)#.to('cuda')
        #x_new = x_new.permute(2,0,1).contiguous()
        #x_new = x_new.unsqueeze_(0)
        y_new = np.array(y, dtype=np.float32) / 255.
        #y_new = torch.from_numpy(y_new)#.to('cuda')
        y_new = np.expand_dims(y_new, 2)
        #y_new = y_new.permute(2,0,1).contiguous()
        #y_new = y_new.unsqueeze_(0)
        return x_new, y_new

def collate_fn(batch):
    x_batch, y_batch = [], []
    for x,y in batch:
        x_batch.append(x), y_batch.append(y)
    x_batch, y_batch = torch.Tensor(x_batch), torch.Tensor(y_batch)
    x_batch, y_batch = x_batch.permute(0, 3, 1, 2), y_batch.permute(0, 3, 1, 2)
    return x_batch, y_batch

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

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs_f = inputs.view(-1)
        targets_f = targets.view(-1)

        intersection = (inputs_f * targets_f).sum()
        dice_loss = (2. * intersection + smooth) / (inputs_f.sum() + targets_f.sum() + smooth)
        ssim_loss = ssim( inputs, targets, data_range=1, size_average=True, nonnegative_ssim=True )
        Combo_loss = 2. - (dice_loss + ssim_loss)
        return Combo_loss, dice_loss, ssim_loss


"""## Getting the Unet, visualizing it"""

########################################################
learning_rate = 0.001  # @param {type:"number"}
batchSize = 4 # @param {type:"number"}
epochs = 300
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
for k in range(7,k_fold):
    torch.cuda.empty_cache()
    model = FCDenseNet()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9)
    criterion = ComboLOSS()

    best_loss = np.inf
    summary(model, input_size=(3, 256, 256))

    cell_dataset = DataGenerator(train_path_image_list[k], train_path_mask_list[k], batchSize, True)
    cell_val_dataset = DataGenerator(val_path_image_list[k], val_path_mask_list[k], batchSize, False)
    dataloader = DataLoader(cell_dataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn, num_workers=8,
                            pin_memory=True)
    val_dataloader = DataLoader(cell_val_dataset, batch_size=batchSize, shuffle=False, collate_fn=collate_fn,
                                num_workers=8, pin_memory=True)
    for epoch in range(epochs):

        avg_loss = []
        avg_loss_train = []
        #i = random.randint(0, len(cell_dataset) - 1)
        model.train()
        for iteration, (input_train, target_train) in enumerate(tqdm(dataloader)):

            #input, target = next(iter(dataloader))
            #input_train, target_train = Variable(batch[0]), Variable(batch[1])
            input_train = input_train.to(device, torch.float32)
            target_train = target_train.to(device,torch.float32)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                output = model(input_train)
                loss, tdice, tssim = criterion(output, target_train)
                loss.backward()
                optimizer.step()
            #loss_item = loss.item()
            avg_loss_train.append(loss.item())


        model.eval()

        for iteration, (input_val, target_val) in enumerate(val_dataloader):
            input_val = input_val.to(device)
            target_val = target_val.to(device)

            with torch.no_grad():
                scores = model.forward(input_val)
                test = torch.sigmoid(scores)
                test = test.to('cpu')
                test = torchvision.transforms.ToPILImage()(test[0])
                test.save(os.path.join(os.getcwd(), '{}.png'.format(iteration)))
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
            torch.save(model, os.path.join(os.getcwd(), 'RESULTS/DENSE/K FOLD- 2 - NI - REDUCED FOV - POLIMI DATASET/model_unet_attention_checkpoint_{:02d}_fold.pth'.format(k+1)))


    #ssim_history = results.history["ssim"]
    #val_ssim_history = results.history["val_ssim"]
    #acc_history = results.history["acc"]
    #val_acc_history = results.history["val_acc"]
    #dice_history = results.history["dice_coeff"]
    #val_dice_history = results.history["val_dice_coeff"]
    '''
    K_ssim_history.append(ssim_history)
    K_val_ssim_history.append(val_ssim_history)
    K_acc_history.append(acc_history)
    K_val_acc_history.append(val_acc_history)
    K_dice_history.append(dice_history)
    K_val_dice_history.append(val_dice_history)
    K_path_model.append(os.path.join(os.getcwd(),'RESULTS/ATTENTIONUNET/K FOLD- 2 - NI - REDUCED FOV - POLIMI DATASET/model_unet_attention_checkpoint_{:02d}_fold.h5'.format(k + 1)))
    # saving the metrics' value in a dataset
    with h5py.File(os.path.join(os.getcwd(), 'RESULTS/ATTENTIONUNET/K FOLD- 2 - NI - REDUCED FOV - POLIMI DATASET/FOLD{0}_Metrics_history_upsampling.hdf5'.format(k + 1)), 'w') as f:
        f.create_dataset('ssim', data=ssim_history)
        f.create_dataset('val_ssim', data=val_ssim_history)
        f.create_dataset('acc', data=acc_history)
        f.create_dataset('val_acc', data=val_acc_history)
        f.create_dataset('dice', data=dice_history)
        f.create_dataset('val_dice', data=val_dice_history)
        f.close
    '''
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
    print('Fold{}'.format(k+1))
    #path_model = K_path_model[k]
    model = torch.load(os.path.join(os.getcwd(), 'RESULTS/DENSE/K FOLD- 2 - NI - REDUCED FOV - POLIMI DATASET/model_unet_attention_checkpoint_{:02d}_fold.pth'.format(k+1)))
    model.eval()
    test_image = []
    predicted_4d = []
    predicted_3d = []
    pred_thr = []
    test_image_list = os.listdir(test_path_image_list[k])
    test_image_list.sort()
    for image in test_image_list:
        t_file_path = os.path.join(test_path_image_list[k], image)
        im = np.array(np.load(t_file_path) / 255.).astype(
            'float32')  # trasforming the image in float 32 with np.max 0< 1
        test_image.append(im)
        im = torch.from_numpy(np.expand_dims(im, 0))
        im = im.to(device, torch.float32)
        im = im.permute(0, 3, 1, 2).float()
        with torch.no_grad():
            prediction = model(im)
        prediction = torch.sigmoid(prediction)
        prediction = prediction.to('cpu').numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        # prediction = torchvision.transforms.ToPILImage()(prediction[0])
        # prediction = np.expand_dims(np.array(prediction),0)
        predicted_4d.append(prediction)
        # deleting the first dimension
        p = prediction[0]  # image in float 32 with np.max 0< 1
        if (p.shape != (256, 256, 1)):
            print('error shape 2')
        predicted_3d.append(p)
        # thresholding the prediction
        a = np.where(p >= 0.5, 1.0, 0)  # image in float 32 with np.max =  1
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
        im = np.array(np.load(t_file_path) / 255.).astype(
            'float32')  # trasforming the image in float 32 with np.max 0< 1
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
    fold = 'Fold{}_'.format(k + 1)
    print('  ')
    print('Visualizing FOLD{} prediction'.format(k + 1))

    print(len(test_image))
    l = 0
    while (l < len(test_image)):
        fig, axs = plt.subplots(4, 4, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(10, 10))
        fig.suptitle('FOLD {0} - Testing'.format(k + 1), fontsize=20)
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
        axs[3].imshow(np.squeeze(np.stack((thr[l] * 255,) * 3, axis=-1)))
        axs[3].set_title('Predicted THR', fontsize=15, loc='center')
        axs[3].set_yticklabels([])
        axs[3].set_xticklabels([])
        axs[4].imshow(test_image[l + 1])
        axs[4].set_yticklabels([])
        axs[4].set_xticklabels([])
        axs[5].imshow(np.squeeze(np.stack((ground_truth[l + 1],) * 3, axis=-1)))
        axs[5].set_yticklabels([])
        axs[5].set_xticklabels([])
        axs[6].imshow(np.squeeze(np.stack((predicted[l + 1],) * 3, axis=-1)))
        axs[6].set_yticklabels([])
        axs[6].set_xticklabels([])
        axs[7].imshow(np.squeeze(np.stack((thr[l + 1] * 255,) * 3, axis=-1)))
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
        axs[11].imshow(np.squeeze(np.stack((thr[l + 2] * 255,) * 3, axis=-1)))
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
        axs[15].imshow(np.squeeze(np.stack((thr[l + 3] * 255,) * 3, axis=-1)))
        axs[15].set_yticklabels([])
        axs[15].set_xticklabels([])
        l = l + 4
        # plt.show()
        fig.savefig(os.path.join(figure_path, fold + 'ImageVSMasks_{0}'.format(l)))

# saving the prediction in png and numpy
for k in range(0, k_fold):
    predicted = K_test_predicted[k]
    thr = K_test_thr_predicted[k]
    fold = 'Fold{}_'.format(k + 1)
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
        image = np.array(image[:, :, 0] * 255., dtype=np.uint8)
        im = Image.fromarray(image, 'L')
        im.save(os.path.join(path_fold_png, 'predicted_{:03d}.png'.format(l)))
    for l, image in enumerate(thr):
        np.save(os.path.join(thr_path_fold_numpy, '{:03d}.npy'.format(l)), np.asarray(image))
        image = np.array(image[:, :, 0] * 255, dtype=np.uint8)
        im = Image.fromarray(image, mode='L')
        im.save(os.path.join(thr_path_fold_png, 'predicted_{:03d}.png'.format(l)))
