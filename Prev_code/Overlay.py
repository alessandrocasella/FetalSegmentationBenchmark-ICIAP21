

import os
import cv2
from matplotlib import pyplot as plt

prediction_folder = os.path.join(os.getcwd(), 'RESULT_2/Model 1/NI - REDUCED FOV/PNG Prediction/THR')
print(prediction_folder)

prediction_subfolders = [ f.path for f in os.scandir(prediction_folder) if f.is_dir() ]
prediction_subfolders.sort()
print(prediction_subfolders)

K_prediction_list = []
for path in prediction_subfolders:
    fold = path
    print(fold)
    prediction_list = os.listdir(fold)
    prediction_list.sort()
    print(len(prediction_list))
    K_prediction_list.append(prediction_list)
print(len(K_prediction_list))

png_folder = os.path.join(os.getcwd(), 'K FOLD- 2 - NI - REDUCED FOV - POLIMI DATASET/PNG')
print(png_folder)
png_subfolders =  [ f.path for f in os.scandir(png_folder) if f.is_dir() ]
png_subfolders.sort()
print(png_subfolders)

K_image_list = []
for path in png_subfolders:
    image_folder = os.path.join(path, 'TEST/image')
    print(image_folder)
    image_list = os.listdir(image_folder)
    image_list.sort()
    print(len(image_list))
    K_image_list.append((image_list))
print(len(K_image_list))

K_ground_list = []
for path in png_subfolders:
    ground_folder = os.path.join(path, 'TEST/mask - ground truth')
    print(ground_folder)
    ground_list = os.listdir(ground_folder)
    ground_list.sort()
    print(len(ground_list))
    K_ground_list.append(ground_list)
print(len(K_ground_list))

overlay_path = os.path.join(os.getcwd(), 'Overlay')
try:
    os.mkdir(overlay_path)
except:
    pass

model_path = os.path.join(overlay_path, 'Model1')
try:
    os.mkdir(model_path)
except:
    pass

figure_path = os.path.join(model_path, 'NI - REDUCED - FOV')
try:
    os.mkdir(figure_path)
except:
    pass

for x, prediction_list in enumerate(K_prediction_list):
    if(x == 1):
        break
    print('FOLD {}'.format(x+1))
    prediction_folder = prediction_subfolders[x]
    image_folder = os.path.join(png_subfolders[x], 'TEST/image')
    image_list = K_image_list[x]
    ground_folder = os.path.join(png_subfolders[x], 'TEST/mask - ground truth')
    ground_list = K_ground_list[x]
    l = 0
    fold = 'FOLD {}'.format(x+1)
    while (l < len(image_list)):
        mas = cv2.imread(prediction_subfolders[x] + '/' + prediction_list[l])
        edges = cv2.Canny(mas, 50, 200)  # canny edge detector

        gr = cv2.imread(ground_folder + '/' + ground_list[l])
        edges_gr = cv2.Canny(gr, 50, 200)

        img = cv2.imread(image_folder + '/' + image_list[l])  # creat RGB image from grayscale
        img2 = img.copy()
        img2[edges_gr == 255] = [0, 255, 0]  # turn edges to red
        img3 = img.copy()
        img3[edges == 255] = [0, 0, 255]  # turn edges to red
        print(prediction_list[l])
        print(ground_list[l])
        print(image_list[l])

        fig, axs = plt.subplots(5, 3, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(10, 10))
        fig.suptitle('FOLD {0}'.format(x+1), fontsize=20)
        for row in range(5):
            l = l + 1
            print(l)
            axs[row, 0].imshow(img)
            if (row == 0):
                axs[row, 0].set_title('Image', fontsize=15, loc='center')
            axs[row, 0].set_yticklabels([])
            axs[row, 0].set_xticklabels([])
            axs[row, 1].imshow(img2)
            if (row == 0):
                axs[row, 1].set_title('Ground Truth', fontsize=15, loc='center')
            axs[row, 1].set_yticklabels([])
            axs[row, 1].set_xticklabels([])
            if (row == 0):
                axs[row, 2].set_title('Predicted', fontsize=15, loc='center')
            axs[row, 2].set_yticklabels([])
            axs[row, 2].set_xticklabels([])
            axs[row, 2].imshow(img3)
            if (l < len(image_list)):
                mas = cv2.imread(prediction_subfolders[x] + '/' + prediction_list[l])
                print(prediction_list[l])
                edges = cv2.Canny(mas, 50, 200)  # canny edge detector

                gr = cv2.imread(ground_folder + '/' + ground_list[l])
                edges_gr = cv2.Canny(gr, 50, 200)
                print(ground_list[l])
                print(image_list[l])
                img = cv2.imread(image_folder + '/' + image_list[l])  # creat RGB image from grayscale
                img2 = img.copy()
                img2[edges_gr == 255] = [0, 255, 0]  # turn edges to red
                img3 = img.copy()
                img3[edges == 255] = [0, 0, 255]  # turn edges to red
        plt.show()
        #fig.savefig(os.path.join(figure_path, fold + '_Overlay_{0}'.format(l)))



        '''fig, axs = plt.subplots(5,3, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(20,20))
        fig.suptitle('FOLD {}'.format(x+1), fontsize=20)
        mas = cv2.imread(prediction_subfolders[x] + '/' + prediction_list[l])
        edges = cv2.Canny(mas,50,200)   # canny edge detector

        gr = cv2.imread(ground_folder + '/' + ground_list[l])
        edges_gr = cv2.Canny(gr, 50, 200)

        img = cv2.imread(image_folder + '/' + image_list[l])  # creat RGB image from grayscale
        img2 = img.copy()
        img2[edges_gr == 255] = [255, 0, 0]  # turn edges to red
        img3 = img.copy()
        img3[edges == 255] = [255, 0, 0]  # turn edges to red

        for row in range(5):
            l = l+1
            axs[row, 0].imshow(img)
            if (row == 0):
                axs[row, 0].set_title('Image', fontsize=15, loc='center')
            axs[row, 0].set_yticklabels([])
            axs[row, 0].set_xticklabels([])
            axs[row, 1].imshow(img2)
            if (row == 0):
                axs[row, 1].set_title('Ground Truth', fontsize=15, loc='center')
            axs[row, 1].set_yticklabels([])
            axs[row, 1].set_xticklabels([])
            if (row == 0):
                axs[row, 2].set_title('Predicted', fontsize=15, loc='center')
            axs[row, 2].set_yticklabels([])
            axs[row, 2].set_xticklabels([])
            axs[row, 2].imshow(img3)
            mas = cv2.imread(prediction_subfolders[x] + '/' + prediction_list[l])
            edges = cv2.Canny(mas, 50, 200)  # canny edge detector

            gr = cv2.imread(ground_folder + '/' + ground_list[l])
            edges_gr = cv2.Canny(gr, 50, 200)

            img = cv2.imread(image_folder + '/' + image_list[l])  # creat RGB image from grayscale
            img2 = img.copy()
            img2[edges_gr == 255] = [255, 0, 0]  # turn edges to red
            img3 = img.copy()
            img3[edges == 255] = [255, 0, 0]  # turn edges to red
        plt.show()







        plt.subplot(131), plt.imshow(img)
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(img2)
        plt.title('Edge Highlighted'), plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.imshow(img3)
        plt.title('Edge Highlighted'), plt.xticks([]), plt.yticks([])





        mas = cv2.imread(prediction_subfolders[x] + '/' + prediction_list[l+1])
        edges = cv2.Canny(mas,50,200)   # canny edge detector

        gr = cv2.imread(ground_folder + '/' + ground_list[l+1])
        edges_gr = cv2.Canny(gr, 50, 200)

        img = cv2.imread(image_folder + '/' + image_list[l+1])  # creat RGB image from grayscale
        img2 = img.copy()
        img2[edges_gr == 255] = [255, 0, 0]  # turn edges to red
        img3 = img.copy()
        img3[edges == 255] = [255, 0, 0]  # turn edges to red

        mas = cv2.imread(prediction_subfolders[x] + '/' + prediction_list[l + 3])
        edges = cv2.Canny(mas, 50, 200)  # canny edge detector

        gr = cv2.imread(ground_folder + '/' + ground_list[l + 3])
        edges_gr = cv2.Canny(gr, 50, 200)

        img = cv2.imread(image_folder + '/' + image_list[l + 3])  # creat RGB image from grayscale
        img2 = img.copy()
        img2[edges_gr == 255] = [255, 0, 0]  # turn edges to red
        img3 = img.copy()
        img3[edges == 255] = [255, 0, 0]  # turn edges to red

'''

