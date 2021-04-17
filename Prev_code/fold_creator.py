import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
seed = 10
DatasetPath = os.path.join(os.getcwd(), 'REDUCED FOV - POLIMI DATASET', 'PNG')
ImagesPath = os.path.join(DatasetPath, 'Image_Reduced_fov')
MasksPath = os.path.join(DatasetPath, 'Mask_Reduced_fov')

patient_num = 12
p_val = 2
p_test = 2

newDatasetPath = os.path.join(os.getcwd(), 'RESIZED - REDUCED FOV - POLIMI DATASET')

if not os.path.exists(newDatasetPath):
	os.mkdir(newDatasetPath)
	os.mkdir(os.path.join(newDatasetPath, 'PNG'))

for p in os.listdir(ImagesPath):
	os.mkdir(os.path.join(newDatasetPath, 'PNG', p))
	os.mkdir(os.path.join(newDatasetPath, 'PNG', p, 'Images'))
	os.mkdir(os.path.join(newDatasetPath, 'PNG', p, 'Annotations'))
	for i in tqdm(os.listdir(os.path.join(ImagesPath,p))):
		m = 'mask_{}'.format(i[6:])
		image = Image.open(os.path.join(ImagesPath, p, i)).convert('RGB')
		image = image.resize((256,256), Image.BILINEAR)
		image.save(os.path.join(newDatasetPath, 'PNG', p, 'Images', i))
		mask = Image.open(os.path.join(MasksPath, p, m)).convert('L')
		mask = mask.resize((256, 256), Image.NEAREST)
		mask2 = np.array(mask)
		mask2[mask2 == 25] = 34
		mask2[mask2 == 22] = 34
		mask2[mask2 != 34] = 0
		if (np.max(mask2)==0):
			print(os.path.join(MasksPath, p, m))
		mask = Image.fromarray(mask2)
		mask.save(os.path.join(newDatasetPath, 'PNG', p, 'Annotations', m))

p_list = os.listdir(os.path.join(os.getcwd(), 'RESIZED - REDUCED FOV - POLIMI DATASET', 'PNG'))
p_list.sort()

newDatasetPath_fold = os.path.join(os.getcwd(), 'KFOLD')
if not os.path.exists(newDatasetPath_fold):
	os.mkdir(newDatasetPath_fold)


np.random.seed(seed)
np.random.shuffle(p_list)
kfold = KFold(patient_num//p_test)
for i, (train, test) in tqdm(enumerate(kfold.split(p_list))):
	i=i+1
	val = train[-2:]
	train = train[:-2]

	train_file = []
	test_file = []
	val_file = []

	foldPath = os.path.join(newDatasetPath_fold, 'FOLD_{:02d}'.format(i))
	os.mkdir(foldPath)

	#
	os.mkdir(os.path.join(foldPath, 'TRAIN'))
	os.mkdir(os.path.join(foldPath, 'TRAIN', 'image'))
	os.mkdir(os.path.join(foldPath, 'TRAIN', 'mask - ground truth'))
	n = 0
	for p in train:
		file_list = os.listdir(os.path.join(newDatasetPath, 'PNG', p_list[p], 'Images'))
		file_list.sort()
		for f in file_list:
			m = 'mask_{}'.format(f[6:])
			image = Image.open(os.path.join(newDatasetPath, 'PNG', p_list[p], 'Images', f))
			mask = Image.open(os.path.join(newDatasetPath, 'PNG', p_list[p], 'Annotations', m)).convert('L')
			image = np.array(image)
			mask = np.array(mask)
			np.save(os.path.join(foldPath, 'TRAIN', 'image', '{:03d}.npy'.format(n)), image)
			np.save(os.path.join(foldPath, 'TRAIN', 'mask - ground truth', '{:03d}.npy'.format(n)), mask)
			n = n+1

	#
	os.mkdir(os.path.join(foldPath, 'TEST'))
	os.mkdir(os.path.join(foldPath, 'TEST', 'image'))
	os.mkdir(os.path.join(foldPath, 'TEST', 'mask - ground truth'))
	n = 0
	for p in test:
		file_list = os.listdir(os.path.join(newDatasetPath, 'PNG', p_list[p], 'Images'))
		file_list.sort()
		for f in file_list:
			m = 'mask_{}'.format(f[6:])
			image = Image.open(os.path.join(newDatasetPath, 'PNG', p_list[p], 'Images', f))
			mask = Image.open(os.path.join(newDatasetPath, 'PNG', p_list[p], 'Annotations', m))
			image = np.array(image)
			mask = np.array(mask)
			np.save(os.path.join(foldPath, 'TEST', 'image', '{:03d}.npy'.format(n)), image)
			np.save(os.path.join(foldPath, 'TEST', 'mask - ground truth', '{:03d}.npy'.format(n)), mask)
			n = n + 1

	#
	os.mkdir(os.path.join(foldPath, 'VALIDATION'))
	os.mkdir(os.path.join(foldPath, 'VALIDATION', 'image'))
	os.mkdir(os.path.join(foldPath, 'VALIDATION', 'mask - ground truth'))

	n = 0
	for p in val:
		file_list = os.listdir(os.path.join(newDatasetPath, 'PNG', p_list[p], 'Images'))
		file_list.sort()
		for f in file_list:
			m = 'mask_{}'.format(f[6:])
			image = Image.open(os.path.join(newDatasetPath, 'PNG', p_list[p], 'Images', f))
			mask = Image.open(os.path.join(newDatasetPath, 'PNG', p_list[p], 'Annotations', m))
			image = np.array(image)
			mask = np.array(mask)
			np.save(os.path.join(foldPath, 'VALIDATION', 'image', '{:03d}.npy'.format(n)), image)
			np.save(os.path.join(foldPath, 'VALIDATION', 'mask - ground truth', '{:03d}.npy'.format(n)), mask)
			n = n + 1