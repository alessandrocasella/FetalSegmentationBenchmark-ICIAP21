import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import sklearn
from tqdm import tqdm

for i in tqdm(range(1,7)):
	gtPath = os.path.join(os.getcwd(), 'K FOLD- 2 - NI - REDUCED FOV - POLIMI DATASET', 'Numpy', 'FOLD_{0:02d}'.format(i), 'TEST', 'mask - ground truth')
	gtLen = len(os.listdir(gtPath))
	models = os.listdir(os.path.join(os.getcwd(), 'RESULTS'))

	fDSC = np.zeros((gtLen, len(models)), dtype=np.float)
	fSSIM = np.zeros((gtLen, len(models)), dtype=np.float)
	fACC = np.zeros((gtLen, len(models)), dtype=np.float)
	fSENS = np.zeros((gtLen, len(models)), dtype=np.float)
	fIoU = np.zeros((gtLen, len(models)), dtype=np.float)

	gtMasks = np.zeros((gtLen, 256, 256), dtype=np.bool)
	modelMasks = np.zeros((gtLen, 256, 256), dtype=np.bool)
	for n in range(0, gtLen):
		image = np.load(os.path.join(gtPath,'{0:03d}.npy'.format(n)))
		gtMasks[n, :, :] = (np.squeeze(image) / 34)*True

	for a, model in enumerate(models):
		testPath = os.path.join(os.getcwd(), 'RESULTS', model, 'K FOLD- 2 - NI - REDUCED FOV - POLIMI DATASET', 'PNG Prediction', 'THR', 'Fold{:d}_'.format(i))
		for n in range(gtLen):
			image = Image.open(os.path.join(testPath, 'predicted_{0:03d}.png'.format(n)))
			modelMasks[n, :, :] = (np.array(image) / 255)*True

		for n in range(0, gtLen):
			gt = gtMasks[n, :, :]*1.
			test = modelMasks[n, :, :]*1.
			gt_array = np.asarray(gtMasks[n, :, :])
			test_array = np.asarray(modelMasks[n, :, :])
			C = (((test_array == True) * 2 + (gt_array == True)).reshape(-1, 1) == range(4)).sum(0)
			fDSC[n, a] = 2. * np.logical_and(gt_array, test_array) .sum() / (gt_array.sum() + test_array.sum())
			fSSIM[n, a] = ssim(gt, test, data_range=1.)
			fACC[n, a] = (gt_array == test_array).mean()
			fSENS[n, a] = C[3]/C[1::2].sum()
			fIoU[n, a] = np.logical_and(gt_array, test_array) .sum() / np.logical_or(gt_array, test_array) .sum()
		np.savetxt('DSC-Fold{:d}.csv'.format(i), fDSC, fmt='%.5f', delimiter=";", header=";".join(models))
		np.savetxt('SSIM-Fold{:d}.csv'.format(i), fSSIM, fmt='%.5f', delimiter=";", header=";".join(models))
		np.savetxt('ACC-Fold{:d}.csv'.format(i), fACC, fmt='%.5f', delimiter=";", header=";".join(models))
		np.savetxt('SENS-Fold{:d}.csv'.format(i), fSENS, fmt='%.5f', delimiter=";", header=";".join(models))
		np.savetxt('IOU-Fold{:d}.csv'.format(i), fIoU, fmt='%.5f', delimiter=";", header=";".join(models))


