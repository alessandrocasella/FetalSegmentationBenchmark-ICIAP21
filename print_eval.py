import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import sklearn
from tqdm import tqdm

models = os.listdir(os.path.join(os.getcwd(), 'RESULTS'))
models.sort()
models = models[:-1]

tDSC = np.empty((0,len(models)), dtype=float)
tSSIM = np.empty((0,len(models)), dtype=float)
tACC = np.empty((0,len(models)), dtype=float)
tSENS = np.empty((0,len(models)), dtype=float)
tIoU = np.empty((0,len(models)), dtype=float)

for i in tqdm(range(1,7)):
	gtPath = os.path.join(os.getcwd(), 'KFOLD', 'Numpy', 'FOLD_{0:02d}'.format(i), 'TEST', 'mask - ground truth')
	gtLen = len(os.listdir(gtPath))

	fDSC = np.zeros((gtLen, len(models)), dtype=float)
	fSSIM = np.zeros((gtLen, len(models)), dtype=float)
	fACC = np.zeros((gtLen, len(models)), dtype=float)
	fSENS = np.zeros((gtLen, len(models)), dtype=float)
	fIoU = np.zeros((gtLen, len(models)), dtype=float)

	gtMasks = np.zeros((gtLen, 256, 256), dtype=bool)
	modelMasks = np.zeros((gtLen, 256, 256), dtype=bool)
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

	tDSC = np.append(tDSC, fDSC, axis=0)
	tSSIM = np.append(tSSIM, fSSIM, axis=0)
	tACC = np.append(tACC, fACC, axis=0)
	tSENS = np.append(tSENS, fSENS, axis=0)
	tIoU = np.append(tIoU, fIoU, axis=0)

	np.savetxt('DSC-Fold{:d}.csv'.format(i), fDSC, fmt='%.5f', delimiter=";", header=";".join(models))
	np.savetxt('SSIM-Fold{:d}.csv'.format(i), fSSIM, fmt='%.5f', delimiter=";", header=";".join(models))
	np.savetxt('ACC-Fold{:d}.csv'.format(i), fACC, fmt='%.5f', delimiter=";", header=";".join(models))
	np.savetxt('SENS-Fold{:d}.csv'.format(i), fSENS, fmt='%.5f', delimiter=";", header=";".join(models))
	np.savetxt('IOU-Fold{:d}.csv'.format(i), fIoU, fmt='%.5f', delimiter=";", header=";".join(models))

meanDSC = np.mean(tDSC, axis=0)
meanSSIM = np.mean(tSSIM, axis=0)
meanACC = np.mean(tACC, axis=0)
meanIOU = np.mean(tIoU, axis=0)

medDSC = np.median(tDSC, axis=0)
stdDSC = np.std(tDSC, axis=0)
medSSIM = np.median(tSSIM, axis=0)
stdSSIM = np.std(tSSIM, axis=0)
medACC = np.median(tACC, axis=0)
stdACC = np.std(tACC, axis=0)
medIOU = np.median(tIoU, axis=0)
stdIOU = np.std(tIoU, axis=0)

np.savetxt('DSC.csv'.format(i), tDSC, fmt='%.5f', delimiter=";", header=";".join(models))
np.savetxt('SSIM.csv'.format(i), tSSIM, fmt='%.5f', delimiter=";", header=";".join(models))
np.savetxt('ACC.csv'.format(i), tACC, fmt='%.5f', delimiter=";", header=";".join(models))
np.savetxt('SENS.csv'.format(i), tSENS, fmt='%.5f', delimiter=";", header=";".join(models))
np.savetxt('IOU.csv'.format(i), tIoU, fmt='%.5f', delimiter=";", header=";".join(models))

format_row = "{:>12}" * (len(models) + 1)
format_row_n = "{:>12}" + "{:>12.4f}" * (len(models))
print(format_row.format("", *models))
print(format_row_n.format('Mean DSC', *meanDSC))
print(format_row_n.format('Std DSC', *stdDSC))
print(format_row_n.format('Mean SSIM', *meanSSIM))
print(format_row_n.format('Std SSIM', *stdSSIM))
print(format_row_n.format('Mean ACC', *meanACC))
print(format_row_n.format('Std ACC', *stdACC))
print(format_row_n.format('Mean IOU', *meanIOU))
print(format_row_n.format('Std IOU', *stdIOU))

print(format_row_n.format('Median DSC', *medDSC))
print(format_row_n.format('Median SSIM', *medSSIM))
print(format_row_n.format('Median ACC', *medACC))
print(format_row_n.format('Median IOU', *medIOU))