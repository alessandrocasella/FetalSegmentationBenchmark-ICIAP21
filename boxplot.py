import csv
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
#matplotlib.rcParams['text.usetex'] = True
import numpy as np
import os
import glob
import matplotlib.cm as cm

matplotlib.use('agg')
metrics = ['ACC', 'DSC', 'IOU', 'SENS', 'SSIM']
for fold in range(0,6):
    for metric in metrics:
        filename = os.path.join(os.getcwd(),'{}-Fold{}.csv'.format(metric, fold+1))
        f = open(filename)
        header = f.readline()
        models = header[2:-1].split(';')
        data = np.loadtxt(filename, delimiter=';')
        fig = plt.figure(figsize=(8,4), dpi=150)
        ax = plt.axes()

        fig.tight_layout()
        color = plt.get_cmap("Set1")
        for i, model in enumerate(models):
            plt.boxplot(data[:, i], positions=[i+1], widths=0.25, patch_artist=True,
                        boxprops=dict(facecolor='#CCCCFF', color='k'), medianprops=dict(color='k'), )
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_ylim(0.0, 1.0)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True) # labels along the bottom edge are off
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                       alpha=0.5)
        ax.set_xticklabels(models)
        fig.savefig('{}-Fold{}.svg'.format(metric, fold+1), bbox_inches='tight')
        plt.close(fig)


