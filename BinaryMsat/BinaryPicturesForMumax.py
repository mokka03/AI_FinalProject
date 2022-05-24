import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import savemat, loadmat

'''Directories'''
basedir = 'BinaryPictures_WD_alpha7p9_CloseOutput/'
plotdir = 'plots/' + basedir
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)
savedir = 'models/' + basedir
if not os.path.isdir(savedir):
    os.makedirs(savedir)    

MsatWhole = loadmat("matlab/Msat_s_WD_alpha7p9_CloseOutput.mat").get('Msat')
MsatWhole = np.flip(MsatWhole,0)
Msat = MsatWhole[166:474,6:314]

# plt.matshow(Msat)
# plt.show()

'''Binary pictures'''
# print(np.min(Msat))
# print(np.max(Msat))
Msat_difference = Msat-np.min(Msat)
Msat_difference_norm = (Msat_difference)/np.max(Msat_difference)

Msat_binary = np.ones_like(Msat)
damping_with = 0
for i in range(damping_with-1,Msat.shape[0]-damping_with):
    for j in range(damping_with-1,Msat.shape[1]-damping_with):
        if Msat_difference_norm[i,j] > random.uniform(0,1):
            Msat_binary[i,j] = 0

MsatWhole_binary = np.ones_like(MsatWhole)
MsatWhole_binary[166:474,6:314] = Msat_binary
# print(MsatWhole_binary.shape)
# plt.matshow(MsatWhole_binary)
# plt.show()

plt.imsave(plotdir+'%d.png' % (0), MsatWhole_binary, cmap=cm.gray)

savemat("matlab/MsatBinary_ForTest.mat", {"Msat_binary": MsatWhole_binary})
