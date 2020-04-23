import pickle
from matplotlib import pyplot as plt
import numpy as np
import morphLib as mL
fname = "/home/spike/repos/sandbarTool/cresenticBar.p"

with open(fname, 'rb') as fid:
    data = pickle.load(fid)
    
xFRF = np.arange(data.shape[1])


plt.figure()
ax1 = plt.subplot(211)
mesh = ax1.pcolormesh(data)
plt.colorbar(mesh)
for i in range(data.shape[0]):
    profile = -data[i]
    bar, trough = mL.findSandBarAndTrough1D(xFRF, profile, plotFname='test.png') #, profileTrend=np.mean(data,axis=0))
    for bb in bar:
        ax1.plot(xFRF[bb.astype(int)], i, 'b.')
    for tt in trough:
        ax1.plot(xFRF[tt.astype(int)], i, 'g.')

ax2 = plt.subplot(212)
for i in [1240, 1260, 1280, 1300]: # in y
    profile = -data[i]
    bar, trough = mL.findSandBarAndTrough1D(xFRF, profile, plotFname='test.png') #, profileTrend=np.mean(data,axis=0))
    ax2.plot(xFRF, -data[i], label='{}'.format(i))
    for bb in bar:
        ax1.plot(xFRF[bb.astype(int)], profile[bb.astype(int)], 'b.')
    for tt in trough:
        ax1.plot(xFRF[tt.astype(int)], profile[bb.astype(int)], 'g.')
    ax2.legend()