import pickle
from matplotlib import pyplot as plt
import numpy as np
import morphLib as mL
fname = "/home/spike/repos/sandbarTool/cresenticBar.p"

with open(fname, 'rb') as fid:
    data = pickle.load(fid)
    
xFRF = np.arange(data.shape[1])
plt.ioff()
for ii in range(0, data.shape[0], 20): # in y
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
    profile = -data[ii]
    bar, trough = mL.findSandBarAndTrough1D(xFRF, profile, plotFname='test.png') #, profileTrend=np.mean(data,axis=0))
    ax2.plot(xFRF, -data[ii], label='{}'.format(ii))
    for bb in bar:
        ax2.plot(xFRF[bb.astype(int)], profile[bb.astype(int)], 'b.')
    for tt in trough:
        ax2.plot(xFRF[tt.astype(int)], profile[bb.astype(int)], 'g.')
    ax2.legend()
    plt.savefig("cresenticBarPlots/profile_{:0004}.png".format(ii))
    plt.close()