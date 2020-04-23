import sys, pickle, os
sys.path.append('/home/spike/repos')
from getdatatestbed import getDataFRF
import datetime as DT
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from matplotlib import patches as mpatches
from pathlib import Path
import morphLib
from shortestPath import shortestPath

xBounds = [0,800]
yBounds = [0, 1600]
############################################################################################################################
gm = getDataFRF.getDataTestBed(DT.datetime(2018,3,3), DT.datetime(2018,3,5))
data = gm.getBathyIntegratedTransect(xbounds=xBounds, ybounds=yBounds)

with open('/home/spike/repos/sandbarTool/meanSandbarProfile.pickle', 'rb') as fid:
    out = pickle.load(fid)
    # interpolate on the way out to now
    meanProfile = np.interp(data['xFRF'], out.pop('xOut'), out.pop('meanProfile'))
############################################################################################################################
morphLib.run1DsandBarin2D(time=data['time'], xFRF=data['xFRF'], yFRF=data['yFRF'], elevation=data['elevation'])
############################################################################################################################
surface = np.abs(np.diff(data['elevation']))
vals, ys = shortestPath(surface)
surface2 = np.abs(np.diff(np.diff(data['elevation'])))
vals2, ys2 = shortestPath(-surface2)
plt.figure();
plt.pcolormesh(vals2)
plt.colorbar()
plt.plot(np.diff(data['elevation'][200,:]))
############################################################################################################################
plt.ion()
plt.figure()
ax1 = plt.subplot(121)
mesh1 = ax1.pcolormesh(data['xFRF'], data['yFRF'], data['elevation'])
ax1.plot(data['xFRF'][ys.astype(int)], data['yFRF'], 'C1.', ms=1)
for yy, peaks in enumerate(xPeak):
    if peaks is not None:
        ax1.plot(peaks, np.tile(yLocs[yy], len(peaks)), 'bD', ms=2)
for yy, troughs in enumerate(xTrough):
    if troughs is not None:
        ax1.plot(troughs, np.tile(yLocs[yy], len(troughs)), 'go', ms=1)

barPatch = mpatches.Patch(color='blue', label='sandbar')
troughPatch = mpatches.Patch(color='green', label='trough')
shortestPatch = mpatches.Patch(color='C1', label='Shortest Path')
ax1.legend([barPatch, troughPatch, shortestPatch], ['SandBar', 'Trough', 'Shortest Path'])
plt.suptitle('Bars and Troughs from {}'.format(data['time'].strftime('%Y-%m-%d')))
ax1.set_ylabel('yFRF [m]')
ax1.set_xlabel('xFRF [m]')
plt.colorbar(mesh1)

ax2 = plt.subplot(122)
mesh2 = ax2.pcolormesh(data['xFRF'], data['yFRF'], vals)
ax2.plot(data['xFRF'][ys.astype(int)], data['yFRF'], 'C1.', ms=1)
for yy, peaks in enumerate(xPeak):
    if peaks is not None:
        ax2.plot(peaks, np.tile(yLocs[yy], len(peaks)), 'bD', ms=2)
for yy, troughs in enumerate(xTrough):
    if troughs is not None:
        ax2.plot(troughs, np.tile(yLocs[yy], len(troughs)), 'go', ms=1)
plt.colorbar(mesh2)





saveFname = os.path.join("2DbathyWorkFlow", 'all_locs_{}.png'.format((data['time'][tt].strftime('%Y-%m-%d'))))
plt.savefig(saveFname)
plt.close()