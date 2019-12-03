import sys, pickle, os
sys.path.append('/home/spike/repos')
from getdatatestbed import getDataFRF
import datetime as DT
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from matplotlib import patches as mpatches
from pathlib import Path
from morphLib import findSandBarAndTrough1D

xBounds = [0,800]
yBounds = [0, 1600]
############################################################################################################################
gm = getDataFRF.getDataTestBed(DT.datetime(2000,3,3), DT.datetime(2019,4,3))
data = gm.getBathyIntegratedTransect(xbounds=xBounds, ybounds=yBounds, forceReturnAll=True)

with open('/home/spike/repos/sandbarTool/meanSandbarProfile.pickle', 'rb') as fid:
    out = pickle.load(fid)
    # interpolate on the way out to now
    meanProfile = np.interp(data['xFRF'], out.pop('xOut'), out.pop('meanProfile'))
############################################################################################################################
for tt in range(data['time'].shape[0]):
    print('working on survey {}\n'.format(data['time'][tt].strftime('%Y%m%d')))
    yLocs, xPeak, xTrough = [], [], []
    for pp in range(data['elevation'][tt].shape[0]):
        profileIn = data['elevation'][tt, pp]
        xIn = data['xFRF']
        plotFname = "2DbathyWorkFlow/{}/profile_y{:03d}.png".format(data['time'][tt].strftime('%Y%m%d'), int(data['yFRF'][pp]))

        try:
            peaks, troughs = findSandBarAndTrough1D(xFRF=xIn,profile=profileIn, profileTrend = meanProfile,
                                                minSandBarSeparation=100/np.median(np.diff(xIn)),
                                                deepWaterCoordCutoff=600, plotFname=plotFname)
        except FileNotFoundError:
            path = Path(os.path.dirname(plotFname))
            path.mkdir(parents=True, exist_ok=True)
            peaks, troughs = findSandBarAndTrough1D(xFRF=xIn,profile=profileIn, profileTrend = meanProfile,
                                                minSandBarSeparation=100/np.median(np.diff(xIn)),
                                                deepWaterCoordCutoff=600, plotFname=plotFname)
        xPeak.append(peaks)
        xTrough.append(troughs)
        yLocs.append(data['yFRF'][pp])
        # print('done Profile {}'.format(data['yFRF'][pp]))
    ############################################################################################################################
    plt.ioff()

    plt.figure();
    plt.pcolormesh(data['xFRF'], data['yFRF'], data['elevation'][tt])
    for yy, peaks in enumerate(xPeak):
        if peaks is not None:
            plt.plot(peaks, np.tile(yLocs[yy], len(peaks)), 'bD', ms=2)
    for yy, troughs in enumerate(xTrough):
        if troughs is not None:
            plt.plot(troughs, np.tile(yLocs[yy], len(troughs)), 'go', ms=1)

    barPatch = mpatches.Patch(color='blue', label='sandbar')
    troughPatch = mpatches.Patch(color='green', label='trough')
    plt.legend([barPatch, troughPatch], ['SandBar', 'Trough'])

    plt.suptitle('Bars and Troughs from {}'.format(data['time'][tt].strftime('%Y-%m-%d')))
    plt.ylabel('yFRF [m]')
    plt.xlabel('xFRF [m]')
    saveFname = os.path.join("2DbathyWorkFlow", 'all_locs_{}.png'.format((data['time'][tt].strftime('%Y-%m-%d'))))
    plt.savefig(saveFname)
    plt.close()