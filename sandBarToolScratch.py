import sys, pickle
from morphLib import findSandBarAndTrough1D
sys.path.append('/home/spike/repos')
from getdatatestbed import getDataFRF
import datetime as DT
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from matplotlib import patches as mpatches

#######################################################################################################################
meanPickleName = 'meanSandbarProfile.pickle'
profileNumber = 1  # 1097
crossShoreMax = 1000  # how far do we want to look in cross-shore
minPoints4Survey = 10

go = getDataFRF.getObs(DT.datetime(1990, 5, 1), DT.datetime(2019, 11, 1))
survey = go.getBathyTransectFromNC(forceReturnAll=True)
NorthIdx = survey['profileNumber'] == profileNumber
# isolate data for North-side line only
timesNorth_all = survey['time'][:]
xFRFNorth = survey['xFRF'][:]
elevationNorth = survey['elevation'][:]
surveyNumberNorth = survey['surveyNumber'][:]
#now remove striding issure
timesNorth_all = timesNorth_all[NorthIdx]
xFRFNorth = xFRFNorth[NorthIdx]
elevationNorth = elevationNorth[NorthIdx]
surveyNumberNorth = surveyNumberNorth[NorthIdx]
UniqueSurveyNumbersNorth = np.unique(surveyNumberNorth)
## now simplify the dates of the data
xOut, tOut = np.arange(0,1800), [] #np.zeros((UniqueSurveyNumbersNorth.shape[0]))
zOutNorth = np.ma.masked_all((UniqueSurveyNumbersNorth.shape[0], xOut.shape[0])) # initialize masked array all values True
for NN, uniqueSurveyNumber in enumerate(UniqueSurveyNumbersNorth):
     idxpointsPerSurvey = (uniqueSurveyNumber == surveyNumberNorth)
     # print('survey on {} has {} points  with mean elevation of {:.2f}'.format(timesNorth_all[idxpointsPerSurvey][0],
     #                                                                          np.sum(idxpointsPerSurvey),
     #                                                                          elevationNorth[idxpointsPerSurvey].mean()))
     if idxpointsPerSurvey.sum() > minPoints4Survey:
        tOut.append(timesNorth_all[idxpointsPerSurvey][0])
        tempf = interpolate.interp1d(xFRFNorth[idxpointsPerSurvey], elevationNorth[idxpointsPerSurvey],
                                     bounds_error=False, kind='linear', fill_value='nan')
        zOutNorth[NN] = tempf(xOut)

fig = plt.figure(figsize=(8,10))
ax1 = plt.subplot()
mesh = ax1.pcolormesh(xOut[:800], tOut, zOutNorth[:,:800])# , cmap='RdBu', norm=(MidpointNormalize(midpoint=0)))
plt.colorbar(mesh)
# notice bar migration patterns


meanProfile = np.nanmean(zOutNorth, axis=0)
out = {'meanProfile': meanProfile,
       'xOut': xOut}
if (go.d2 - go.d1).days/365 > 30:
    with open(meanPickleName, 'wb') as fid:
        pickle.dump(out, fid, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(meanPickleName, 'rb') as fid:
        out = pickle.load(fid)
        meanProfile = out.pop('meanProfile')
        xOut = out.pop('xOut')

for tt in range(zOutNorth.shape[0]):
    bathy = zOutNorth[tt][~np.isnan(zOutNorth[tt])]
    bathyX = xOut[~np.isnan(zOutNorth[tt])]
    if bathy[~np.isnan(bathy)].any():
        fname = "/home/spike/repos/sandbarTool/sandBarImages/BarId_{}.png".format(tOut[tt].strftime("%Y%m%d"))
        xFRFbar, xFRFtrough = findSandBarAndTrough1D(bathyX, bathy, plotFname=fname, smoothLengthScale=50, profileTrend=meanProfile[np.in1d(xOut, bathyX)])
        if xFRFbar is not None:
            for sandbarX in xFRFbar:
                ax1.plot(sandbarX, tOut[tt], 'ro', label='bar')
        if xFRFtrough is not None:
            for troughX in xFRFtrough:
                ax1.plot(troughX, tOut[tt], 'bd', label='trough')
        barPatch = mpatches.Patch(color='red', label='sandbar')
        troughPatch = mpatches.Patch(color='blue', label='trough')
plt.ion()
# barLine = mlines.Line2D([],[],color='blue', marker='d', ms=15, label='trough')
# troughLine  = mlines.Line2D([],[],color='red', marker='o', ms=15, label='sandbar')
plt.legend([barPatch, troughPatch], ['SandBar', 'Trough'])
plt.xlim([0, 800])
fig.savefig('TotalSandBarPosition_south.png')
print('done')