import peakutils, sys
sys.path.append('/home/spike/repos')
from getdatatestbed import getDataFRF
import datetime as DT
import numpy as np
from matplotlib import pyplot as plt
from testbedutils import sblib as sb
import netCDF4 as nc
from scipy import interpolate


go = getDataFRF.getObs(DT.datetime(1983, 6, 25), DT.datetime(1992, 11, 1))
survey = go.getBathyTransectFromNC(forceReturnAll=True)
# what are my north and south lines of interest
NorthIdx = survey['profileNumber'] == 1097
southIdx = survey['profileNumber'] == 1
crossShoreMax = 1000   # how far do we want to look in cross-shore

# isolate data for North-side line only
timesNorth_all = survey['time'][NorthIdx]
xFRFNorth = survey['xFRF'][NorthIdx]
elevationNorth = survey['elevation'][NorthIdx]
surveyNumberNorth = survey['surveyNumber'][NorthIdx]
UniqueSurveyNumbersNorth = np.unique(surveyNumberNorth)
## now simplify the dates of the data
# epochTimesNorth, timesNorth = [], []
# for u in UniqueSurveyNumbersNorth:
#     temp = (roundtime(timesNorth_all[u == surveyNumberNorth], roundTo=60*60*24))
#     epochTimesNorth.extend(nc.date2num(temp, 'seconds since 1970-01-01'))
#     timesNorth.extend(temp)

# # now grid data in cross shore
#
#
# tOut = np.unique(epochTimesNorth)
# #how many unique surveys do i have??
# zOutNorth = np.ma.masked_all((tOut.shape[0], xOut.shape[0])) # initialize masked array all values True
# for ss, surveyTime in enumerate(tOut):                       # ss is index of value surveyNum
#     idxCurrentSurveyPoints = (epochTimesNorth == surveyTime)
#     print('{} survey points in survey on {}'.format(np.sum(idxCurrentSurveyPoints), nc.num2date(surveyTime, 'seconds since 1970-01-01')))
#     print('Mean elevations {:.2f}'.format(elevationNorth[idxCurrentSurveyPoints].mean()))
#     if idxCurrentSurveyPoints.sum() > minPoints4Survey:    # sometimes there's not enough points on a day (i don't know why)
#         tempf = interpolate.interp1d(xFRFNorth[idxCurrentSurveyPoints], elevationNorth[idxCurrentSurveyPoints],
#                                      bounds_error=False, kind='linear', fill_value='nan')
#         zOutNorth[ss] = tempf(xOut)

## try again using unique survey numbers instead of dates (could remove above portion)
minPoints4Survey = 10
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

fig = plt.figure(figsize=(18,6))
ax1 = plt.subplot()
mesh = ax1.pcolormesh(xOut[:800], tOut, zOutNorth[:,:800])# , cmap='RdBu', norm=(MidpointNormalize(midpoint=0)))
plt.colorbar(mesh)
# notice bar migration patterns

def findSandBarAndTrough1D(xFRF, profile, plotFname=None, **kwargs):
    """ Finds multiple bars on a single profile, will also plot QA/QC plot.  The process begins by finding a trend (see
    keyword args for details) removing it from the data.  Then smooths the profile and trend and finds peaks on that
    detrended line.  It removes portions of the line that are above mean water level (Default=0), that are towards the
    offshore portion of the domain (default = 10%), and sandbars that are within x number of cells (default = 150).  It
    will return the indices of the bars and troughs of interest.

    Assumptions:
        profiles are shoreward to seaward, left to right

    Args:
        xFRF: cross-shore coordinates to describe profile values
        profile: bathymetry values (positive up)
        plotFname: if not None, will generate a QA/QC plot that (default=None)

    Keyword Args
        'waterLevel': finds only bars below this value (Default = 0)
        'trendOrder': removes trend of order (default = 3)
        'deepWaterPercentile': remove sandbars found in the last x percent of the domain (default = 0.9)
        'minSandBarSeparation':  separation in peaks to be found in cells(Default=150 Cells)
        'meanProfile': input a trend to remove data from

    Returns:
        idxPeak (list): indices of sandbar crest positions
        idxTrough (list): indices of trough positions

    """
    waterLevel = kwargs.get('waterLevel', 0)
    polyFit = kwargs.get('trendOrder', 3)
    deepwaterPercentile = kwargs.get('deepWaterPercentile', .9)
    minSandBarSeparation = kwargs.get('minSandBarSeparation', 100)
    smoothLine = kwargs.get('smoothLengthScale', 10)
    profileTrend = kwargs.get('profileTrend',
                              peakutils.baseline(profile, deg=polyFit))  # find the profiles general trend
    ################################################################
    # start working on data

    profile_smooth = sb.running_mean(profile, smoothLine)                                                 # first smooth the data to clean out noise
    xFRF_smooth = sb.running_mean(xFRF, smoothLine)                                                       # smooth the  cross-shore coordinates
    profileTrend_smooth = sb.running_mean(profileTrend, smoothLine)

    findPeaksOnThisLine = profile_smooth - profileTrend_smooth                                            # remove trend
    findPeaksOnThisLine[profile_smooth >= waterLevel] = np.nan                                            # find only the one's that are below the water level
    if not np.isnan(findPeaksOnThisLine).all():                                                           # if the whole thing is nans' don't do anything
        ###################################
        # 1. find sandbar first cut peaks #
        ###################################
        peakIdx = peakutils.indexes(findPeaksOnThisLine[~np.isnan(findPeaksOnThisLine)], min_dist=minSandBarSeparation)
        peakIdx = np.argmin(np.isnan(findPeaksOnThisLine)) + peakIdx               # SHIFT back to the original baseline (with nans)
        peakIdx = peakIdx[peakIdx < len(profile_smooth)*deepwaterPercentile]       # remove any peaks found oceanward of 90% of line
        ############################################
        # 1a. refine peaks to point of inflection  #
        ############################################
        # make sure now that each peak is actually a local maximum by looking at the slope at each peakIDX and finding
        # the nearest zero slope towards shore
        peakIdxNew, troughIdx = [], []
        for pp, peak in enumerate(peakIdx):
            # find point most shoreward of point with slope greater than zero (add additional point to find point before)
            dElevation = np.diff(profile_smooth[:peak])                   # take the derivative of the smoothed profile
            if (dElevation > 0).any():                                    # are any of the slopes positive
                peakIdxNew.append(np.argwhere(dElevation > 0).max() + 1)  # find the one that is most shoreward

        #######################
        # 2. now find troughs #
        #######################
        for peak in peakIdxNew:
            if profile[np.argmin(profile_smooth[:peak.squeeze()])] < profile_smooth[peak]:      # check that its shallower than the sand bar
                troughIdx.append(np.argmin(profile_smooth[:peak.squeeze()]))
        ########################################################################################
        # 3. check to see if peaks are really peaks, find local maximum between bar and trough #
        ########################################################################################
        # if np.size(peakIdx) == np.size(troughIdx):                     # we have same number of peaks and troughs
        #     for pp, peak in enumerate(peakIdx):
        #         peakIdx[pp] = troughIdx[pp] + np.argmax(profile_smooth[troughIdx[pp]:peak])
        # else:                                                          # we found a peak, but no troughs to the sand bar
        #     # now reiterate the find method, but start from found peak move towards shore
        #     for peak in peakIdx:
        #         findPeaksOnThisLine[range(len(findPeaksOnThisLine)) > peak] = np.nan
        #####################################
        # Last: plot for QA/QC              #
        #####################################
        if plotFname is not None:            # now plot if interested
            plt.figure(figsize=(8,5))
            plt.title(DT.datetime.strptime(plotFname.split('_')[-1].split('.')[0], '%Y%m%d'))
            plt.plot(xFRF, profile, '.', label='Raw')
            plt.plot(xFRF_smooth, profile_smooth, 'c.', ms=2, label='smoothed')
            plt.plot(xFRF_smooth, findPeaksOnThisLine, label='Find peaks on this line')
            plt.plot(xFRF_smooth, profileTrend_smooth, label='Trend')
            plt.plot([0, len(profile)], [0,0], 'k--')
            plt.plot(xFRF_smooth[peakIdx], profile_smooth[peakIdx], 'r.', ms=5, label='Inital bar Location')
            plt.plot(xFRF_smooth[peakIdx], findPeaksOnThisLine[peakIdx], 'r.', ms=5)
            plt.plot(xFRF_smooth[peakIdxNew], profile_smooth[peakIdxNew], 'rd', ms=7, label='Refined Bar location')
            plt.plot(xFRF_smooth[troughIdx], profile_smooth[troughIdx], 'bo', ms=6, label='Trough Location')
            plt.legend(loc='upper right')
            plt.savefig(plotFname)
            plt.close()

    else:
        peakIdx, troughIdx = None, None

    return peakIdx, troughIdx

# def findSandBarAndTroughIterative(profile, plotFname=None, **kwargs):
#     """
#     use a smoothed profile start at some long length scale, remove that trend, find profile on that
#     iterate in a way that shortens the length scale until the sand bar picks out the same location
#     if multiple sand bars uses offshore identified one, then shortens profile from that point to shore
#     """
meanProfile = np.nanmean(zOutNorth, axis=0)

for tt in range(zOutNorth.shape[0]):
    bathy = zOutNorth[tt][~np.isnan(zOutNorth[tt])]
    bathyX = xOut[~np.isnan(zOutNorth[tt])]
    if bathy[~np.isnan(bathy)].any():
        fname = "/home/spike/repos/sandbarTool/sandBarImages/BarId_{}.png".format(tOut[tt].strftime("%Y%m%d"))
        barIdx, troughIdx = findSandBarAndTrough1D(bathyX, bathy, plotFname=fname, profileTrend=meanProfile[np.in1d(xOut, bathyX)])
        if barIdx is not None:
            for sandbar in barIdx:
                ax1.plot(xOut[sandbar], tOut[tt], 'rx')
        if troughIdx is not None:
            for trough in troughIdx:
                ax1.plot(xOut[trough], tOut[tt], 'bd')

fig.savefig('TotalSandBarPosition_North.png')
print('done')