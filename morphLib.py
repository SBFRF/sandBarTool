import datetime as DT
import numpy as np
import peakutils,os
from matplotlib import pyplot as plt
from scipy import signal


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
        'deepWaterCoordCutoff': a cutoff to say sandbars can't exist beyond this cross-shore coordinate (value must in
                xFRF coordinates)
        'verbose': if on, will print more information to screen (default = False)

    Returns:
        Peak (list): cross-shore location of sandbar crest positions
        Trough (list): cross-shore location of trough positions

    """
    verbose = kwargs.get('verbose', False)
    waterLevel = kwargs.get('waterLevel', 0)
    polyFit = kwargs.get('trendOrder', 3)
    deepwaterPercentile = kwargs.get('deepWaterPercentile', None)
    deepwaterCoordinate = kwargs.get('deepWaterCoordCutoff', 600)
    minSandBarSeparation = kwargs.get('minSandBarSeparation', 100)
    smoothLine = kwargs.get('smoothLengthScale', 10)
    profileTrend = kwargs.get('profileTrend',
                              peakutils.baseline(profile, deg=polyFit))  # find the profiles general trend
    ################################################################
    # start working on data
    assert np.size(profile) == np.size(profileTrend) == np.size(xFRF), 'ProfileTrend must be same size as input profile data, and xFRF'
    #profile_smooth = sb.running_mean(profile, smoothLine)                                                 # first smooth the data to clean out noise
    #xFRF_smooth = sb.running_mean(xFRF, smoothLine)                                                       # smooth the  cross-shore coordinates
    #profileTrend_smooth = sb.running_mean(profileTrend, smoothLine)
    filterDegree = 2   # int(np.ceil(smoothLine/4)*2-1)  # always round to odd number (half of smoothline)
    smoothLine = int(np.ceil(smoothLine/2)*2+1)    # always round to odd number (up from smoothline)
    profile_smooth = signal.savgol_filter(profile, smoothLine, filterDegree)
    xFRF_smooth = signal.savgol_filter(xFRF, smoothLine, filterDegree)
    profileTrend_smooth = signal.savgol_filter(profileTrend, smoothLine, filterDegree)
    ### check profile Trend to make sure
    findPeaksOnThisLine = profile_smooth - profileTrend_smooth                                            # remove trend
    findPeaksOnThisLine[profile_smooth >= waterLevel] = np.nan                                            # find only the one's that are below the water level
    if not np.isnan(findPeaksOnThisLine).all():                                                           # if the whole thing is nans' don't do anything
        ###################################
        # 1. find sandbar first cut peaks #
        ###################################
        peakIdx = peakutils.indexes(findPeaksOnThisLine[~np.isnan(findPeaksOnThisLine)], min_dist=minSandBarSeparation)
        # peakIdx = peakutils.indexes(findPeaksOnThisLine[~np.isnan(findPeaksOnThisLine)], min_dist=minSandBarSeparation,
        #                    thres_abs=True, thres=0.25)
        peakIdx = np.argmin(np.isnan(findPeaksOnThisLine)) + peakIdx               # SHIFT back to the original baseline (with nans)
        if deepwaterPercentile is not None:
            peakIdx = peakIdx[peakIdx < len(profile_smooth)*deepwaterPercentile]       # remove any peaks found oceanward of 90% of line
        else:
            peakIdx = peakIdx[xFRF_smooth[peakIdx] < deepwaterCoordinate]

        peakIdx = peakIdx[::-1]  # flip peaks to move offshore to onshore
        ############################################
        # 1a. refine peaks to point of inflection  #
        ############################################
        # make sure now that each peak is actually a local maximum by looking at the slope at each peakIDX and finding
        # the nearest zero slope towards shore
        peakIdxNew, troughIdx = [], []
        shorelineIDX = np.nanargmin(np.abs(profile_smooth))  # identify shoreline on smoothed profile
        for pp, peak in enumerate(peakIdx):
            # find point most shoreward of point with slope greater than zero (add additional point to find point before)
            dElevation = np.diff(profile_smooth[shorelineIDX:peak])                   # take the derivative of the smoothed profile
            if (dElevation > 0).any():                                    # are any of the slopes positive
                idxDiff = np.argwhere(dElevation > 0).squeeze()   # find all values that have positive slope
                idxMax = np.max(idxDiff) + shorelineIDX              # find max cross-shore location add shoreline location
                peakIdxNew.append(idxMax + 1)                     # add one to find before point of inflection
        peakIdxNew = np.unique(peakIdxNew)
        #######################
        # 2. now find troughs #
        #######################
        for peak in peakIdxNew:
            if profile_smooth[np.argmin(profile_smooth[:peak]).squeeze()] <  profile_smooth[peak]:      # check that its shallower than the sand bar
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
        plt.ioff()                           # turn off plot visible
        if plotFname is not None:            # now plot if interested
            if verbose: print("plotting {}".format(plotFname))
            plt.figure(figsize=(8, 5))
            try:
                plt.suptitle(DT.datetime.strptime(plotFname.split('_')[-1].split('.')[0], '%Y%m%d'))
            except ValueError: # happens when there's not a date in the filename
                plt.suptitle(os.path.basename(plotFname))
            plt.plot(xFRF, profile, 'C1.', label='Raw')
            plt.plot(xFRF_smooth, profile_smooth, 'c.', ms=2, label='smoothed')
            plt.plot(xFRF_smooth, findPeaksOnThisLine, label='Find peaks on this line')
            plt.plot(xFRF_smooth, profileTrend_smooth, label='Trend')
            plt.plot([0, len(profile)], [0,0], 'k--')
            if np.size(peakIdx) > 0:
                plt.plot(xFRF_smooth[peakIdx], profile_smooth[peakIdx], 'r.', ms=5, label='Inital bar Location')
                plt.plot(xFRF_smooth[peakIdx], findPeaksOnThisLine[peakIdx], 'r.', ms=5)
            if np.size(peakIdxNew) >0:
                plt.plot(xFRF_smooth[peakIdxNew], profile_smooth[peakIdxNew], 'rd', ms=7, label='Refined Bar location')
            if np.size(troughIdx) >0:
                plt.plot(xFRF_smooth[troughIdx], profile_smooth[troughIdx], 'bo', ms=6, label='Trough Location')
            plt.legend(loc='upper right')
            plt.ylabel('Elevation NAVD88 [m]')
            plt.xlabel('cross-shore location [m]')
            plt.savefig(plotFname)
            plt.close()
        if np.size(peakIdxNew) > 0 and np.size(troughIdx) > 0:
            return xFRF_smooth[peakIdxNew], xFRF_smooth[troughIdx]
        elif np.size(peakIdxNew) > 0 and np.size(troughIdx) == 0:
            return xFRF_smooth[peakIdxNew], None
        elif np.size(peakIdxNew) == 0 and np.size(troughIdx) > 0:
            return None, xFRF_smooth[troughIdx]
        else:
            return None, None
    else:
        return None, None