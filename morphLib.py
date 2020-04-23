import datetime as DT
import numpy as np
import peakutils,os, affine, pyproj, rasterio,pickle, shutil, subprocess, sys
from matplotlib import pyplot as plt
from scipy import signal
from pathlib import Path


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
    profileTrend = kwargs.get('profileTrend', peakutils.baseline(profile, deg=polyFit))  # find the trend

    ################################################################
    # start working on data
    assert np.size(profile) == np.size(profileTrend) == np.size(xFRF), 'ProfileTrend must be same size as input profile data, and xFRF'
    assert not np.isnan(profileTrend).any(), 'Profile Trend must not have NaNs'
    ########################################################################
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


def readGeotiff(fname, lonLat=False):
    """reads a raster image using rasterio library

    Args:
        fname: geotiff filename
        LatLon(bool): will return latitude and longitude values as well as stateplane

    Returns:
        eastings, northings, Array of values
        if lonLat is true
        eastings, northings, Array of values, lon, lat

    """
    with rasterio.open(fname) as r:
        T0 = r.transform         # upper left pixel corner affine transform
        p1 = pyproj.Proj(r.crs)
        A = r.read()             # pixelvalues

    cols, rows = np.meshgrid(np.arange(A.shape[2]), np.arange(A.shape[1]))

    # Get affine transform for pixel centres
    T1 = T0 * affine.Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T1

    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(rc2en, otypes=[np.float, np.float])(rows, cols)

    if lonLat is True:
        # Project all longitudes, latitudes
        p2 = pyproj.Proj(proj='latlong', datum='WGS84')
        longs, lats = pyproj.transform(p1, p2, eastings, northings)
        return eastings, northings, A, longs, lats

    else:
        return eastings, northings, A

def run1DsandBarin2D(time, xFRF, yFRF, elevation, meanProfile):
    """ run the 1D profile sand bar identifier on a 2D dem (

    Args:
        time: datetime elements
        xFRF: cross-shore values
        yFRF: along-shore values
        elevation: to find profile on: dimensioned [t, y, x]
        meanProfile: mean profile local grid (over long duration) assumed to start at cross-shore origin and be the length
            of the elevation grid

    Returns:
        xPeak - cross-shore peak of sandbar
        xTroughs - cross-shore trough of sandbar
        yLocs - where in alongshore value

    """
    assert np.ndim(elevation) > 1, 'DEM needs to be at least 2 dimensional'
    if np.ndim(elevation) > 2:
        assert np.size(time) == np.shape(elevation)[0], "DEM must be dimensioned [t, y, x]"
        assert len(meanProfile) == np.shape(elevation)[-1], 'mean profile must be same length as x '
    deepWaterCutoff = 600           # meters in local cross-shore coordinate
    minSandBarSeparation = 100      # meters
    #######################################
    print('working on survey {}\n'.format(time.strftime('%Y%m%d')))
    yLocs, xPeak, xTrough = [], [], []
    for pp in range(elevation.shape[0]):
        profileIn = elevation[pp]
        xIn = xFRF
        plotFname = "2DbathyWorkFlow/{}/profile_y{:03d}.png".format(time.strftime('%Y%m%d'), int(yFRF[pp]))

        try:
            peaks, troughs = findSandBarAndTrough1D(xFRF=xIn,profile=profileIn, profileTrend = meanProfile,
                                                minSandBarSeparation=minSandBarSeparation/np.median(np.diff(xIn)),
                                                deepWaterCoordCutoff=deepWaterCutoff, plotFname=plotFname)
        except FileNotFoundError:
            path = Path(os.path.dirname(plotFname))
            path.mkdir(parents=True, exist_ok=True)
            peaks, troughs = findSandBarAndTrough1D(xFRF=xIn,profile=profileIn, profileTrend = meanProfile,
                                                minSandBarSeparation=100/np.median(np.diff(xIn)),
                                                deepWaterCoordCutoff=600, plotFname=plotFname)
        xPeak.append(peaks)
        xTrough.append(troughs)
        yLocs.append(yFRF[pp])

    return xPeak, xTrough, yLocs


def setupGrass(epsg=None, bin='/usr/bin/grass78'):
    """

    Args:
        epsg:
        bin:

    Returns:
        grass script instance (gs) that has information
    """
    # See: https://grasswiki.osgeo.org/wiki/Working_with_GRASS_without_starting_it_explicitly#Python:_GRASS_GIS_7_without_existing_location_using_metadata_only

    # Assume longitude/latitude location in WGS84 datum

    gisbase = str(subprocess.check_output([bin, "--config", "path"]), 'utf-8').strip().split()[-1]
    os.environ['GISBASE'] = gisbase
    sys.path.append(os.path.join(gisbase, "etc", "python"))
    # put temp GRASS location in current working directory
    gisdb = './grassdata'
    location = 'tmp_location'
    mapset   = 'PERMANENT'
    location_path = os.path.join(gisdb, location)
    if os.path.exists(location_path):
        shutil.rmtree(location_path)
    # Create new location from EPSG code, and exit
    if epsg is not None:
        epsg_code = 'EPSG:{}'.format(epsg)  # LL 4326
        startcmd = ("{} -c {} -e {}".format(bin, epsg_code, location_path))
    else:
        startcmd = ("{} -c -e {}".format(bin, location_path))

    print(startcmd)
    p = subprocess.Popen(startcmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        print('ERROR: %s' % err)
        print('ERROR: Cannot generate location (%s)' % startcmd)
        sys.exit(-1)
    else:
        print('Created location %s' % location_path)
    # Now the location with PERMANENT mapset exists.
    import grass.script as gs
    from grass.script import setup as gsetup
    gsetup.init(gisbase, gisdb, location, mapset)

    return gs