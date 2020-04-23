import os, sys, subprocess, shutil, pickle
import numpy as np
from matplotlib import pyplot as plt
import datetime as DT
import morphLib
from matplotlib import patches as mpatches
sys.path.append('/home/spike/repos')
from testbedutils import geoprocess as gp
from getdatatestbed import getDataFRF

#######################################
# Get numpy arrays from geotiffs      #
#######################################
# fname = '/home/spike/repos/sandbarTool/GRASS_geotiffExport_clean/F_19890221.tif'
# easting, northing, elevation = morphLib.readGeotiff(fname)
# time = DT.datetime.strptime(os.path.basename(fname).split('.')[0].split('_')[-1], '%Y%m%d')
# coords = gp.FRFcoord(easting, northing)
# # np.argwhere()

#######################################
# load Grid from netCDF               #
#######################################
xBounds = [0,800]
yBounds = [0, 1000]
gm = getDataFRF.getDataTestBed(DT.datetime(2017,10,3), DT.datetime(2017,10,25))
data = gm.getBathyIntegratedTransect(xbounds=xBounds, ybounds=yBounds)
#######################################
# Launch session and set region from the numpy array extents
#######################################
gs = morphLib.setupGrass()
# Region settings
# gs.run_command('g.region', n=northing.max(), s=northing.min(), e=easting.max(), w=easting.min(),  # for use in projected coords
#                rows=np.shape(elevation)[-2], cols=np.shape(elevation)[-1], flags="p")
gs.run_command('g.region', n=data['yFRF'].max(), s=data['yFRF'].min(), e=data['xFRF'].max(), w=data['yFRF'].min(),
               rows=data['elevation'].shape[-2], cols=data['elevation'].shape[-1], flags="p")
from grass.script import array as garray
# Create new array and assign values from the elev numpy array
elev_arr = garray.array()
elev_arr[:] =  data['elevation'] # elevation.squeeze()
# Save it to a GRASS raster.
elev_arr.write('elev', overwrite=True)
# Now run GRASS modules on that raster
gs.run_command('r.info', map_="elev")

############## 2D version of 1D sandbarTool #####################################
with open('/home/spike/repos/sandbarTool/meanSandbarProfile.pickle', 'rb') as fid:
    out = pickle.load(fid)
    # interpolate on the way out to now
    meanProfile = np.interp(data['xFRF'], out.pop('xOut'), out.pop('meanProfile'))
meanProfile[np.isnan(meanProfile)] = np.nanmax(meanProfile)
xPeak, xTrough, yLocs = morphLib.run1DsandBarin2D(data['time'], data['xFRF'], data['yFRF'], data['elevation'], meanProfile)
#######################################
# Geomorphon
#######################################
# string variable to hold name of raster result
flatVal = 0.25
for flatVal in [0.1,0.15,0.2,0.25,0.5]:
    geomorph_result = 'geomorph_result'
    gs.run_command('r.geomorphon', elevation="elev", forms=geomorph_result, flat=flatVal, overwrite=True)
    gs.run_command('r.category', map=geomorph_result)
    # Convert GRASS raster to numpy array
    geomorph_arr = gs.array.array()
    geomorph_arr.read(geomorph_result)
    geomorph_arr = np.asarray(geomorph_arr, dtype = np.integer)
    gCrestLocs = np.argwhere(geomorph_arr == 3)
    gTroughLocs = np.argwhere(geomorph_arr == 1)
    #########################################
    # Make Plot
    #########################################
    plt.ion()
    plt.figure(figsize=(12,5))
    ax1 = plt.subplot(131)
    ax1.set_title('1D approach')
    mesh = ax1.pcolormesh(data['xFRF'], data['yFRF'], data['elevation'].squeeze())
    plt.colorbar(mesh)
    ax1.contour(data['xFRF'], data['yFRF'], data['elevation'].squeeze(), c='k', levels=[-8,-6,-3,-2,-1])
    for yy, peaks in enumerate(xPeak):
        if peaks is not None:
            ax1.plot(peaks, np.tile(yLocs[yy], len(peaks)), 'bD', ms=2)
    for yy, troughs in enumerate(xTrough):
        if troughs is not None:
            ax1.plot(troughs, np.tile(yLocs[yy], len(troughs)), 'go', ms=1)
    
    ax2 = plt.subplot(132, sharey=ax1, sharex=ax1)
    ax2.set_title('Geomorphons classification')
    mappable = ax2.pcolormesh(data['xFRF'], data['yFRF'], geomorph_arr, vmin=0, vmax=10, cmap='tab10')
    cbar = plt.colorbar(mappable)
    cbar.set_clim([0,10])

    ax3 = plt.subplot(133, sharey=ax1, sharex=ax1)
    ax3.set_title('combined')
    mesh = ax3.pcolormesh(data['xFRF'], data['yFRF'], data['elevation'].squeeze())
    plt.colorbar(mesh)
    ax3.plot(data['xFRF'][gCrestLocs[:,1]], data['yFRF'][gCrestLocs[:,0]],  'k.', label='geomorph_C')
    ax3.plot(data['xFRF'][gTroughLocs[:,1]], data['yFRF'][gTroughLocs[:,0]],  'c.', label='geomorph_T')

    for yy, peaks in enumerate(xPeak):
        if peaks is not None:
            ax3.plot(peaks, np.tile(yLocs[yy], len(peaks)), 'b.', ms=2)
    for yy, troughs in enumerate(xTrough):
        if troughs is not None:
            ax3.plot(troughs, np.tile(yLocs[yy], len(troughs)), 'g.', ms=2)
    ax3.plot(peaks, np.tile(yLocs[yy], len(peaks)), 'b.', ms=2, label='Crest')
    ax3.plot(troughs, np.tile(yLocs[yy], len(troughs)), 'g.', ms=2, label='Trough')
    ax3.legend()
    
    barPatch = mpatches.Patch(color='blue', label='sandbar')
    troughPatch = mpatches.Patch(color='green', label='trough')
    ax1.legend([barPatch, troughPatch], ['SandBar', 'Trough'])
    ax1.set_ylabel('yFRF [m]')
    ax1.set_xlabel('xFRF [m]')
    ax2.set_xlabel('xFRF [m]')
    plt.suptitle('Geomorph Result with flat = {}\n1-flat, 3-ridge, 4-shoulder, 5-spur,\n6-slope, 7-hollow, 8-footslope, 9-valley, 10-depression'.format(flatVal))
    
    plt.tight_layout(rect=[0.01,0.01,.95,.9])
    plt.savefig('GrassTest_{}.png'.format(flatVal))
    plt.close()

