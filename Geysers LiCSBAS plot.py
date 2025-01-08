import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tifffile
from osgeo import gdal
import pyproj

#USE DPI COMMAND IN PLT.FIGURE TO MAKE FIGURES MORE DETAILED RATHER THAN
#MAKING THE WHOLE FIGURE BIGGER.
vel_filt_mskd=tifffile.imread('C:\\Users\\user\\Desktop\\Rundle Research\\Research Projects\\The geysers geothermal project (potential paper)\\LiCSBAS results\\vel.geo.tif')

#I got the coordinates for the 'extent' by running licsbas and double clicking the
#top left corner and bottom right corner and looking at the coordinates in the time
#series window
#I also took the vmin and vmax values from the png file's colorbar.
fig=plt.figure(figsize=(17,17),dpi=200)
matplotlib.rcParams.update({'font.size': 23})
plt.imshow(vel_filt_mskd,cmap='jet',vmin=-27,vmax=49,extent=[-124.74483,-120.85283,36.76939,41.33539])
im_ratio = vel_filt_mskd.shape[0]/vel_filt_mskd.shape[1]
cbar=plt.colorbar(fraction=0.045*im_ratio)

plt.plot(-122.7553,38.7749,'*',color='yellow',markersize=25)

#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
plt.grid(linewidth=0.25,color='gray')
plt.savefig('LiCSBAS Geysers velocity map.pdf')
plt.show()