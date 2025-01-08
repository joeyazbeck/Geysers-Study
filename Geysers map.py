import cartopy.crs as ccrs
import cartopy.geodesic as cgeo
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import geopandas as gpd
from datetime import datetime
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import requests


def _axes_to_lonlat(ax, coords):
    """(lon, lat) from axes coordinates."""
    display = ax.transAxes.transform(coords)
    data = ax.transData.inverted().transform(display)
    lonlat = ccrs.PlateCarree().transform_point(*data, ax.projection)

    return lonlat


def _upper_bound(start, direction, distance, dist_func):
    """A point farther than distance from start, in the given direction.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        direction  Nonzero (2, 1)-shaped array, a direction vector.
        distance:  Positive distance to go past.
        dist_func: A two-argument function which returns distance.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    if distance <= 0:
        raise ValueError(f"Minimum distance is not positive: {distance}")

    if np.linalg.norm(direction) == 0:
        raise ValueError("Direction vector must not be zero.")

    # Exponential search until the distance between start and end is
    # greater than the given limit.
    length = 0.1
    end = start + length * direction

    while dist_func(start, end) < distance:
        length *= 2
        end = start + length * direction

    return end


def _distance_along_line(start, end, distance, dist_func, tol):
    """Point at a distance from start on the segment  from start to end.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        end:       Outer bound on point's location.
        distance:  Positive distance to travel.
        dist_func: Two-argument function which returns distance.
        tol:       Relative error in distance to allow.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    initial_distance = dist_func(start, end)
    if initial_distance < distance:
        raise ValueError(f"End is closer to start ({initial_distance}) than "
                         f"given distance ({distance}).")

    if tol <= 0:
        raise ValueError(f"Tolerance is not positive: {tol}")

    # Binary search for a point at the given distance.
    left = start
    right = end

    while not np.isclose(dist_func(start, right), distance, rtol=tol):
        midpoint = (left + right) / 2

        # If midpoint is too close, search in second half.
        if dist_func(start, midpoint) < distance:
            left = midpoint
        # Otherwise the midpoint is too far, so search in first half.
        else:
            right = midpoint

    return right


def _point_along_line(ax, start, distance, angle=0, tol=0.01):
    """Point at a given distance from start at a given angle.

    Args:
        ax:       CartoPy axes.
        start:    Starting point for the line in axes coordinates.
        distance: Positive physical distance to travel.
        angle:    Anti-clockwise angle for the bar, in radians. Default: 0
        tol:      Relative error in distance to allow. Default: 0.01

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    # Direction vector of the line in axes coordinates.
    direction = np.array([np.cos(angle), np.sin(angle)])

    geodesic = cgeo.Geodesic()

    # Physical distance between points.
    def dist_func(a_axes, b_axes):
        a_phys = _axes_to_lonlat(ax, a_axes)
        b_phys = _axes_to_lonlat(ax, b_axes)

        # Geodesic().inverse returns a NumPy MemoryView like [[distance,
        # start azimuth, end azimuth]].
        return geodesic.inverse(a_phys, b_phys).base[0, 0]

    end = _upper_bound(start, direction, distance, dist_func)

    return _distance_along_line(start, end, distance, dist_func, tol)


def scale_bar(ax, location, length, metres_per_unit=1000, unit_name='km',
              tol=0.01, angle=0, color='black', linewidth=3, text_offset=0.005,
              ha='center', va='bottom', plot_kwargs=None, text_kwargs=None,
              **kwargs):
    """Add a scale bar to CartoPy axes.

    For angles between 0 and 90 the text and line may be plotted at
    slightly different angles for unknown reasons. To work around this,
    override the 'rotation' keyword argument with text_kwargs.

    Args:
        ax:              CartoPy axes.
        location:        Position of left-side of bar in axes coordinates.
        length:          Geodesic length of the scale bar.
        metres_per_unit: Number of metres in the given unit. Default: 1000
        unit_name:       Name of the given unit. Default: 'km'
        tol:             Allowed relative error in length of bar. Default: 0.01
        angle:           Anti-clockwise rotation of the bar.
        color:           Color of the bar and text. Default: 'black'
        linewidth:       Same argument as for plot.
        text_offset:     Perpendicular offset for text in axes coordinates.
                         Default: 0.005
        ha:              Horizontal alignment. Default: 'center'
        va:              Vertical alignment. Default: 'bottom'
        **plot_kwargs:   Keyword arguments for plot, overridden by **kwargs.
        **text_kwargs:   Keyword arguments for text, overridden by **kwargs.
        **kwargs:        Keyword arguments for both plot and text.
    """
    # Setup kwargs, update plot_kwargs and text_kwargs.
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}

    plot_kwargs = {'linewidth': linewidth, 'color': color, **plot_kwargs,
                   **kwargs}
    text_kwargs = {'ha': ha, 'va': va, 'rotation': angle, 'color': color,
                   **text_kwargs, **kwargs}

    # Convert all units and types.
    location = np.asarray(location)  # For vector addition.
    length_metres = length * metres_per_unit
    angle_rad = angle * np.pi / 180

    # End-point of bar.
    end = _point_along_line(ax, location, length_metres, angle=angle_rad,
                            tol=tol)

    # Coordinates are currently in axes coordinates, so use transAxes to
    # put into data coordinates. *zip(a, b) produces a list of x-coords,
    # then a list of y-coords.
    ax.plot(*zip(location, end), transform=ax.transAxes, **plot_kwargs)

    # Push text away from bar in the perpendicular direction.
    midpoint = (location + end) / 2
    offset = text_offset * np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    text_location = midpoint + offset

    # 'rotation' keyword argument is in text_kwargs.
    ax.text(*text_location, f"{length} {unit_name}", rotation_mode='anchor',
            transform=ax.transAxes, **text_kwargs)


start_time=datetime.now()

fig=plt.figure(figsize=(10,10))
#Stamen does not work because it is bought out by Stadia
#So I looked up an addition of StadiaMapsTiles written by someone on Github
#and I added it to my cartopy\io\img_tiles.py file
tiler=cimgt.StadiaMapsTiles('ed549d6d-431c-42dc-bd90-78bd88957332',style="stamen_terrain_background")
mercator=tiler.crs
ax=plt.axes(projection=mercator)

zoom = 10
ax.add_image(tiler, zoom)

#Area of study coordinates

minlatitude=38.6
maxlatitude=38.9
minlongitude=-123
maxlongitude=-122.6

#calculating bounds
map_size=0.1
left_bound=minlongitude-map_size
right_bound=maxlongitude+map_size
lower_bound=minlatitude-map_size
upper_bound=maxlatitude+map_size

extent=[left_bound,right_bound,lower_bound,upper_bound]
ax.set_extent(extent)



# ax.add_feature(cfeature.BORDERS)
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.LAKES)
# ax.add_feature(cfeature.LAND)
# ax.add_feature(cfeature.OCEAN)
# ax.add_feature(cfeature.RIVERS)
# ax.add_feature(cfeature.STATES)

text_kwargs = dict(size='xx-large')
scale_bar(ax, (0.1, 0.04), 1_0,text_kwargs=text_kwargs)

minlatitude=38.74
maxlatitude=38.86
minlongitude=-122.89
maxlongitude=-122.69

#Area of study
lons=[minlongitude,minlongitude]
lats=[minlatitude,maxlatitude]
ax.plot(lons,lats,color='blue',transform=ccrs.Geodetic())
lons=[minlongitude,maxlongitude]
lats=[maxlatitude,maxlatitude]
ax.plot(lons,lats,color='blue',transform=ccrs.Geodetic())
lons=[maxlongitude,maxlongitude]
lats=[maxlatitude,minlatitude]
ax.plot(lons,lats,color='blue',transform=ccrs.Geodetic())
lons=[maxlongitude,minlongitude]
lats=[minlatitude,minlatitude]
ax.plot(lons,lats,color='blue',transform=ccrs.Geodetic())

minlatitude=38.6
maxlatitude=38.9
minlongitude=-123
maxlongitude=-122.6

#Get earthquake data to plot
url='https://earthquake.usgs.gov/fdsnws/event/1/query?'
params={'format': 'geojson',
        'starttime': '1975-01-01',
        'endtime': '2023-01-01',
        'minlatitude': minlatitude,
        'minlongitude': minlongitude,
        'maxlatitude': maxlatitude,
        'maxlongitude': maxlongitude,
        'minmagnitude': 4}

r=requests.get(url=url,params=params)
earthquakes_data=r.json()

number_of_earthquakes=earthquakes_data['metadata']['count']
magnitude_of_earthquake=earthquakes_data['features'][0]['properties']['mag']
time_of_earthquake=datetime.fromtimestamp(earthquakes_data['features'][8]['properties']['time']/1000)
longitude_of_earthquake=earthquakes_data['features'][0]['geometry']['coordinates'][0]
latitude_of_earthquake=earthquakes_data['features'][0]['geometry']['coordinates'][1]
depth_of_earthquake=earthquakes_data['features'][0]['geometry']['coordinates'][2] #in km

switch1=True
switch2=True
switch3=True
for i in range(number_of_earthquakes):
    longitude_of_earthquake=earthquakes_data['features'][i]['geometry']['coordinates'][0]
    latitude_of_earthquake=earthquakes_data['features'][i]['geometry']['coordinates'][1]
    magnitude_of_earthquake=earthquakes_data['features'][i]['properties']['mag']
    if magnitude_of_earthquake > 5.0:
        if switch1==True:
            ax.plot(longitude_of_earthquake,latitude_of_earthquake,'x',markersize=20,color='black',markeredgecolor='black',label='M=5.01',transform=ccrs.Geodetic())
            switch1=False
        else:
            ax.plot(longitude_of_earthquake,latitude_of_earthquake,'o',markersize=20,color='black',markeredgecolor='black',transform=ccrs.Geodetic())
    elif magnitude_of_earthquake > 4.5:
        if switch2==True and switch1==False:
            ax.plot(longitude_of_earthquake,latitude_of_earthquake,'o',markersize=20,color='none',markeredgecolor='red',label='4.5<M<=5.0',transform=ccrs.Geodetic())
            switch2=False
        else:
            ax.plot(longitude_of_earthquake,latitude_of_earthquake,'o',markersize=20,color='none',markeredgecolor='red',transform=ccrs.Geodetic())
   
    elif magnitude_of_earthquake >= 4.0:
        if switch3==True and switch1==False and switch2==False:
            ax.plot(longitude_of_earthquake,latitude_of_earthquake,'o',markersize=10,color='none',markeredgecolor='green',label='4.0<=M<=4.5',transform=ccrs.Geodetic())
            switch3=False
        else:
            ax.plot(longitude_of_earthquake,latitude_of_earthquake,'o',markersize=10,color='none',markeredgecolor='green',transform=ccrs.Geodetic())
            

ax.plot(-122.7553,38.7749,'*',color='yellow',markersize=25,transform=ccrs.Geodetic())



#North Arrow
x_pos=-122.52
start=38.51
finish=38.54

lons=[x_pos,x_pos]
lats=[start,finish]
ax.plot(lons,lats,color='black',transform=ccrs.Geodetic())

arrow_size=0.007
trianglex=[x_pos,x_pos+arrow_size,x_pos-arrow_size,x_pos]
triangley=[finish,finish-arrow_size,finish-arrow_size,finish]
ax.plot(trianglex,triangley,'black',transform=ccrs.Geodetic())
ax.fill(trianglex,triangley,'black',transform=ccrs.Geodetic())

ax.text(-122.526,38.545,'N',fontsize=15,transform=ccrs.Geodetic())

#Plotting CA faults
input_file_name = "C:\\Users\\user\\Desktop\\Rundle Research\\Research Projects\\The geysers geothermal project (potential paper)\\California_Faults.txt"
input_file  =   open(input_file_name, 'r')

for line in input_file:
    items = line.strip().split()
    number_points = int(len(items)/2)
    
    for i in range(number_points-1):
        x = [float(items[2*i]),float(items[2*i+2])]
        y = [float(items[2*i+1]), float(items[2*i+3])]
        #ax.plot(x,y,'r-', lw=0.55, zorder=2)
        ax.plot(x,y,'-', color='red',linewidth=2, zorder=2,transform=ccrs.Geodetic())
        
input_file.close()

ax.text(-122.69,38.735,'Collayomi Fault Zone',rotation=-35,fontsize=12,transform=ccrs.Geodetic())
ax.text(-122.8,38.59,'Maacama Fault Zone',rotation=-55,fontsize=12,transform=ccrs.Geodetic())
ax.text(-122.99,38.69,'Healdsburg Fault',rotation=-51,fontsize=12,transform=ccrs.Geodetic())
ax.text(-122.965,38.924,'Wight Way Fault',rotation=21,fontsize=12,transform=ccrs.Geodetic())


gl = ax.gridlines(draw_labels=True,
                  linewidth=2, color='grey', alpha=0.2)
gl.top_labels = True
gl.left_labels = True
gl.xlines = True
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 15, 'color': 'black'}
gl.ylabel_style = {'size': 15, 'color': 'black'}

ax.legend(loc='best',ncol=3,edgecolor='black',fontsize=15)

# plt.savefig('Geysers map.pdf')
plt.show()











