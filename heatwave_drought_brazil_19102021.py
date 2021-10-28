from __future__ import division
import glob
import iris
import iris.coord_categorisation as icc
from iris.time import PartialDateTime
import numpy as np
import numpy.ma as ma
from datetime import date, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import iris.plot as iplt
​from iris.experimental.equalise_cubes import equalise_attributes
​
''' Adapted from Eunice's code '''
''' Corr between heat waves, Z500, and drought '''
''' Using ERA5 daily '''
​
​
#''' EVENT DEFINITION
# Brazil October 7 2020
Mon = 10
Lon = [297, 307]
Lat = [-20, -10]
  
# paths
pera5 = "/bp1store/geog-tropical/data/ERA-5/"
​
# Data
# tasmax, this is from 1950
ftasmax = glob.glob(pera5+"day/tasmax/*_*"+str(Mon)+".nc")
# z500, from 1950
fz = glob.glob(pera5+"day/z/z500/*_*"+str(Mon)+".nc")
# precip, from 1950
fps = glob.glob(pera5+"day/total_precipitation/*_*"+str(Mon)+".nc")
# evaporation, from 1950
fep = glob.glob(pera5+"day/evaporation/*_*"+str(Mon)+".nc")


# To use specific dates ​
#    # last week of June and first week of July?
#    twoweek_cons = iris.Constraint(time=lambda cell: PartialDateTime(month=6, day=24) <= cell.point <= PartialDateTime(month=7, day=7))    
    # first week of 1/2/3 months prior
    #prior_cons = iris.Constraint(time=lambda cell: PartialDateTime(day=1) <= cell.point <= PartialDateTime(day=14))
​
# load data
# tasmax
tasmax_cubes = iris.load(ftasmax)
# z500, at 1 deg res
z1_cubes = iris.load(fz)
z_cubes = iris.cube.CubeList()
# some files have other pressure levels
for c6 in z1_cubes:
    if c6.ndim==4:
        c6_new = c6.extract(iris.Constraint(air_pressure=500.))
        c6_new.remove_coord("air_pressure")
        z_cubes.append(c6_new)
    else:
        z_cubes.append(c6)

# precip 
ps_cubes = iris.load(fps)
# evaporation
ep_cubes = iris.load(fep)
​
# concatenate for each year, tasmax, z500, p-e
equalise_attributes(tasmax_cubes)
tasmax_cube = tasmax_cubes.concatenate_cube()
equalise_attributes(z_cubes)
z_cube = z_cubes.concatenate_cube()
equalise_attributes(ps_cubes)
ps_cube = ps_cubes.concatenate_cube()
equalise_attributes(ep_cubes)
ep_cube = ep_cubes.concatenate_cube()


    
# extract area of interest
lat_cons = iris.Constraint(latitude=lambda cell: Lat[0] <= cell <= Lat[1]) 
lon_cons = iris.Constraint(longitude=lambda cell: Lon[0] <= cell <= Lon[1])
# tasmax
tasmax_reg = tasmax_cube.extract(lat_cons & lon_cons)
# area weighted average, taxmax in deg C
tasmax_reg.coord("latitude").guess_bounds()
tasmax_reg.coord("longitude").guess_bounds()
tasmax_wghs = iris.analysis.cartography.area_weights(tasmax_reg) 
tasmax_means = tasmax_reg.collapsed(["latitude", "longitude"], iris.analysis.MEAN, weights=tasmax_wghs)-273.15
# z500
z_reg = z_cube.extract(lat_cons & lon_cons) 
# area-weighted average, z500 in m
z_reg.coord("latitude").guess_bounds()
z_reg.coord("longitude").guess_bounds()
z_wghs_1deg = iris.analysis.cartography.area_weights(z_reg)
z_means = z_reg.collapsed(["latitude", "longitude"], iris.analysis.MEAN, weights=z_wghs_1deg) 
# P-E in m
ps_reg = ps_cube.extract(lat_cons & lon_cons)  
ps_means = ps_reg.collapsed(["latitude", "longitude"], iris.analysis.MEAN, weights=tasmax_wghs) 
ep_reg = ep_cube.extract(lat_cons & lon_cons)  
ep_means = ep_reg.collapsed(["latitude", "longitude"], iris.analysis.MEAN, weights=tasmax_wghs) 
pe_means = ps_means.copy()
pe_means.data = ps_means.data - ep_means.data
    

# plot graph
iris.coord_categorisation.add_year(pe_means, 'time')
pe_plot = pe_means.aggregated_by('year', iris.analysis.MEAN)   # mean of each year
iris.coord_categorisation.add_year(z_means, 'time')
z_plot = z_means.aggregated_by('year', iris.analysis.MEAN)     # mean of each year
iris.coord_categorisation.add_year(tasmax_means, 'time')
tasmax_plot = tasmax_means.aggregated_by('year', iris.analysis.MAX)   # max of each year
   
# scatter plots
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
plt.sca(ax)       # set current axis
cm = plt.cm.get_cmap("Reds", 20)
# scatter plots
sc = plt.scatter(pe_plot.data, z_plot.data, c=tasmax_plot.data, \
                 vmin=20, vmax=40, cmap=cm, marker="o", alpha=0.7)
plt.colorbar(sc, orientation="vertical", label="Max daily max temp. (deg C)")
ax3.set_ylabel("Z500 (m)")
ax3.set_xlabel("P-E (m)")
ax3.set_title("(c)", loc="left")
plt.tight_layout()
plt.ion()
plt.show()
#plt.savefig("z500_pe_brazil.png", format="png", dpi=300)
#plt.close()
