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
''' This script plots the time series of T '''
''' and heat stress metrcis '''
''' in the NW Pacific region '''
''' from ERA5 data '''
''' Eunice Lo '''
''' Created 02/08/2021 '''
''' Updated 04/08/2021, with Daithi's mask and an added distribution panel '''
''' Updated 05/08/2021, with z500 '''
''' Updated 12/08/2021, with v500 '''
​''' Editted by V Thompson, 17/09/2021 '''
''' Just fig 1c - changing region and dates '''
​
​
# Region / Event
#Mon = 10; Lon = [297, 307]; Lat = [-20, -10] # Brazil, October 7 2020
#Mon=6; 
Lon=[237, 241]; Lat=[45, 52] # western North America (WWA), June 2021
  
# paths
pera5 = "/bp1store/geog-tropical/data/ERA-5/"
​
# Load Data, single cube for each variable
# tasmax, from 1950
ftasmax = glob.glob(pera5+"day/tasmax/*_*[45678].nc") #456 means Apr May Jun
tasmax_cubes = iris.load(ftasmax)
equalise_attributes(tasmax_cubes)
tasmax = tasmax_cubes.concatenate_cube()
# z500, from 1950
fz = glob.glob(pera5+"day/z/z500/*_*[45678].nc")
z1_cubes = iris.load(fz)
z500_cubes = iris.cube.CubeList()
# some files have other pressure levels
for c6 in z1_cubes:
    if c6.ndim==4:
        c6_new = c6.extract(iris.Constraint(air_pressure=500.))
        c6_new.remove_coord("air_pressure")
        z500_cubes.append(c6_new)
    else:
        z500_cubes.append(c6)

equalise_attributes(z500_cubes)
z500 = z500_cubes.concatenate_cube()
# precip, from 1950
fps = glob.glob(pera5+"day/total_precipitation/*_*[45678].nc")
ps1_cubes = iris.load(fps)
ps_cubes = iris.cube.CubeList()
for each in ps1_cubes:
    if each.ndim==4:
        each_new = each[:,0,:,:]
        each_new.remove_coord("expver")
        ps_cubes.append(each_new)
    else:
        ps_cubes.append(each)
        

equalise_attributes(ps_cubes)
ps = ps_cubes.concatenate_cube()
# evaporation, from 1950
fep = glob.glob(pera5+"day/evaporation/*_*[45678].nc")
ep1_cubes = iris.load(fep)
ep_cubes = iris.cube.CubeList()
for each in ep1_cubes:
    if each.ndim==4:
        each_new = each[:,0,:,:]
        each_new.remove_coord("expver")
        ep_cubes.append(each_new)
    else:
        ep_cubes.append(each)
     
   
equalise_attributes(ep_cubes)
ep = ep_cubes.concatenate_cube()
​

# extract area of interest
lat_cons = iris.Constraint(latitude=lambda cell: Lat[0] <= cell <= Lat[1]) 
lon_cons = iris.Constraint(longitude=lambda cell: Lon[0] <= cell <= Lon[1])
tasmax_reg = tasmax.extract(lat_cons & lon_cons).collapsed(["latitude", "longitude"], iris.analysis.MEAN)-273.15
z500_reg = z500.extract(lat_cons & lon_cons).collapsed(["latitude", "longitude"], iris.analysis.MEAN)
ps_reg = ps.extract(lat_cons & lon_cons).collapsed(["latitude", "longitude"], iris.analysis.MEAN)
ep_reg = ep.extract(lat_cons & lon_cons).collapsed(["latitude", "longitude"], iris.analysis.MEAN)
#dry_reg = ps_reg.copy()
#dry_reg.data = ps_reg.data - ep_reg.data

# add coords
iris.coord_categorisation.add_year(tasmax_reg, 'time')
iris.coord_categorisation.add_day_of_year(tasmax_reg, 'time')
iris.coord_categorisation.add_year(z500_reg, 'time')
iris.coord_categorisation.add_day_of_year(z500_reg, 'time')
iris.coord_categorisation.add_year(ps_reg, 'time')
iris.coord_categorisation.add_day_of_year(ps_reg, 'time')
iris.coord_categorisation.add_year(ep_reg, 'time')
iris.coord_categorisation.add_day_of_year(ep_reg, 'time')





### Plotting hottest day, with z500 of that day, and preceeding drought state
### 
# Identify max T day of each year   
yr = np.arange(1950, 2022)
z_max = tasmax_reg.aggregated_by('year', iris.analysis.MAX)
z_day = []
for each in yr:
    z_yr = tasmax_reg.extract(iris.Constraint(year=each))
    z_day.append(z_yr.coord('day_of_year').points[np.where(z_yr.data == z_max.extract(iris.Constraint(year=each)).data)])

# Identify values 
Z_data = [] # Z of day
Z2_data = []
T_data = [] # T of day
P1_data = [] # P prev 2 weeks
P2_data = [] # E prev 2 weeks
E1_data = [] # P prev 4 weeks
E2_data = [] # E prev 4 weeks
for n, each in enumerate(yr):
    print(n, each)
    yr_cons = iris.Constraint(year=each)
    day_cons = iris.Constraint(day_of_year=z_day[n]) # day of max
    prev2week_cons = iris.Constraint(day_of_year=lambda cell: z_day[n][0]-7 <= cell <= z_day[n][0])
    Z_data.append(z500_reg.extract(yr_cons & day_cons).data)
    Z2_data.append(z500_reg.extract(yr_cons & prev2week_cons).data)
    T_data.append(tasmax_reg.extract(yr_cons & day_cons).data)
    P1_data.append(np.mean(ps_reg.extract(yr_cons & prev2week_cons).data))
    E1_data.append(np.mean(ep_reg.extract(yr_cons & prev2week_cons).data))

zip_object = zip(P1_data, E1_data)
D1_data = []
for l1, l2 in zip_object:
    D1_data.append(l1-l2)

# scatter plots
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
plt.sca(ax)       # set current axis
#cm = plt.cm.get_cmap("Reds", 20)
# scatter plots
sc = plt.scatter(D1_data, T_data, marker="+", alpha=0.7)
#plt.colorbar(sc, orientation='vertical', label='TASMAX on Z500 max day')
ax.set_ylabel("T (day)")
ax.set_xlabel("P-E (m) (previous 1 week)")
plt.xlim([0.0015, 0.0065])
plt.tight_layout()
plt.ion()
plt.show()

# scatter plots
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
plt.sca(ax)       # set current axis
cm = plt.cm.get_cmap("Reds", 20)
# scatter plots
sc = plt.scatter(D1_data, Z_data, c=T_data, vmin=25, vmax=40, cmap=cm,  marker="o", alpha=0.7)
plt.colorbar(sc, orientation='vertical', label='TASMAX')
ax.set_ylabel("Z500 (m) (prev week)")
ax.set_xlabel("P-E (m) (prev week)")
plt.xlim([0.0015, 0.0065])
plt.tight_layout()
plt.ion()
plt.show()















