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
''' Assessing top 10% of E-P and T events '''
''' 26th Oct 2021 '''
''' Vikki Thompson '''
​
# Region / Event
#Lon=[237, 241]; Lat=[45, 52]; Mon=6 # western North America (WWA), June 2021
Lon=[12, 18]; Lat=[36, 42]; Mon=8 # Italy/Greece (Aug 11 2021)
extreme_year=2021
#Lon=[297, 307]; Lat=[-20, -10]; Mon=10 # SE Brazil (7th October 2020)  
#extreme_year=2020

# paths
pera5 = "/bp1store/geog-tropical/data/ERA-5/"
​
# Load Data, single cube for each variable
# tasmax, from 1950

ftasmax = glob.glob(pera5+"day/tasmax/*_*"+str(Mon)+".nc") 
tasmax_cubes = iris.load(ftasmax)
equalise_attributes(tasmax_cubes)
tasmax = tasmax_cubes.concatenate_cube()

# z500, from 1950
fz = glob.glob(pera5+"day/z/z500/*_*"+str(Mon)+".nc")
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
fps = glob.glob(pera5+"day/total_precipitation/*_*"+str(Mon)+".nc")
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
fep = glob.glob(pera5+"day/evaporation/*_*"+str(Mon)+".nc")
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
# drought index
dry = ps.copy()
dry.data = ps.data - ep.data

# extract area of interest
lat_cons = iris.Constraint(latitude=lambda cell: Lat[0] <= cell <= Lat[1]) 
lon_cons = iris.Constraint(longitude=lambda cell: Lon[0] <= cell <= Lon[1])
tasmax_reg = tasmax.extract(lat_cons & lon_cons).collapsed(["latitude", "longitude"], iris.analysis.MEAN)-273.15
z500_reg = z500.extract(lat_cons & lon_cons).collapsed(["latitude", "longitude"], iris.analysis.MEAN)
dry_reg = dry.extract(lat_cons & lon_cons).collapsed(["latitude", "longitude"], iris.analysis.MEAN)

# add coords
iris.coord_categorisation.add_year(tasmax_reg, 'time')
iris.coord_categorisation.add_day_of_year(tasmax_reg, 'time')
iris.coord_categorisation.add_year(z500_reg, 'time')
iris.coord_categorisation.add_day_of_year(z500_reg, 'time')
iris.coord_categorisation.add_year(dry_reg, 'time')
iris.coord_categorisation.add_day_of_year(dry_reg, 'time')




### PLOT
'''
fig, axs = plt.subplots(nrows=1, ncols=1, gridspec_kw={'width_ratios': [4, 1, 0.5, 5]}, figsize=(5,6))
# time series plot, tasmax
ax1 = axs[0]
sdate = date(2021,Mon,1)
edate = date(2021,Mon,31)
x = [sdate+timedelta(days=x) for x in range((edate-sdate).days)]

# loop years
for y in np.arange(1950, 2021):
    t_data = tasmax_reg.extract(iris.Constraint(year=y)).data[:30]
    ax1.plot(x, t_data, color="black", alpha=0.6)
#ax1.plot(x, pnw_junjul_means[-2].data, color="black", alpha=0.6, label="1950-2020")    # for the sake of single labeling
ax1.plot(x, tasmax_reg.extract(iris.Constraint(year=extreme_year)).data[:30], color="red", linewidth=3, label=str(extreme_year))
​
# y-axis
ax1.set_ylabel("Daily max temp. (deg C)")
# x-axis
fmt_week = mdates.DayLocator(interval=5)
fmt_day = mdates.DayLocator()
ax1.xaxis.set_major_locator(fmt_week)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax1.xaxis.set_minor_locator(fmt_day)
ax1.set_xlabel("Date")
#fig.autofmt_xdate()
ax1.legend(loc=0, fontsize=8) 
ax1.set_title("(a)", loc="left")
'''

# drought plot, daily
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,4)) 
ax1=axs[0]
plt.sca(ax1)       # set current axis
cm = plt.cm.get_cmap("Reds", 20)
T = tasmax_reg.data[:2201]
Z = z500_reg.data[:2201]
D = dry_reg.data[:2201]
sc = plt.scatter(D, T, c=Z, vmin=53500, vmax=59000, cmap=cm, marker='o', alpha=0.7)
ax1.set_ylabel('T')
ax1.set_xlabel('D')

ax2=axs[1]
plt.sca(ax2)       # set current axis
cm = plt.cm.get_cmap("Reds", 20)
sc = plt.scatter(D, Z, c=T, vmin=10, vmax=45, cmap=cm, marker='o', alpha=0.7)
ax2.set_ylabel('Z')
ax2.set_xlabel('D')

ax3=axs[2]
plt.sca(ax3)       # set current axis
cm = plt.cm.get_cmap("Reds", 20)
sc = plt.scatter(T, Z, c=D, vmin=0, vmax=0.03, cmap=cm, marker='o', alpha=0.7)
ax3.set_ylabel('Z')
ax3.set_xlabel('T')




# drought plot, monthly
T = tasmax_reg.aggregated_by('year', iris.analysis.MAX).data[:71]
Z = z500_reg.aggregated_by('year', iris.analysis.MAX).data[:71]
D = dry_reg.aggregated_by('year', iris.analysis.MEAN).data[:71]

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,4)) 
ax1=axs[0]
plt.sca(ax1)       # set current axis
cm = plt.cm.get_cmap("Reds", 20)
sc = plt.scatter(D, T, c=Z, vmin=57000, vmax=59000, cmap=cm, marker='o', alpha=0.7)
ax1.set_xlim([0, 0.002])
ax1.set_ylabel('T')
ax1.set_xlabel('D')

ax2=axs[1]
plt.sca(ax2)       # set current axis
cm = plt.cm.get_cmap("Reds", 20)
sc = plt.scatter(D, Z, c=T, vmin=25, vmax=32, cmap=cm, marker='o', alpha=0.7)
ax2.set_ylabel('Z')
ax2.set_xlabel('D')
ax2.set_xlim([0, 0.002])
ax2.set_title('North America, June')

ax3=axs[2]
plt.sca(ax3)       # set current axis
cm = plt.cm.get_cmap("Reds", 20)
sc = plt.scatter(T, Z, c=D, vmin=0, vmax=0.002, cmap=cm, marker='o', alpha=0.7)
ax3.set_ylabel('Z')
ax3.set_xlabel('T')



















# Identify max temp day of each year   
yr = np.arange(1950, 2021)
t_max = tasmax_reg.aggregated_by('year', iris.analysis.MAX)
t_day = []
for each in yr:
    t_yr = tasmax_reg.extract(iris.Constraint(year=each))
    t_day.append(t_yr.coord('day_of_year').points[np.where(t_yr.data == t_max.extract(iris.Constraint(year=each)).data)])

# Identify values for correlations
# Hottest day for T, 2 weeks run up for D
T_data = []
D_data = []
for n, each in enumerate(yr):
    print(n, each)
    yr_cons = iris.Constraint(year=each)
    day_consT = iris.Constraint(day_of_year=t_day[n]) # day of max
    day_consD = iris.Constraint(day_of_year=lambda cell: t_day[n][0]-14 <= cell <= t_day[n][0])
    T_data.append(tasmax_reg.extract(yr_cons & day_consT).data)
    D_data.append(np.mean(dry_reg.extract(yr_cons & day_consD).data))

# scatter plots
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
plt.sca(ax)       # set current axis
# scatter plots
sc = plt.scatter(D_data, T_data, marker="+")
ax.set_ylabel("TASMAX")
ax.set_xlabel("P-E (m)")
plt.tight_layout()
plt.ion()
plt.show()



# Identify values for correlations
# Hottest day for T, weeks -6 to -2 for D
T_data = []
D_data = []
for n, each in enumerate(yr):
    print(n, each)
    yr_cons = iris.Constraint(year=each)
    day_consT = iris.Constraint(day_of_year=t_day[n]) # day of max
    day_consD = iris.Constraint(day_of_year=lambda cell: t_day[n][0]-42 <= cell <= t_day[n][0]-14)
    T_data.append(tasmax_reg.extract(yr_cons & day_consT).data)
    D_data.append(np.mean(dry_reg.extract(yr_cons & day_consD).data))

# scatter plots
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
plt.sca(ax)       # set current axis
# scatter plots
sc = plt.scatter(D_data, T_data, marker="+")
ax.set_ylabel("TASMAX")
ax.set_xlabel("P-E (m)")
plt.tight_layout()
plt.ion()
plt.show()


