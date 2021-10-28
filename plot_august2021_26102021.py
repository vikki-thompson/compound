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

​    
# paths
pera5 = "/bp1store/geog-tropical/data/ERA-5/"

# 1979-present July and August data
# tasmax, this is from 1950!
fjun = glob.glob(pera5+"day/tasmax/*_*07.nc")
fjul = glob.glob(pera5+"day/tasmax/*_*08.nc")
## v500, june july 2021
#fj21_v = "/home/yl17544/ERA5_extras/ERA5_v500_hour_20210607.nc"
# z500, from 1950
fjun_z = glob.glob(pera5+"day/z/z500/*_*07.nc")
fjul_z = glob.glob(pera5+"day/z/z500/*_*08.nc")
# precip, from 1950
fps_1 = glob.glob(pera5+"day/total_precipitation/*_*07.nc")
fps_2 = glob.glob(pera5+"day/total_precipitation/*_*08.nc")
# evaporation, from 1950
fep_1 = glob.glob(pera5+"day/evaporation/*_*07.nc")
fep_2 = glob.glob(pera5+"day/evaporation/*_*08.nc")
​
# first two weeks of August
twoweek_cons = iris.Constraint(time=lambda cell: PartialDateTime(month=8, day=1) <= cell.point <= PartialDateTime(month=8, day=12))    
​
    # first week of 1/2/3 months prior
    #prior_cons = iris.Constraint(time=lambda cell: PartialDateTime(day=1) <= cell.point <= PartialDateTime(day=14))
​
# load data
# tasmax
#pnw_jun_cubes = iris.load(fjun, twoweek_cons)
pnw_jul_cubes = iris.load(fjul, twoweek_cons)
## v500
#pnw_junjul_cube_v = iris.load_cube(fj21_v, twoweek_cons)
## turn into daily mean
#icc.add_day_of_year(pnw_junjul_cube_v, "time")
#pnw_daily_cube_v = pnw_junjul_cube_v.aggregated_by("day_of_year", iris.analysis.MEAN)
# z500, at 1 deg res
#pnw_jun_cubes_z1 = iris.load(fjun_z, twoweek_cons)
pnw_jul_cubes_z1 = iris.load(fjul_z, twoweek_cons)
#pnw_jun_cubes_z = iris.cube.CubeList()
pnw_jul_cubes_z = iris.cube.CubeList()
# some files have other pressure levels
for c7 in pnw_jul_cubes_z1:
    if c7.ndim==4:
        c7_new = c7.extract(iris.Constraint(air_pressure=500.))
        c7_new.remove_coord("air_pressure")
        pnw_jul_cubes_z.append(c7_new)
    else:
        pnw_jul_cubes_z.append(c7)
# precip 
pnw_ps_2 = iris.load(fps_2, twoweek_cons)
# evaporation
pnw_ep_2 = iris.load(fep_2, twoweek_cons)
​
# concatenate for each year, tasmax, z500, p-e
pnw_junjul_cubes = iris.cube.CubeList()
pnw_junjul_cubes_z = iris.cube.CubeList()
pnw_cubes_pme = iris.cube.CubeList()
# tasmax is from 1950
for n in range(len(pnw_jun_cubes)):
    cubelist = iris.cube.CubeList([pnw_jul_cubes[n]])
    equalise_attributes(cubelist)
    cube = cubelist.concatenate_cube()
    pnw_junjul_cubes.append(cube)
    
# other variables from 1950
for n in range(len(pnw_jun_cubes_z)):
    # z500
    cubelist_z = iris.cube.CubeList([pnw_jul_cubes_z[n]])
    equalise_attributes(cubelist_z)
    cube_z = cubelist_z.concatenate_cube()
    # turn into m
    cube_z.data //= 9.80665
    cube_z.units = "m"
    pnw_junjul_cubes_z.append(cube_z) 

for n in range(len(pnw_ps_2)):    
    # precip
    if pnw_ps_2[n].ndim==4:
        new_1 = pnw_ps_2[n][:,0,:,:].copy()
        new_1.remove_coord("expver")
        cubelist_p = iris.cube.CubeList([new_1])
    else:
        cubelist_p = iris.cube.CubeList([pnw_ps_2[n]])
    equalise_attributes(cubelist_p)
    cube_p = cubelist_p.concatenate_cube() 
    # evaporation
    if pnw_ep_2[n].ndim==4:
        new_e = pnw_ep_1[n][:,0,:,:].copy()
        new_e.remove_coord("expver")
        cubelist_e = iris.cube.CubeList([new_e])
    else:
        cubelist_e = iris.cube.CubeList([pnw_ep_2[n]])
    equalise_attributes(cubelist_e)
    cube_e = cubelist_e.concatenate_cube()
    cube_e.units = "m"
    new_cube = cube_p.copy()
    new_cube.data = cube_p.data - cube_e.data
    pnw_cubes_pme.append(new_cube)
​
# area of interest
## WWA box
#lat_cons = iris.Constraint(latitude=lambda cell: 45 <= cell <= 52) 
#lon_cons = iris.Constraint(longitude=lambda cell: 237 <= cell <= 241)
# Italy-Greece box
lat_cons = iris.Constraint(latitude=lambda cell: 36 <= cell <= 42) 
lon_cons = iris.Constraint(longitude=lambda cell: 12 <= cell <= 18)
    

# Northern Hemisphere box
pna_lat_cons = iris.Constraint(latitude=lambda cell: 0 <= cell <= 90) 
pna_lon_cons = iris.Constraint(longitude=lambda cell: 0 <= cell <= 360)
​
# extract area of interest
# tasmax
pnw_junjul_reg = pnw_junjul_cubes.extract(lat_cons & lon_cons)
      
# area weighted average, taxmax in deg C
pnw_junjul_reg[0].coord("latitude").guess_bounds()
pnw_junjul_reg[0].coord("longitude").guess_bounds()
pnw_wghs = iris.analysis.cartography.area_weights(pnw_junjul_reg[0]) 
pnw_junjul_means = [c.collapsed(["latitude", "longitude"], iris.analysis.MEAN, weights=pnw_wghs)-273.15 for c in pnw_junjul_reg]
     
# v winds 500 hPa in the Northern Hemisphere, in m/s
# 11th August
#date_cons = iris.Constraint(time=lambda cell: cell.point == PartialDateTime(month=8, day=11)) 
    #pnw_plot_cube_v = pnw_daily_cube_v.extract(date_cons & pna_lat_cons & pna_lon_cons)
    
# two-week average
#pnw_nh_cube_v = pnw_daily_cube_v.extract(pna_lat_cons & pna_lon_cons)
#pnw_plot_cube_v = pnw_nh_cube_v.collapsed("time", iris.analysis.MEAN)
​
    # z500 in WWA box
pnw_z_reg = pnw_junjul_cubes_z.extract(lat_cons & lon_cons) 
​
  
    
# area-weighted average, z500 in m
pnw_z_reg[0].coord("latitude").guess_bounds()
pnw_z_reg[0].coord("longitude").guess_bounds()
pnw_wghs_1deg = iris.analysis.cartography.area_weights(pnw_z_reg[0])
pnw_z_means = [mzc.collapsed(["latitude", "longitude"], iris.analysis.MEAN, weights=pnw_wghs_1deg) for mzc in pnw_z_reg] 
​
# P-E in WWA box, in m
pnw_pme_reg = pnw_cubes_pme.extract(lat_cons & lon_cons)    
# area-weighted average, P-E in m
pnw_pme_means = [mpec.collapsed(["latitude", "longitude"], iris.analysis.MEAN, weights=pnw_wghs) for mpec in pnw_pme_reg] 
    
    # plot graph
fig, axs = plt.subplots(nrows=1, ncols=4, gridspec_kw={'width_ratios': [4, 1, 0.5, 5]}, figsize=(5,6))
# time series plot, tasmax
ax1 = axs[0,0]
sdate = date(2021,8,1)
edate = date(2021,8,13)
x = [sdate+timedelta(days=x) for x in range((edate-sdate).days)]
    
# loop years
for y in range(len(pnw_junjul_means)-2):
    ax1.plot(x, pnw_junjul_means[y].data, color="black", alpha=0.6)

ax1.plot(x, pnw_junjul_means[-2].data, color="black", alpha=0.6, label="1950-2020")    # for the sake of single labeling
ax1.plot(x, pnw_junjul_means[-1].data, color="red", linewidth=3, label="2021")
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
# rotated distribution, tasmax
ax2 = axs[0,1]
ax2.sharey=ax1
# convert into arrays
pre2021_data = np.zeros((len(pnw_junjul_means[:-1]), 14))
for yc in range(len(pnw_junjul_means[:-1])):
    pre2021_data[yc, :] = pnw_junjul_means[yc].data
pre2021_data = pre2021_data.flatten()   
yr2021_data = pnw_junjul_means[-1].data
​
# plot dist
ax2.hist(pre2021_data, color="black", alpha=0.6, histtype='bar', ec='black', orientation="horizontal")
ax2.axhline(np.max(yr2021_data), color="red", linewidth=3)  
ax2.set_yticklabels([]) 
ax2.tick_params(axis='y', which='major', pad=-20)
ax2.set_xlabel("Frequency") 
ax2.set_title("(b)", loc="left")
​
# share y axis
ax1.get_shared_y_axes().join(ax1, ax2)
ax2.autoscale()
​
# invisible ax to add space 
axv = axs[0,2]
axv.set_visible(False)
    
# drought plot    
ax3 = axs[0,3]
plt.sca(ax3)       # set current axis
cm = plt.cm.get_cmap("Reds", 20)

# z500 vs p-e, two-week mean in each year
plot_pme = []
plot_z = []
for cpme, cz in zip(pnw_pme_means, pnw_z_means):
    twoweek_mean_pme = cpme.collapsed("time", iris.analysis.MEAN)
    twoweek_mean_z = cz.collapsed("time", iris.analysis.MEAN)
    plot_pme.append(twoweek_mean_pme.data)
    plot_z.append(twoweek_mean_z.data)
​
# max tasmax each two-week period of each year, 1950-2021
plot_tmax = np.array([ct.collapsed("time", iris.analysis.MAX).data for ct in pnw_junjul_means])
​
# scatter plots
sc = plt.scatter(plot_pme[:71], plot_z, c=plot_tmax[:71], \
                 vmin=20, vmax=40, cmap=cm, marker="o", alpha=0.7)
plt.colorbar(sc, orientation="vertical", label="Max daily max temp. (deg C)")
ax3.set_ylabel("Z500 (m)")
ax3.set_xlabel("P-E (m)")
ax3.set_title("(c)", loc="left")
ax3.set_xlim([0.0015, 0.006])

