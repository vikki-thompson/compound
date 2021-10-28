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


def load_tasmax(Mon):
    ' Loads cube of daily tasmax 1850-July 2021, for single month '
    ftasmax = glob.glob(pera5+"day/tasmax/*_*"+str(Mon)+".nc") 
    tasmax_cubes = iris.load(ftasmax)
    equalise_attributes(tasmax_cubes)
    return(tasmax_cubes.concatenate_cube())

def load_z500(Mon):
    ' Loads cube of daily tasmax 1850-July 2021, for single month '
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
    return(z500_cubes.concatenate_cube())

def load_dry(Mon):
    ' Loads cube of daily drought (p-e) 1850-July 2021, for single month '
    #precip    
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
    # drought index
    dry = ps.copy()
    dry.data = ps.data - ep.data
    return(dry)

def extract_region(cube, Lat, Lon):
    # extract area of interest
    lat_cons = iris.Constraint(latitude=lambda cell: Lat[0] <= cell <= Lat[1]) 
    lon_cons = iris.Constraint(longitude=lambda cell: Lon[0] <= cell <= Lon[1])
    return(cube.extract(lat_cons & lon_cons).collapsed(["latitude", "longitude"], iris.analysis.MEAN))

def TD_dailydata(Mon, Lat, Lon):
    # for given month and region
    # returns count per year above/below 10%
    tas = load_tasmax(Mon)
    dry = load_dry(Mon)
    tas_reg = extract_region(tas, Lat, Lon)-273.15
    iris.coord_categorisation.add_year(tas_reg, 'time')
    dry_reg = extract_region(dry, Lat, Lon)
    iris.coord_categorisation.add_year(dry_reg, 'time')
    return(tas_reg, dry_reg)

def compound_count(tas, dry, tas_limit, dry_limit):
    # count days per year in upper 10%
    Tts = []
    Dts = []
    TDts = []
    for yr in np.arange(1950, 2021):
        x = tas.extract(iris.Constraint(year=yr)).data
        y = dry.extract(iris.Constraint(year=yr)).data
        T_count=0;D_count=0; TD_count=0
        for T, D in zip(x, y):
            if T > tas_limit:
                T_count+=1
            if D < dry_limit:
                D_count+=1
            if T > tas_limit and D < dry_limit:
                TD_count+=1
        Tts.append(T_count)
        Dts.append(D_count)
        TDts.append(TD_count)
    return(Tts, Dts, TDts)



def T_limit(tas, tas_limit):
    # Extract ts of number of days above limit within month each year
    # tas: 2d cube (e.g. tas_reg)
    # tas_limit: percentage to include (eg 10 = upper 10% of days)
    tas10 = tas.collapsed('time', iris.analysis.PERCENTILE, percent=[100-tas_limit]).data[0]
    Tts = []
    for yr in np.arange(1950, 2021): #change to use cube coord
        x = tas.extract(iris.Constraint(year=yr)).data
        T_count=0;
        for T in x:
            if T > tas10:
                T_count+=1
        Tts.append(T_count)
    return Tts

def D_limit(dry, dry_limit):
    # Extract ts of number of days above limit within month each year
    # tas: 2d cube (e.g. tas_reg)
    # tas_limit: percentage to include (eg 10 = lowerr 10% of days)
    dry10 = dry.collapsed('time', iris.analysis.PERCENTILE, percent=[dry_limit]).data[0]
    Dts = []
    for yr in np.arange(1950, 2021): #change to use cube coord
        x = dry.extract(iris.Constraint(year=yr)).data
        D_count=0;
        for D in x:
            if D > dry10:
                D_count+=1
        Dts.append(D_count)
    return Dts


    dry10 = dry_reg.collapsed('time', iris.analysis.PERCENTILE, percent=[10, 90]).data[0]
    Tts, Dts, TDts = compound_count(tas_reg, dry_reg, tas10, dry10)
    return(Tts, Dts, TDts)






# Region / Event
#Lon=[237, 241]; Lat=[45, 52]; Mon=6 # western North America (WWA), June 2021
#Lon=[236, 240]; Lat=[38, 42] # California North
#Lon=[12, 18]; Lat=[36, 42]; Mon=8 # Italy/Greece (Aug 11 2021)
#Lon=[297, 307]; Lat=[-20, -10]; Mon=10 # SE Brazil (7th October 2020)  
Lon = [351, 359]; Lat = [36, 44]; Mon=7 # Spain

# paths
pera5 = "/bp1store/geog-tropical/data/ERA-5/"
​
# Load Data
tas_reg, dry_reg = TD_dailydata(Mon, Lat, Lon)
Tts = T_limit(tas_reg, 10)
Dts = D_limit(dry_reg, 10)
TDts = 

Tts_jun, Dts_jun, TDts_jun = TD_10per(6, Lat, Lon)
Tts_jul, Dts_jul, TDts_jul = TD_10per(7, Lat, Lon)
#Tts_aug, Dts_aug, TDts_aug = TD_10per(8, Lat, Lon) # doesn't work, not all data avail
Tts_sep, Dts_sep, TDts_sep = TD_10per(9, Lat, Lon)


Tts = []
Dts = []
TDts = []
for x in np.arange(len(Tts_jun)):
    print(x)
    Tts.append(Tts_jun[x]+Tts_jul[x]+Tts_sep[x])
    Dts.append(Dts_jun[x]+Dts_jul[x]+Dts_sep[x])
    TDts.append(TDts_jun[x]+TDts_jul[x]+TDts_sep[x])



# plot timeseries of this
yr = np.arange(1950, 2021)

fig, ax = plt.subplots()
fig, ax = plt.subplots()
index = np.arange(len(yr))
bar_width = 0.35
opacity = 0.8
rects1 = plt.bar(index, Tts, bar_width,
alpha=opacity,
color='b',
label='Tts')
rects2 = plt.bar(index + bar_width, TDts, bar_width,
alpha=opacity,
color='g',
label='TDts')
plt.xlabel('Year')
plt.ylabel('Days above 10%')
#plt.title('Scores by person')
plt.xticks(index[0::10] + bar_width, yr[0::10])
plt.legend()

plt.tight_layout()
plt.ion()
plt.show()



# do for another month, do for multiple months, remove trend (rolling 10% value?)





### PLOT
fig, axs = plt.subplots(nrows=1, ncols=4, gridspec_kw={'width_ratios': [4, 1, 0.5, 5]}, figsize=(5,6))
    
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
#ax1.plot(x, tasmax_reg.extract(iris.Constraint(year=extreme_year)).data[:30], color="red", linewidth=3, label=str(extreme_year))
​# y-axis
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
#ax1.set_title("(a)", loc="left")


##
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5,6))
    
# time series plot, tasmax
ax1 = axs[0]
T = tasmax_reg.data
D = dry_reg.data
ax1.plot(T, np.arange(2232))
ax1.plot(D, np.arange(2232), 'r')

import iris.quickplot as qplt
import iris.plot as iplt
qplt.plot(tasmax_reg)
qplt.plot(dry_reg, 'r')


















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


