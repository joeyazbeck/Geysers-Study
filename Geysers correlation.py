from datetime import datetime,timedelta
from time import sleep
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import requests
from scipy.stats import pearsonr, spearmanr

start=datetime.now()



# -----------------------------------------------------------------------------

# url='https://earthquake.usgs.gov/fdsnws/event/1/query?'
# params={'format': 'geojson',
#         'starttime': '2018-01-01',
#         'endtime': '2023-01-01',
#         'minlatitude': 38.74,
#         'minlongitude': -122.89,
#         'maxlatitude': 38.86,
#         'maxlongitude': -122.69,
#         'minmagnitude': 4}

# r=requests.get(url=url,params=params)
# earthquakes_data=r.json()
# number_of_earthquakes=earthquakes_data['metadata']['count']
# magnitude_of_earthquake=earthquakes_data['features'][0]['properties']['mag']
# time_of_earthquake=datetime.fromtimestamp(earthquakes_data['features'][1451]['properties']['time']/1000)


# -----------------------------------------------------------------------------

#Get production/injection data downloaded through conservation.ca.gov
production_injection_file_location='C:\\users\\user\\Desktop\\Rundle Research\\Research Projects\\The geysers geothermal project (potential paper)\\Correlation data\\geysers injection and production.csv'
prod_inj_df=pd.read_csv(production_injection_file_location)
#Extract values from start of 1975 to end of 2022 from dataframe
production_data_monthly=[]
injection_data_monthly=[]
for i in range (88,len(prod_inj_df)):
    production_string_to_int=int(prod_inj_df['Unnamed: 3'][i].replace(',',''))
    production_data_monthly.append(production_string_to_int)
    injection_string_to_int=int(prod_inj_df['Unnamed: 4'][i].replace(',',''))
    injection_data_monthly.append(injection_string_to_int)


#Create list of datetimes months (1975->2022) to use as the x axis when plotting
x_axis_datetimes_monthly=[]
for i in range (2022-1975+1):
    for j in range(12):
        x_axis_datetimes_monthly.append(datetime(1975+i,j+1,1,0,0,0))

#Defining Study Area Bounds
#The OG bounds (might need to forget about these soon.)
# minlatitude=38.633
# maxlatitude=38.886
# minlongitude=-122.903
# maxlongitude=-122.543

#The experimental bounds
minlatitude=38.74
maxlatitude=38.86
minlongitude=-122.89
maxlongitude=-122.69

#Counting the monthly earthquakes with magnitude bigger than the minimum magnitude
url='https://earthquake.usgs.gov/fdsnws/event/1/count?'
minimum_magnitude_list=[1.5]
number_of_counts=len(minimum_magnitude_list)
counts_monthly=np.zeros((number_of_counts,len(x_axis_datetimes_monthly)))
#For each month
for i in range(len(x_axis_datetimes_monthly)):
    #Keep track of 'counting' running time in order to see if it exceeds threshold later
    running_time_1=datetime.now()
    #Add an extra month to x_axis_datetimes_monthly so I can count the earthquakes
    #for that last month. I will then pop the last element after loop is done to plot easier.
    if i == len(x_axis_datetimes_monthly)-1:
        x_axis_datetimes_monthly.append(datetime(2023,1,1,0,0,0))
    #'Count' the number of earthquakes in that time interval
    for j in range(number_of_counts):
        starttime=str(datetime.date(x_axis_datetimes_monthly[i]))
        endtime=str(datetime.date(x_axis_datetimes_monthly[i+1]))
        
        params={'format': 'text',
                'starttime': starttime,
                'endtime': endtime,
                'minlatitude': minlatitude,
                'minlongitude': minlongitude,
                'maxlatitude': maxlatitude,
                'maxlongitude': maxlongitude,
                'minmagnitude': minimum_magnitude_list[j],
                'maxmagnitude': 10}
        
        r=requests.get(url=url,params=params)
        number_of_earthquakes=r.json()
        counts_monthly[j][i]=number_of_earthquakes
        
    total_running_time=datetime.now()-start
    #print('Running time so far is: ' + str(total_running_time))
    print('Month number is: ' + str(x_axis_datetimes_monthly[i]))
    
    running_time_2=datetime.now()
    time_interval=running_time_2-running_time_1
    #print('Time interval between the monthly requests is: ' + str(time_interval))
    
    #Let the program stop if the requests are taking too long so the server can 'breathe'
    if time_interval > timedelta(minutes=1):
        sleep_seconds=60
        print('Requests are taking too long (>1 minute)')
        print('Program will sleep for: %d seconds' %(sleep_seconds))
        sleep(sleep_seconds)
        print('Sleeping is over. Now, trying to do requests again.')
        
#Let's remove that last element that we had to add in the for loop
x_axis_datetimes_monthly.pop()



#Let's plot the earthquake counts as well as the prod/inj values as a function of time.
#First start by converting production and injection values to Mega Tons.
production_plotting_monthly=[]
injection_plotting_monthly=[]
for i in range(len(production_data_monthly)):
    production_plotting_monthly.append(production_data_monthly[i]/1000000)
    injection_plotting_monthly.append(injection_data_monthly[i]/1000000)

fig,ax=plt.subplots()
fig.set_size_inches(60,20)
plt.rcParams.update({'font.size': 50})
plot1=ax.plot(x_axis_datetimes_monthly,counts_monthly[0],color='red',label='Earthquakes M>=%.1f' %minimum_magnitude_list[0])
ax2=ax.twinx()
plot2=ax2.plot(x_axis_datetimes_monthly,production_plotting_monthly,color='green',label='Steam Production')
plot3=ax2.plot(x_axis_datetimes_monthly,injection_plotting_monthly,color='blue',label='Water Injection')

all_plots = plot1+plot2+plot3
labels = [p.get_label() for p in all_plots]

#Ensuring the yticks of both y axes line up
n_ticks = 6
y1_min = 0
y1_max = 250
y2_min = 0
y2_max = 10
ax.set_yticks(np.linspace(y1_min, y1_max, n_ticks))
ax2.set_yticks(np.linspace(y2_min, y2_max, n_ticks))
ax.set_ylim(y1_min, y1_max)
ax2.set_ylim(y2_min, y2_max)

ax.grid()
ax.legend(all_plots, labels, loc='upper left', shadow=True)
ax.set_ylabel('Monthly Earthquake Count')
ax2.set_ylabel('Monthly Production/Injection Data (in mega tons)')
#plt.savefig('seismicity_prod_inj_monthly_1.5up.pdf')
plt.show()

#From the previous plot, it looked like there was a stronger correlation between injection
#and seismicity rather than production and seismicity, so let's explore that.

#Let's plot a scatter plot between injection and seismicity (counts) to see the relation.
fig,ax=plt.subplots()
fig.set_size_inches(20,10)
ax.scatter(injection_plotting_monthly,counts_monthly[0],s=200,marker='x',color='red')

n_ticks_y = 8
n_ticks_x = 9
y_min = 0
y_max = 175
x_min = 0
x_max = 8
ax.set_yticks(np.linspace(y_min, y_max, n_ticks_y))
ax.set_xticks(np.linspace(x_min, x_max, n_ticks_x))
ax.set_ylim(y_min, y_max)
ax.set_xlim(x_min, x_max)

#Include linear fit
p=np.polyfit(injection_plotting_monthly,counts_monthly[0],1)
z=np.poly1d(p)
ax.plot(injection_plotting_monthly,z(injection_plotting_monthly),color='black',label='y=%.1f*x+%.1f' %(p[0],p[1]))

ax.set_xlabel('Monthly Injection (in mega tons)')
ax.set_ylabel('Monthly Earthquake Count with M>=%.1f' %minimum_magnitude_list[0])
ax.legend(loc=0)
ax.grid()
#plt.savefig('seismicity_inj_scatter_monthly_1.5up.pdf')
plt.show()

#Let's compute Pearson correlation coefficient which is a measure of how linearly correlated
#two variables are. I don't expect a perfectly linear relationship between the 2, but
#a higher Pearson value indicates a relationship nonetheless even if it was non-linear.
pearson_correlation_coefficient_monthly=pearsonr(injection_data_monthly,counts_monthly[0])
print(pearson_correlation_coefficient_monthly[0])
spearman_correlation_coefficient_monthly=spearmanr(injection_data_monthly,counts_monthly[0])
print(spearman_correlation_coefficient_monthly[0])



#------------------------------------------------------------------------------

print('Taking a lil break between monthly and annual. Phew!')
sleep(60)
print('Starting up again.')

#------------------------------------------------------------------------------




#------------------------------------------------------------------------------
#Annual analysis section

x_axis_datetimes_annual=[]
for i in range (2022-1975+1):
    x_axis_datetimes_annual.append(datetime(1975+i,1,1,0,0,0))


#Adding up the monthly prod/inj values to produce yearly values
production_data_annual=[]
injection_data_annual=[]
i=0
while i < len(production_data_monthly):
    S1=0
    S2=0
    for j in range(12):
        S1=S1+production_data_monthly[i+j]
        S2=S2+injection_data_monthly[i+j]
    
    production_data_annual.append(S1)
    injection_data_annual.append(S2)
    i=i+12

#Counting the yearly earthquakes with magnitude bigger than the minimum magnitude
url='https://earthquake.usgs.gov/fdsnws/event/1/count?'
minimum_magnitude_list=[1.5]
number_of_counts=len(minimum_magnitude_list)
counts_annual=np.zeros((number_of_counts,len(x_axis_datetimes_annual)))
#For each year
for i in range(len(x_axis_datetimes_annual)):
    #'Count' the number of earthquakes for each minimum magnitude
    running_time_1=datetime.now()
    if i == len(x_axis_datetimes_annual)-1:
        x_axis_datetimes_annual.append(datetime(2023,1,1,0,0,0))
    for j in range(number_of_counts):
        starttime=str(datetime.date(x_axis_datetimes_annual[i]))
        endtime=str(datetime.date(x_axis_datetimes_annual[i+1]))
        
        params={'format': 'text',
                'starttime': starttime,
                'endtime': endtime,
                'minlatitude': minlatitude,
                'minlongitude': minlongitude,
                'maxlatitude': maxlatitude,
                'maxlongitude': maxlongitude,
                'minmagnitude': minimum_magnitude_list[j],
                'maxmagnitude': 10}

        r=requests.get(url=url,params=params)
        number_of_earthquakes=r.json()
        counts_annual[j][i]=number_of_earthquakes
        
    
    total_running_time=datetime.now()-start
    #print('Running time so far is: ' + str(total_running_time))
    print('Year number is: ' + str(x_axis_datetimes_annual[i]))
    
    running_time_2=datetime.now()
    time_interval=running_time_2-running_time_1
    #print('Time interval between the monthly requests is: ' + str(time_interval))
    
    #Let the program stop if the requests are taking too long so the server can 'breathe'
    if time_interval > timedelta(minutes=1):
        sleep_seconds=60
        print('Requests are taking too long (>1 minute)')
        print('Program will sleep for: %d seconds' %(sleep_seconds))
        sleep(sleep_seconds)
        print('Sleeping is over. Now, trying to do requests again.')
        
#Let's remove that last element that we had to add in the for loop
x_axis_datetimes_annual.pop()


#Let's plot the earthquake counts as well as the prod/inj values as a function of time.
production_plotting_annual=[]
injection_plotting_annual=[]
for i in range(len(production_data_annual)):
    production_plotting_annual.append(production_data_annual[i]/1000000)
    injection_plotting_annual.append(injection_data_annual[i]/1000000)

fig,ax=plt.subplots()
fig.set_size_inches(60,20)
plt.rcParams.update({'font.size': 50})
plot1=ax.plot(x_axis_datetimes_annual,counts_annual[0],color='red',label='Earthquakes M>=%.1f' %minimum_magnitude_list[0])
ax2=ax.twinx()
plot2=ax2.plot(x_axis_datetimes_annual,production_plotting_annual,color='green',label='Steam Production')
plot3=ax2.plot(x_axis_datetimes_annual,injection_plotting_annual,color='blue',label='Water Injection')

#Making labels
all_plots = plot1+plot2+plot3
labels = [p.get_label() for p in all_plots]

#Ensuring the yticks of both y axes line up
n_ticks = 6
y1_min = 0
y1_max = 1500
y2_min = 0
y2_max = 125
ax.set_yticks(np.linspace(y1_min, y1_max, n_ticks))
ax2.set_yticks(np.linspace(y2_min, y2_max, n_ticks))
ax.set_ylim(y1_min, y1_max)
ax2.set_ylim(y2_min, y2_max)

ax.grid()
ax.legend(all_plots, labels, loc='upper left', shadow=True)
ax.set_ylabel('Annual Earthquake Count')
ax2.set_ylabel('Annual Production/Injection Data (in mega tons)')
#plt.savefig('seismicity_prod_inj_annual_1.5up.pdf')
plt.show()


#Let's plot a scatter between injection and seismicity to show correlation
fig,ax=plt.subplots()
fig.set_size_inches(20,10)
ax.scatter(injection_plotting_annual,counts_annual[0],s=200,marker='x',color='red')

n_ticks_y = 8
n_ticks_x = 7
y_min = 0
y_max = 1400
x_min = 0
x_max = 60
ax.set_yticks(np.linspace(y_min, y_max, n_ticks_y))
ax.set_xticks(np.linspace(x_min, x_max, n_ticks_x))
ax.set_ylim(y_min, y_max)
ax.set_xlim(x_min, x_max)

#Include linear fit
p=np.polyfit(injection_plotting_annual,counts_annual[0],1)
z=np.poly1d(p)
ax.plot(injection_plotting_annual,z(injection_plotting_annual),color='black',label='y=%.1f*x+%.1f' %(p[0],p[1]))

ax.set_xlabel('Annual Injection (in mega tons)')
ax.set_ylabel('Annual Earthquake Count with M>=%.1f' %minimum_magnitude_list[0])
ax.legend(loc=0)
ax.grid()
#plt.savefig('seismicity_inj_scatter_annual_1.5up.pdf')
plt.show()


pearson_correlation_coefficient_annual=pearsonr(injection_data_annual,counts_annual[0])
print(pearson_correlation_coefficient_annual[0])
spearman_correlation_coefficient_annual=spearmanr(injection_data_annual,counts_annual[0])
print(spearman_correlation_coefficient_annual[0])













running_time=datetime.now()-start
print('Running time is: ' + str(running_time))











































