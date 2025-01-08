import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import glob
from scipy.interpolate import splrep, BSpline
from sklearn.preprocessing import MinMaxScaler
import pickle
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression

start=datetime.datetime.now()

x_pixels=201 #2056-1856+1
y_pixels=121 #2596-2476+1
sampling_period=6 #days
training_percentage=0.90
look_back=1 #using the last X images


def PreprocessingGeothermalDataset():
    #Function to preprocess (mostly normalize) the injection data (most correlation)
    
    print('Preprocessing Geothermal Dataset ...')
    
    
    #Calculate injection for every single day from 2017/5/1 to 2022/4/10
    
    injection_production_times=[]
    for i in range (2022-2017+1):
        for j in range(12):
            injection_production_times.append(datetime.date(2017+i,j+1,1))
    injection_production_times=injection_production_times[4:]
    injection_production_times=injection_production_times[:-8]
    
    first_day=datetime.date(2017,5,1)
    last_day=datetime.date(2022,4,10)
    
    complete_injection_production_times=[]
    complete_injection_production_times.append(first_day)
    i=0
    add_1_day=datetime.timedelta(days=1)
    while complete_injection_production_times[i] != last_day:
        complete_injection_production_times.append(complete_injection_production_times[i]+add_1_day)
        i=i+1
        
    divisor=[] #basically a list of the amount of days in each month for my time period.
    for i in range(0,len(injection_production_times)-1):
        divisor.append((injection_production_times[i+1]-injection_production_times[i]).days)
        if i == len(injection_production_times) - 2:
            last_date_in_list=injection_production_times[i+1]
            if last_date_in_list.month == 12:
                next_date=datetime.date(last_date_in_list.year+1,1,1)
            else:
                next_date=datetime.date(last_date_in_list.year,last_date_in_list.month+1,1)
            divisor.append((next_date-last_date_in_list).days)
    
    daily_injection=[]
    for i in range(len(injection_data_monthly)):
        daily_value=injection_data_monthly[i]/divisor[i]
        for j in range(divisor[i]):
            daily_injection.append(daily_value)

    daily_production=[]
    for i in range(len(production_data_monthly)):
        daily_value=production_data_monthly[i]/divisor[i]
        for j in range(divisor[i]):
            daily_production.append(daily_value)
        
    #Trim the start so the series starts on 5/18/2017 same as the images
    #Trim the end of daily_injection since the last day is on 4/10 not 4/30
    complete_injection_production_times=complete_injection_production_times[17:]
    daily_injection=daily_injection[17:]
    daily_injection=daily_injection[:-20]
    daily_production=daily_production[17:]
    daily_production=daily_production[:-20]
    
    #Add up the injections between image dates (for look_back=1 take injection day-of)
    #This should have a size of len(complete_times)-1 so 298 because I am grouping
    #consecutive pairs of images. (basically counting the gaps between images)
    added_injection_values=[]
    added_production_values=[]
    if look_back != 1:
        for i in range(0,len(daily_injection)-(sampling_period*(look_back-1))+1,sampling_period):
            S1=0
            S2=0
            for j in range(sampling_period*(look_back-1)):
                S1=S1+daily_injection[i+j]
                S2=S2+daily_production[i+j]
            added_injection_values.append(S1)
            added_production_values.append(S2)
    else: #In case, I test look_back = 1, this just takes the injection value on the day-of
        for i in range(0,len(daily_injection),sampling_period):
            added_injection_values.append(daily_injection[i])
            added_production_values.append(daily_production[j])
    
    train_size=int(len(added_injection_values)*training_percentage)
    training_added_injection_values=added_injection_values[0:train_size]
    testing_added_injection_values=added_injection_values[train_size:len(added_injection_values)]
    training_added_production_values=added_production_values[0:train_size]
    testing_added_production_values=added_production_values[train_size:len(added_production_values)]
    
    
    #Dropping the last value so it matches to TrainX since TrainX doesn't have the last
    #one because there would be no associated trainY (target) for it.
    training_added_injection_values=training_added_injection_values[:-1]
    testing_added_injection_values=testing_added_injection_values[:-look_back]
    training_added_production_values=training_added_production_values[:-1]
    testing_added_production_values=testing_added_production_values[:-look_back]
    
    #Normalize
    training_injection_data_reshaped=np.reshape(training_added_injection_values, (len(training_added_injection_values),1))
    scaler = MinMaxScaler(feature_range=(0,1))
    train_injection_data = scaler.fit_transform(training_injection_data_reshaped)
    
    training_production_data_reshaped=np.reshape(training_added_production_values, (len(training_added_production_values),1))
    scaler = MinMaxScaler(feature_range=(0,1))
    train_production_data = scaler.fit_transform(training_production_data_reshaped)
    
    testing_injection_data_reshaped=np.reshape(testing_added_injection_values, (len(testing_added_injection_values),1))
    scaler = MinMaxScaler(feature_range=(0,1))
    test_injection_data = scaler.fit_transform(testing_injection_data_reshaped)
    
    testing_production_data_reshaped=np.reshape(testing_added_production_values, (len(testing_added_production_values),1))
    scaler = MinMaxScaler(feature_range=(0,1))
    test_production_data = scaler.fit_transform(testing_production_data_reshaped)
    
    
    return train_injection_data, test_injection_data, train_production_data, test_production_data


def PreprocessingDeformationDataset(deformation_images_augmented,training_percentage, look_back):
    #Function to split dataset, normalize it, and create an input X and output Y.
    
    print('Preprocessing Deformation Dataset ...')
    
    #Think about getting a library to augment image dataset
    #Realistically though, you should probably build some simple model (Dense/CNN+LSTM)
    #without incorporating prod/inj data and look at the results maybe compared to 
    #a baseline linear model. If performance is bad, look into augmentation.
    #It'd be nice to build many models with and without prod and inj and looking at
    #effect of incorporating that data. Makes a nice table and nice plots I think.
    
    #Split entire image dataset into training and testing.
    train,test=TrainTestSplit(deformation_images_augmented, training_percentage)
    #Normalize each of the train and test sets
    train_normalized, train_scaler = NormalizeDataset(train)
    test_normalized, test_scaler = NormalizeDataset(test)
    
    #Create an input X and an output Y for each of the train and test sets
    trainX, trainY = CreateInputOutput(train_normalized,look_back)
    testX, testY = CreateInputOutput(test_normalized,look_back)
    
    return trainX, trainY, testX, testY, train_scaler, test_scaler

def InverseNormalizeDataset(dataset,scaler):
    #Function to invert values back to original scaling
    
    dataset_reshaped = np.reshape(dataset,(len(dataset),x_pixels*y_pixels))
    inverted_dataset = scaler.inverse_transform(dataset_reshaped)
    inverted_dataset = np.reshape(inverted_dataset,(len(inverted_dataset),y_pixels,x_pixels))
    
    return inverted_dataset

def NormalizeDataset(dataset):
    #Function to normalize dataset to a range of 0,1
    
    dataset_reshaped = np.reshape(dataset,(len(dataset),x_pixels*y_pixels))
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset_normalized = scaler.fit_transform(dataset_reshaped)
    dataset_normalized = np.reshape(dataset_normalized,(len(dataset_normalized),y_pixels,x_pixels,1))
    
    return dataset_normalized,scaler

def CreateInputOutput(dataset,look_back):
    #Function to create trainX, trainY, testX, testY
    
    number_of_samples = len(dataset) - look_back
    
    
    X = np.zeros((number_of_samples,look_back,y_pixels,x_pixels,1))
    Y = np.zeros((number_of_samples,y_pixels,x_pixels,1))
    
    for k in range(number_of_samples):
        for i in range(look_back):
            X[k,i]=dataset[k+i]
        Y[k]=dataset[k+look_back]
    
    
    return X,Y

def TrainTestSplit (dataset,training_percentage):
    #Function that splits the dataset into train and test given a percentage.
    train_size = int(len(dataset)*training_percentage)
    train = dataset[0:train_size]
    test = dataset[train_size:len(dataset)]
    
    train = np.reshape(train,(np.shape(train)[0],np.shape(train)[1],np.shape(train)[2],1))
    test = np.reshape(test,(np.shape(test)[0],np.shape(test)[1],np.shape(test)[2],1))
    
    return train,test


def PlotAverageTimeSeries():
    #Function to plot the average deformation timeseries (of all pixels)
    #I used this function mainly to tune the smoothing coefficient to use in the Splining
    #I chose the s value that just barely produces a 'sane' linear fit from this function.
    #so, a value of around 700. I am hoping that this will allow me to beat the baseline
    #easier. Note that a value of 700 does still produce weird splining of some pixels
    #like i=25 j=25 but overall (across all pixels), it does an okay job.
    #Actually, nvm, with s=700, the linear baseline model does horribly. I resorted to enforcing an
    #up_down_limit of 8 and automatically assigning a smoothing coefficient for each pixel
    #based on that up_down_limit, and it works great and makes sense.
    
    average_deformation_time_series=[]
    for k in range (len(deformation_images_augmented)):
        average_deformation_time_series.append(np.average(deformation_images_augmented[k]))
    plt.plot(complete_times,average_deformation_time_series,'x',color='Blue',label='Average Time Series Deformation')
    plt.ylabel('LOS Deformation (mm)')
    plt.title('Deformation Time Series for All Pixels (Averaged)')
    plt.legend(loc=0)
    plt.show()
    
    return
    


def SplineDeformationData():
    #Function that applies a spline interpolation on the dataset to regularize it
    #and also inadvertently augmenting it.
    
    print('Splining Deformation Data ...')
    
    times_in_int_format=[]
    for i in range(len(times)):
        time_difference=times[i]-times[0]
        times_in_int_format.append(time_difference.days)
    
    #Making a list of ALL the times in int format based on the sampling period.
    complete_times_in_int_format=list(range(times_in_int_format[0],times_in_int_format[-1]+sampling_period,sampling_period))
    #Initialize the new Image Dataset which will additionally contain the filled out time gaps.
    deformation_images_boosted=np.zeros(shape=(len(complete_times_in_int_format),y_pixels,x_pixels))
    #Convert the complete times list from integer format to datetime format (plotting purposes).
    complete_times=[]
    for i in range(len(complete_times_in_int_format)):
        time_difference_in_days=datetime.timedelta(days=complete_times_in_int_format[i])
        complete_times.append(times[0]+time_difference_in_days)
    
    
    smoothing_coefficient = 100
    smoothing_coefficient_increment_increase = 50
    up_down_limit = 8
    
    for i in range(y_pixels):
        for j in range(x_pixels):
            
            #There are some pixels with weird time series but if I fix them
            #then the test error will be much lower than my train error
            #This pixel is annoying so I have to manually set it
            # if i==35 and j==63:
            #     smoothing_coefficient=900
            #     tck=splrep(times_in_int_format,deformation_images[:,i,j],s=smoothing_coefficient)
            #     pixel_timeseries=BSpline(*tck)(complete_times_in_int_format)
            #     deformation_images_boosted[:,i,j]=pixel_timeseries
            #This pixel is also not super great
            # elif i==6 and j==83:
            #     smoothing_coefficient=1100
            #     tck=splrep(times_in_int_format,deformation_images[:,i,j],s=smoothing_coefficient)
            #     pixel_timeseries=BSpline(*tck)(complete_times_in_int_format)
            #     deformation_images_boosted[:,i,j]=pixel_timeseries
                
            # else:
                
            tck=splrep(times_in_int_format,deformation_images[:,i,j],s=smoothing_coefficient)
            pixel_timeseries=BSpline(*tck)(complete_times_in_int_format)
            while count_trend_changes(pixel_timeseries, up_down_limit) == False:
                smoothing_coefficient += smoothing_coefficient_increment_increase
                tck=splrep(times_in_int_format,deformation_images[:,i,j],s=smoothing_coefficient)
                pixel_timeseries=BSpline(*tck)(complete_times_in_int_format)
                
            deformation_images_boosted[:,i,j]=pixel_timeseries
            smoothing_coefficient = 100
    
    
    #Plotting
    i=70
    j=40
    plt.figure(figsize=(9,5))
    plt.plot(times,deformation_images[:,i,j],'x',color='blue',label='Actual Deformation')
    plt.plot(complete_times,deformation_images_boosted[:,i,j],color='red',label='Interpolated Deformation')
    plt.ylabel('LOS Deformation (mm)')
    #plt.title('Deformation Time Series for Pixel: i=%d j=%d' %(i,j))
    plt.legend(loc=0)
    plt.grid()
    plt.savefig('interpolation_example.pdf')
    plt.show()
    
    
    return deformation_images_boosted, complete_times


def count_trend_changes(time_series, up_down_limit):
    # Find local minima and maxima indices
    local_minima_idx = argrelextrema(time_series, comparator=lambda x, y: x < y, order=1)[0]
    local_maxima_idx = argrelextrema(time_series, comparator=lambda x, y: x > y, order=1)[0]

    # Count the number of local minima and maxima
    num_minima = len(local_minima_idx)
    num_maxima = len(local_maxima_idx)

    # Count the number of times the series changes from increasing to decreasing or vice versa
    trend_changes = num_minima + num_maxima

    # Check if the trend changes meet the specified up_down_limit
    return trend_changes <= up_down_limit


def GettingProdInjData():
    #Function that imports the monthly prod/inj data starting from 1/1/1975
    
    print('Getting Prod/Inj Data ...')
    
    #Get production/injection data downloaded through conservation.ca.gov
    production_injection_file_location='C:\\users\\user\\Desktop\\Rundle Research\\Research Projects\\The geysers geothermal project (potential paper)\\Correlation data\\geysers injection and production.csv'
    prod_inj_df=pd.read_csv(production_injection_file_location)
    #Extract values from start of 1975 to end of 2022 from dataframe
    production_data_monthly=[]
    injection_data_monthly=[]
    #Index 596 is where the 2017/5/1 value starts (Same time as deformation images)
    for i in range (596,656):
        production_string_to_int=int(prod_inj_df['Unnamed: 3'][i].replace(',',''))
        production_data_monthly.append(production_string_to_int)
        injection_string_to_int=int(prod_inj_df['Unnamed: 4'][i].replace(',',''))
        injection_data_monthly.append(injection_string_to_int)
        
    return production_data_monthly, injection_data_monthly, prod_inj_df

def GettingDeformationData():
    #Function that imports deformation data and returns the times and deformation images
    #after averaging out the irregular NaN values.
    
    print('Getting Deformation Data ...')
    
    #Path to all the txt files containing individual pixel deformation data
    file_path='C:\\Users\\user\\Desktop\\Rundle Research\\Research Projects\\The geysers geothermal project (potential paper)\\Deformation data\\*txt'
    
    #Obtaining all the individual file paths of each txt file
    file_paths = glob.glob(file_path)
    
    #Reading all the pixel data and assigning them to a list where each element
    #in this list has all the data of one pixel
    pixels_data=[]
    for filepath in file_paths:
        pixels_data.append(ReadLines(filepath))
        
    #Index 8 is where the deformation values begin in each pixel
    starting_index=8
    
    #The number of images is the number of deformation data in one pixel
    number_of_images = len(pixels_data[0]) - starting_index
    
    #The number of pixels in x direction (left to right) and y direction (top to bottom)
    #retrieved from the x/y information in the pixel_data
    
    #Putting all the deformation times in a list (each time corresponds to one image)
    times=[]
    for j in range(0,number_of_images):
        time_and_value=pixels_data[0][j+starting_index].split()
        times.append(datetime.datetime.strptime(time_and_value[0],"%Y%m%d"))
        
    #Deformation images have shape : (nb of images, nb of rows, nb of cols)
    #Assigning the pixel deformation values to the deformation matrix image by image.
    deformation_images=np.zeros((number_of_images,y_pixels,x_pixels))    
    for k in range(number_of_images):
        counter=0
        for j in range(x_pixels):
            for i in range(y_pixels):
                time_and_value=pixels_data[counter][k+starting_index].split()
                deformation_images[k][i][j]=time_and_value[1]
                counter+=1
                
                
    #Getting rid of Nan values (they exist at irregular times in random pixels) using
    #2nd order polynomial interpolation. (Does a good job visually)
    #With this interpolation method, the last 2 timeseries points cannot be interpolated.
    #To solve this problem, I linearly interpolated the last Nan value (if it exists)
    #and then I performed a 2nd order polynomial interpolation to fill the other Nan values
    #Previously, I did a local average for the last Nan value and it didn't do so well
    #for the pixels (35,162) and (36,163) (which i found out when noticing a very high
    #residual after building my machine learning model). The reason why it was bad was
    #because there is a big difference in values in the local vicinity of the Nan values.
    #So, it is better to rely on each pixel's behaviour rather than the nearby pixels' values.
    
    #Turn times into an integer list to use in Linear Model fit to
    #interpolate the last Nan value
    times_in_int_format=[]
    for i in range(len(times)):
        time_difference=times[i]-times[0]
        times_in_int_format.append(time_difference.days)
    times_in_int_format=np.array(times_in_int_format)
    
    deformation_images_new=np.zeros((number_of_images,y_pixels,x_pixels))
    
    last_image=deformation_images[-1]
    for i in range(y_pixels):
        for j in range(x_pixels):
            #Check if time series has Nan as its last value
            if np.isnan(last_image[i][j]):
                
                
                #Locate the Nan indices to be removed so I can make Linear Model
                nan_indices = np.where(np.isnan(deformation_images[:,i,j]))[0]
                new_values=np.delete(deformation_images[:,i,j],nan_indices)
                new_times_in_int_format=np.delete(times_in_int_format,nan_indices)
                
                #Build Linear model
                linear_model=LinearRegression().fit(new_times_in_int_format.reshape(-1,1),new_values)
                a=linear_model.coef_
                b=linear_model.intercept_
                #Fit Linear model to the old times in int format
                linearly_fit_data=a*times_in_int_format.reshape(-1,1)+b
                #Assign the last value to the previously Nan value
                deformation_images[-1][i][j]=linearly_fit_data[-1][0]
                
                #Now, I can do the 2nd order interpolation since I have that last value
                pixel_timeseries=deformation_images[:,i,j]
                pixel_timeseries=pd.Series(data=pixel_timeseries,index=times)
                pixel_timeseries=pixel_timeseries.interpolate(method='polynomial',order=2)
                pixel_timeseries=pixel_timeseries.to_numpy()
                deformation_images_new[:,i,j]=pixel_timeseries
                
                
            else:
                pixel_timeseries=deformation_images[:,i,j]
                pixel_timeseries=pd.Series(data=pixel_timeseries,index=times)
                pixel_timeseries=pixel_timeseries.interpolate(method='polynomial',order=2)
                pixel_timeseries=pixel_timeseries.to_numpy()
                deformation_images_new[:,i,j]=pixel_timeseries
            
    #Visualizing an image
    # plt.imshow(deformation_images[87],cmap='inferno',extent=[-122.88983,-122.68983,38.74039,38.86039])
    # plt.colorbar()
    # plt.show()
    
    return times,deformation_images_new

def ReadLines (file_path):
    #Function that reads all the lines of a txt file. The first 8 lines of each txt file
    #are just information about the time series. The actual displacements/times start from index 8.
    with open(file_path) as file:
        lines = file.readlines()
        
    return lines



if __name__ == '__main__':
    
    
    production_data_monthly,injection_data_monthly,prod_inj_df=GettingProdInjData()
    
    times,deformation_images=GettingDeformationData()
    
    deformation_images_augmented,complete_times=SplineDeformationData()
    
    # PlotAverageTimeSeries()
    
    trainX,trainY,testX,testY,train_scaler,test_scaler=PreprocessingDeformationDataset(deformation_images_augmented,training_percentage, look_back)
    
    train_injection_data,test_injection_data,train_production_data,test_production_data=PreprocessingGeothermalDataset()
    
    collection_of_variables={
        'deformation_images_augmented': deformation_images_augmented,
        'times': times,
        'complete_times': complete_times,
        'trainX': trainX,
        'trainY': trainY,
        'train_scaler': train_scaler,
        'testX': testX,
        'testY': testY,
        'test_scaler': test_scaler,
        'train_injection_data': train_injection_data,
        'test_injection_data': test_injection_data,
        'train_production_data': train_production_data,
        'test_production_data': test_production_data}
    
    with open('preprocessed_data_dict.pkl', 'wb') as file:
        pickle.dump(collection_of_variables, file)
    
    
    
    
    running_time=datetime.datetime.now()-start
    print('Running time is: ' + str(running_time))
    
    
    