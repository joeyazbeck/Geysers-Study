from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, LSTM, Flatten, TimeDistributed, Reshape, Concatenate
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from Geysers_Preprocessing import InverseNormalizeDataset, TrainTestSplit, CreateInputOutput
import pickle
from tqdm import tqdm #PROGRESS BAR (IAN'S IDEA) #Can't get it to work on a function


#My goal for today (1/25/2023) is to create fake injection/production and interferogram data
#and test it out on a custom model built using Keras Functional API in hopes of
#understanding how to set up custom models in the future.
#My goal for today (3/21/2023) is to designate an area of study within the Geysers and use
#the LOS data to predict deformation. I guess I have to look at the LiCSBAS results now
#and try to export the data first.
#My goal for today (4/17/2023) is to use the LiCSBAS data that I downloaded and incorporate
#them into some kind of model.
#My goal for today and tomorrow (5/2/2023) is to prep the dataset ahead of ML model
#implementation at end of week.
#My goal for the next few days (5/11/2023) is to figure out the right architecture.
#Today (6/8/2023), I was finally able to figure out the
#best smoothing/regularization/augmentation method for my dataset, and I am ready to
#implement my model that will beat the baseline and make a bunch of plots FINALLY.
#Today (6/21/2023), I am going to have some fun and incorporate geothermal data
#kind of randomly to see how it works out. I gotta make sure that the dimensionality
#of the geothermal input matches that of the deformation input because if you think about it,
#the neural network has to run the inputs at the same time so if I have a total of 267
#samples of InSAR images to run through, there should be an accompanying 267 samples of
#geothermal data where each sample bears significance to each sample of the InSAR set.
#Today (6/27/2023), I managed to incorporate geothermal data in my model, and I think the
#train error is lower than baseline and the LSTMCNN model. I still have to check the test
#error, but I have high hopes...
#Today (7/13/2023), I am struggling to have a model with injection data to beat the one
#without injection data, so I am going to try and incorporate production data as well, and
#see if that helps.
#Today (7/23/2023), I streamlined my code and split the preprocessing and ML learning parts
#into different scripts. So, now the preprocessing saves all the variables I need, and
#I just load them up in the ML script (this one).
#Today (7/25/2023), I came up with a method to optimize the smoothing_coefficient for
#every pixel in the deformation data, and I am happy with it.





    # inputs=Input(shape=(look_back,121,201,1))
    # x=TimeDistributed(Conv2D(5,3))(inputs)
    # x=TimeDistributed(MaxPooling2D(2))(x)
    # x=TimeDistributed(Conv2D(5,3))(x)
    # x=TimeDistributed(MaxPooling2D(2))(x)
    # x=Reshape((x.shape[1],x.shape[2]*x.shape[3],x.shape[4]))(x)
    # x=TimeDistributed(LSTM(16))(x)
    # x=Flatten()(x)
    # x=Dense(50)(x)
    # x=Dense(121*201)(x)
    # x=Reshape((121,201,1))(x)



start=datetime.datetime.now()

x_pixels=201 #2056-1856+1
y_pixels=121 #2596-2476+1

#Best parameters I think (although I ended up using batch size of 64 since it's faster)
#0.9
#1
#32
#200

training_percentage=0.90 #(HAS TO MATCH THE ONE IN PREPROCESSING)
look_back=1 #using the last X images (HAS TO MATCH THE ONE IN PREPROCESSING)
batch_size=64
epochs=5

def PlotScatter(testY,test_scaler,testPredict_A,testPredict_B):
    
    #PLOTTING FOR PAPER (START)
    
    #Going to average each image into one data point for the scatter plot.
    testY=InverseNormalizeDataset(testY, test_scaler)
    actual_values=[]
    predicted_values_A=[]
    predicted_values_B=[]
    
    for i in range(len(testY)):
        actual_values.append(np.average(testY[i]))
        predicted_values_A.append(np.average(testPredict_A[i]))
        predicted_values_B.append(np.average(testPredict_B[i]))
    
    #Initializing figure
    fig,ax=plt.subplots(1,2)
    fig.set_size_inches(15,6)
    fig.subplots_adjust(wspace=0.3)
    #y=x line
    x=[]
    y=[]
    x.append(np.min(actual_values))
    x.append(np.max(actual_values))
    y=x
    #Linear fits of each model
    p_A=np.polyfit(actual_values,predicted_values_A,1)
    first_value_A=p_A[0]*np.min(actual_values)+p_A[1]
    last_value_A=p_A[0]*np.max(actual_values)+p_A[1]
    line_fit_A_y=[]
    line_fit_A_y.append(first_value_A)
    line_fit_A_y.append(last_value_A)
    line_fit_A_x=[]
    line_fit_A_x.append(np.min(actual_values))
    line_fit_A_x.append(np.max(actual_values))
    
    p_B=np.polyfit(actual_values,predicted_values_B,1)
    first_value_B=p_B[0]*np.min(actual_values)+p_B[1]
    last_value_B=p_B[0]*np.max(actual_values)+p_B[1]
    line_fit_B_y=[]
    line_fit_B_y.append(first_value_B)
    line_fit_B_y.append(last_value_B)
    line_fit_B_x=[]
    line_fit_B_x.append(np.min(actual_values))
    line_fit_B_x.append(np.max(actual_values))
    
    #Plotting
    ax[0].scatter(actual_values, predicted_values_A, color='blue', marker='x')
    ax[0].plot(x,y,color='black',label='y=x')
    ax[0].plot(line_fit_A_x,line_fit_A_y,color='red',label='y = %.3f x - %.3f' %(p_A[0],np.absolute(p_A[1])))
    ax[0].set_xlabel('Actual Values')
    ax[0].set_ylabel('Predicted Values')
    ax[0].legend(loc='upper left')
    ax[0].grid(True)
    ax[0].set_title('Model A')
    
    ax[1].scatter(actual_values, predicted_values_B, color='blue', marker='x')
    ax[1].plot(x,y,color='black',label='y=x')
    ax[1].plot(line_fit_B_x,line_fit_B_y,color='red',label='y = %.3f x - %.3f' %(p_B[0],np.absolute(p_B[1])))
    ax[1].set_xlabel('Actual Values')
    ax[1].set_ylabel('Predicted Values')
    ax[1].legend(loc='upper left')
    ax[1].grid(True)
    ax[1].set_title('Model B')
    
    
    plt.rcParams.update({'font.size': 15})
    plt.savefig('scatter.pdf')
    plt.show()

    
    #PLOTTING FOR PAPER (END)
    
    return


def BuildLSTMCNNWithInjectionAndProductionData(trainY,testY,plot_performance):
    
    print('Incorporating Geothermal Injection and Production Data ...')
    
    
    inputs1=Input(shape=(look_back,y_pixels,x_pixels,1))
    x=TimeDistributed(Conv2D(16,3))(inputs1)
    x=TimeDistributed(MaxPooling2D(2))(x)
    # x=TimeDistributed(Conv2D(16,3))(x)
    # x=TimeDistributed(MaxPooling2D(2))(x)
    x=Reshape((x.shape[1],x.shape[2]*x.shape[3],x.shape[4]))(x)
    x=TimeDistributed(LSTM(64))(x)
    x=Flatten()(x)
    inputs2=Input(shape=(1))
    x2=Dense(100)(inputs2)
    inputs3=Input(shape=(1))
    x3=Dense(100)(inputs3)
    x=Concatenate(axis=1)([x,x2,x3])
    # x=Dense(50)(x)
    x=Dense(y_pixels*x_pixels)(x)
    x=Reshape((y_pixels,x_pixels,1))(x)
    
    model=keras.Model(inputs=[inputs1,inputs2,inputs3],outputs=x)
    # model.summary()
    # keras.utils.plot_model(model,'Insarwithgeomodel.pdf',show_shapes=True)
    
    model.compile(optimizer='adamax', loss='mse')
    
    model.fit([trainX,train_injection_data,train_production_data],trainY,batch_size=batch_size,epochs=epochs,verbose=0)
    
    #Predict
    trainPredict = model.predict([trainX,train_injection_data,train_production_data])
    testPredict = model.predict([testX,test_injection_data,test_production_data])
    
    #Inverse normalize predictions and targets
    trainPredict=InverseNormalizeDataset(trainPredict, train_scaler)
    trainY=InverseNormalizeDataset(trainY, train_scaler)
    testPredict=InverseNormalizeDataset(testPredict, test_scaler)
    testY=InverseNormalizeDataset(testY, test_scaler)
    
    #Calculate error by averaging MSE's of all the images
    train_error,train_error_std=CalculateError(trainY, trainPredict)
    test_error,test_error_std=CalculateError(testY, testPredict)
    print('Injection and Production Train error is: %.2f \u00B1 %.2f' %(train_error,train_error_std))
    print('Injection and Production Test error is: %.2f \u00B1 %.2f' %(test_error,test_error_std))
    
    
    if plot_performance == True:
        
        minlatitude=38.74
        maxlatitude=38.86
        minlongitude=-122.89
        maxlongitude=-122.69
        
        
        plt.figure(figsize=(70,50))
        matplotlib.rcParams.update({'font.size': 35})
        im_ratio = y_pixels/x_pixels
        residuals=[]
        
        plt.subplot(3,3,1)
        plt.imshow(testY[0],cmap='jet',extent=[minlongitude,maxlongitude,minlatitude,maxlatitude])
        plt.colorbar(fraction=0.046*im_ratio)
        num_ticks=5
        x_ticks = np.linspace(minlongitude+((maxlongitude-minlongitude)*0.05), maxlongitude-((maxlongitude-minlongitude)*0.05), num_ticks)
        plt.xticks(x_ticks,rotation=-45)
        plt.title('Data')
        
        plt.subplot(3,3,2)
        plt.imshow(testPredict[0],cmap='jet',vmin=np.min(testY[0]),vmax=np.max(testY[0]),extent=[minlongitude,maxlongitude,minlatitude,maxlatitude])
        plt.colorbar(fraction=0.046*im_ratio)
        num_ticks=5
        x_ticks = np.linspace(minlongitude+((maxlongitude-minlongitude)*0.05), maxlongitude-((maxlongitude-minlongitude)*0.05), num_ticks)
        plt.xticks(x_ticks,rotation=-45)
        plt.title('Model')
        
        plt.subplot(3,3,3)
        residuals.append(testY[0]-testPredict[0])
        plt.imshow(residuals[0],cmap='jet',vmin=-3*np.std(residuals[0]),vmax=3*np.std(residuals[0]),extent=[minlongitude,maxlongitude,minlatitude,maxlatitude])
        plt.colorbar(fraction=0.046*im_ratio)
        num_ticks=5
        x_ticks = np.linspace(minlongitude+((maxlongitude-minlongitude)*0.05), maxlongitude-((maxlongitude-minlongitude)*0.05), num_ticks)
        plt.xticks(x_ticks,rotation=-45)
        plt.title('Residuals')
        
        
        
        
        
        
        plt.subplot(3,3,4)
        plt.imshow(testY[1],cmap='jet',extent=[minlongitude,maxlongitude,minlatitude,maxlatitude])
        plt.colorbar(fraction=0.046*im_ratio)
        num_ticks=5
        x_ticks = np.linspace(minlongitude+((maxlongitude-minlongitude)*0.05), maxlongitude-((maxlongitude-minlongitude)*0.05), num_ticks)
        plt.xticks(x_ticks,rotation=-45)
        
        plt.subplot(3,3,5)
        plt.imshow(testPredict[1],cmap='jet',vmin=np.min(testY[1]),vmax=np.max(testY[1]),extent=[minlongitude,maxlongitude,minlatitude,maxlatitude])
        plt.colorbar(fraction=0.046*im_ratio)
        num_ticks=5
        x_ticks = np.linspace(minlongitude+((maxlongitude-minlongitude)*0.05), maxlongitude-((maxlongitude-minlongitude)*0.05), num_ticks)
        plt.xticks(x_ticks,rotation=-45)
        
        plt.subplot(3,3,6)
        residuals.append(testY[1]-testPredict[1])
        plt.imshow(residuals[1],cmap='jet',vmin=-3*np.std(residuals[1]),vmax=3*np.std(residuals[1]),extent=[minlongitude,maxlongitude,minlatitude,maxlatitude])
        plt.colorbar(fraction=0.046*im_ratio)
        num_ticks=5
        x_ticks = np.linspace(minlongitude+((maxlongitude-minlongitude)*0.05), maxlongitude-((maxlongitude-minlongitude)*0.05), num_ticks)
        plt.xticks(x_ticks,rotation=-45)
        
        
        
        
        
        
        plt.subplot(3,3,7)
        plt.imshow(testY[2],cmap='jet',extent=[minlongitude,maxlongitude,minlatitude,maxlatitude])
        plt.colorbar(fraction=0.046*im_ratio)
        num_ticks=5
        x_ticks = np.linspace(minlongitude+((maxlongitude-minlongitude)*0.05), maxlongitude-((maxlongitude-minlongitude)*0.05), num_ticks)
        plt.xticks(x_ticks,rotation=-45)
        
        plt.subplot(3,3,8)
        plt.imshow(testPredict[2],cmap='jet',vmin=np.min(testY[2]),vmax=np.max(testY[2]),extent=[minlongitude,maxlongitude,minlatitude,maxlatitude])
        plt.colorbar(fraction=0.046*im_ratio)
        num_ticks=5
        x_ticks = np.linspace(minlongitude+((maxlongitude-minlongitude)*0.05), maxlongitude-((maxlongitude-minlongitude)*0.05), num_ticks)
        plt.xticks(x_ticks,rotation=-45)
        
        plt.subplot(3,3,9)
        residuals.append(testY[2]-testPredict[2])
        plt.imshow(residuals[2],cmap='jet',vmin=-3*np.std(residuals[2]),vmax=3*np.std(residuals[2]),extent=[minlongitude,maxlongitude,minlatitude,maxlatitude])
        plt.colorbar(fraction=0.046*im_ratio)
        num_ticks=5
        x_ticks = np.linspace(minlongitude+((maxlongitude-minlongitude)*0.05), maxlongitude-((maxlongitude-minlongitude)*0.05), num_ticks)
        plt.xticks(x_ticks,rotation=-45)
        
        plt.savefig('datavsmodelvsresidualsB.pdf')
        plt.show()
        
        
        
        print(np.average(residuals[1]))
        print(np.std(residuals[1]))
        print(np.max(residuals[1]))
        print(np.min(residuals[1]))
        max_index=np.argmax(residuals[1])
        row_index, col_index = np.unravel_index(max_index, residuals[1].shape)
        print("Index of the highest value:", row_index, col_index)
        min_index=np.argmin(residuals[1])
        row_index, col_index = np.unravel_index(min_index, residuals[1].shape)
        print("Index of the lowest value:", row_index, col_index)
        
        
        
    return model, testPredict

def BuildLSTMCNNWithProductionData(trainY,testY):
    
    print('Incorporating Geothermal Production Data ...')
    
    inputs1=Input(shape=(look_back,y_pixels,x_pixels,1))
    x=TimeDistributed(Conv2D(16,3))(inputs1)
    x=TimeDistributed(MaxPooling2D(2))(x)
    # x=TimeDistributed(Conv2D(16,3))(x)
    # x=TimeDistributed(MaxPooling2D(2))(x)
    x=Reshape((x.shape[1],x.shape[2]*x.shape[3],x.shape[4]))(x)
    x=TimeDistributed(LSTM(64))(x)
    x=Flatten()(x)
    inputs2=Input(shape=(1))
    x2=Dense(100)(inputs2)
    x=Concatenate(axis=1)([x,x2])
    # x=Dense(50)(x)
    x=Dense(y_pixels*x_pixels)(x)
    x=Reshape((y_pixels,x_pixels,1))(x)
    
    model=keras.Model(inputs=[inputs1,inputs2],outputs=x)
    # model.summary()
    #keras.utils.plot_model(model,show_shapes=True)
    
    model.compile(optimizer='adamax', loss='mse')
    
    model.fit([trainX,train_production_data],trainY,batch_size=batch_size,epochs=epochs,verbose=0)
    
    #Predict
    trainPredict = model.predict([trainX,train_production_data])
    testPredict = model.predict([testX,test_production_data])
    
    #Inverse normalize predictions and targets
    trainPredict=InverseNormalizeDataset(trainPredict, train_scaler)
    trainY=InverseNormalizeDataset(trainY, train_scaler)
    testPredict=InverseNormalizeDataset(testPredict, test_scaler)
    testY=InverseNormalizeDataset(testY, test_scaler)
    
    #Calculate error by averaging MSE's of all the images
    train_error,train_error_std=CalculateError(trainY, trainPredict)
    test_error,test_error_std=CalculateError(testY, testPredict)
    print('Production Train error is: %.2f \u00B1 %.2f' %(train_error,train_error_std))
    print('Production Test error is: %.2f \u00B1 %.2f' %(test_error,test_error_std))
    
    
    return model


def BuildLSTMCNNWithInjectionData(trainY,testY):
    
    print('Incorporating Geothermal Injection Data ...')
    
    inputs1=Input(shape=(look_back,y_pixels,x_pixels,1))
    x=TimeDistributed(Conv2D(16,3))(inputs1)
    x=TimeDistributed(MaxPooling2D(2))(x)
    # x=TimeDistributed(Conv2D(5,3))(x)
    # x=TimeDistributed(MaxPooling2D(2))(x)
    x=Reshape((x.shape[1],x.shape[2]*x.shape[3],x.shape[4]))(x)
    x=TimeDistributed(LSTM(64))(x)
    x=Flatten()(x)
    inputs2=Input(shape=(1))
    x2=Dense(100)(inputs2)
    x=Concatenate(axis=1)([x,x2])
    # x=Dense(50)(x)
    x=Dense(y_pixels*x_pixels)(x)
    x=Reshape((y_pixels,x_pixels,1))(x)
    
    model=keras.Model(inputs=[inputs1,inputs2],outputs=x)
    # model.summary()
    #keras.utils.plot_model(model,show_shapes=True)
    
    model.compile(optimizer='adamax', loss='mse')
    
    model.fit([trainX,train_injection_data],trainY,batch_size=batch_size,epochs=epochs,verbose=0)
    
    #Predict
    trainPredict = model.predict([trainX,train_injection_data])
    testPredict = model.predict([testX,test_injection_data])
    
    #Inverse normalize predictions and targets
    trainPredict=InverseNormalizeDataset(trainPredict, train_scaler)
    trainY=InverseNormalizeDataset(trainY, train_scaler)
    testPredict=InverseNormalizeDataset(testPredict, test_scaler)
    testY=InverseNormalizeDataset(testY, test_scaler)
    
    #Calculate error by averaging MSE's of all the images
    train_error,train_error_std=CalculateError(trainY, trainPredict)
    test_error,test_error_std=CalculateError(testY, testPredict)
    print('Injection Train error is: %.2f \u00B1 %.2f' %(train_error,train_error_std))
    print('Injection Test error is: %.2f \u00B1 %.2f' %(test_error,test_error_std))
    
    
    return model


def BuildLSTMCNN(trainY,testY,plot_performance):
    #Function that builds a CNN+LSTM model and prints out the error
    
    print('Building LSTM+CNN model ...')
    
    #The idea of doing 2D CNN+LSTM model. CNN captures spatial patterns in deformation
    #values and the LSTM captures the temporal dependencies between the images
    #and the prediction is the next image in the sequence.
    #I expect it to beat linear baseline because it captures spatial patterns using CNN.
    
    inputs=Input(shape=(look_back,121,201,1))
    x=TimeDistributed(Conv2D(16,3))(inputs)
    x=TimeDistributed(MaxPooling2D(2))(x)
    # x=TimeDistributed(Conv2D(5,3))(x)
    # x=TimeDistributed(MaxPooling2D(2))(x)
    x=Reshape((x.shape[1],x.shape[2]*x.shape[3],x.shape[4]))(x)
    x=TimeDistributed(LSTM(64))(x)
    x=Flatten()(x)
    # x=Dense(50)(x)
    x=Dense(121*201)(x)
    x=Reshape((121,201,1))(x)
    
    model=keras.Model(inputs=inputs,outputs=x)
    # model.summary()
    # keras.utils.plot_model(model,'Insarmodel.pdf',show_shapes=True)
    
    model.compile(optimizer='adamax', loss='mse')
    
    #Use history= in order to save the metrics computed for each epoch so I can maybe plot
    #later as a function of epochs
    
    model.fit(trainX,trainY,batch_size=batch_size,epochs=epochs,verbose=0)
    
    #Predict
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    #Inverse normalize predictions and targets
    trainPredict=InverseNormalizeDataset(trainPredict, train_scaler)
    trainY=InverseNormalizeDataset(trainY, train_scaler)
    testPredict=InverseNormalizeDataset(testPredict, test_scaler)
    testY=InverseNormalizeDataset(testY, test_scaler)
    
    #Calculate error by averaging MSE's of all the images
    train_error,train_error_std=CalculateError(trainY, trainPredict)
    test_error,test_error_std=CalculateError(testY, testPredict)
    print('Train error is: %.2f \u00B1 %.2f' %(train_error,train_error_std))
    print('Test error is: %.2f \u00B1 %.2f' %(test_error,test_error_std))
    
    # train_loss = history.history['loss']
    # plt.plot(range(1, len(train_loss) + 1), train_loss)
    # plt.title('Training Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.show()

    
    
    
    if plot_performance == True:
        
        minlatitude=38.74
        maxlatitude=38.86
        minlongitude=-122.89
        maxlongitude=-122.69
        
        
        plt.figure(figsize=(70,50))
        matplotlib.rcParams.update({'font.size': 35})
        im_ratio = y_pixels/x_pixels
        residuals=[]
        
        plt.subplot(3,3,1)
        plt.imshow(testY[0],cmap='jet',extent=[minlongitude,maxlongitude,minlatitude,maxlatitude])
        plt.colorbar(fraction=0.046*im_ratio)
        num_ticks=5
        x_ticks = np.linspace(minlongitude+((maxlongitude-minlongitude)*0.05), maxlongitude-((maxlongitude-minlongitude)*0.05), num_ticks)
        plt.xticks(x_ticks,rotation=-45)
        plt.title('Data')
        
        plt.subplot(3,3,2)
        plt.imshow(testPredict[0],cmap='jet',vmin=np.min(testY[0]),vmax=np.max(testY[0]),extent=[minlongitude,maxlongitude,minlatitude,maxlatitude])
        plt.colorbar(fraction=0.046*im_ratio)
        num_ticks=5
        x_ticks = np.linspace(minlongitude+((maxlongitude-minlongitude)*0.05), maxlongitude-((maxlongitude-minlongitude)*0.05), num_ticks)
        plt.xticks(x_ticks,rotation=-45)
        plt.title('Model')
        
        plt.subplot(3,3,3)
        residuals.append(testY[0]-testPredict[0])
        plt.imshow(residuals[0],cmap='jet',vmin=-3*np.std(residuals[0]),vmax=3*np.std(residuals[0]),extent=[minlongitude,maxlongitude,minlatitude,maxlatitude])
        plt.colorbar(fraction=0.046*im_ratio)
        num_ticks=5
        x_ticks = np.linspace(minlongitude+((maxlongitude-minlongitude)*0.05), maxlongitude-((maxlongitude-minlongitude)*0.05), num_ticks)
        plt.xticks(x_ticks,rotation=-45)
        plt.title('Residuals')
        
        
        
        
        
        
        plt.subplot(3,3,4)
        plt.imshow(testY[1],cmap='jet',extent=[minlongitude,maxlongitude,minlatitude,maxlatitude])
        plt.colorbar(fraction=0.046*im_ratio)
        num_ticks=5
        x_ticks = np.linspace(minlongitude+((maxlongitude-minlongitude)*0.05), maxlongitude-((maxlongitude-minlongitude)*0.05), num_ticks)
        plt.xticks(x_ticks,rotation=-45)
        
        plt.subplot(3,3,5)
        plt.imshow(testPredict[1],cmap='jet',vmin=np.min(testY[1]),vmax=np.max(testY[1]),extent=[minlongitude,maxlongitude,minlatitude,maxlatitude])
        plt.colorbar(fraction=0.046*im_ratio)
        num_ticks=5
        x_ticks = np.linspace(minlongitude+((maxlongitude-minlongitude)*0.05), maxlongitude-((maxlongitude-minlongitude)*0.05), num_ticks)
        plt.xticks(x_ticks,rotation=-45)
        
        plt.subplot(3,3,6)
        residuals.append(testY[1]-testPredict[1])
        plt.imshow(residuals[1],cmap='jet',vmin=-3*np.std(residuals[1]),vmax=3*np.std(residuals[1]),extent=[minlongitude,maxlongitude,minlatitude,maxlatitude])
        plt.colorbar(fraction=0.046*im_ratio)
        num_ticks=5
        x_ticks = np.linspace(minlongitude+((maxlongitude-minlongitude)*0.05), maxlongitude-((maxlongitude-minlongitude)*0.05), num_ticks)
        plt.xticks(x_ticks,rotation=-45)
        
        
        
        
        
        
        plt.subplot(3,3,7)
        plt.imshow(testY[2],cmap='jet',extent=[minlongitude,maxlongitude,minlatitude,maxlatitude])
        plt.colorbar(fraction=0.046*im_ratio)
        num_ticks=5
        x_ticks = np.linspace(minlongitude+((maxlongitude-minlongitude)*0.05), maxlongitude-((maxlongitude-minlongitude)*0.05), num_ticks)
        plt.xticks(x_ticks,rotation=-45)
        
        plt.subplot(3,3,8)
        plt.imshow(testPredict[2],cmap='jet',vmin=np.min(testY[2]),vmax=np.max(testY[2]),extent=[minlongitude,maxlongitude,minlatitude,maxlatitude])
        plt.colorbar(fraction=0.046*im_ratio)
        num_ticks=5
        x_ticks = np.linspace(minlongitude+((maxlongitude-minlongitude)*0.05), maxlongitude-((maxlongitude-minlongitude)*0.05), num_ticks)
        plt.xticks(x_ticks,rotation=-45)
        
        plt.subplot(3,3,9)
        residuals.append(testY[2]-testPredict[2])
        plt.imshow(residuals[2],cmap='jet',vmin=-3*np.std(residuals[2]),vmax=3*np.std(residuals[2]),extent=[minlongitude,maxlongitude,minlatitude,maxlatitude])
        plt.colorbar(fraction=0.046*im_ratio)
        num_ticks=5
        x_ticks = np.linspace(minlongitude+((maxlongitude-minlongitude)*0.05), maxlongitude-((maxlongitude-minlongitude)*0.05), num_ticks)
        plt.xticks(x_ticks,rotation=-45)
        
        plt.savefig('datavsmodelvsresidualsA.pdf')
        plt.show()
        
    
    
    
    
    
    return model, testPredict

def LinearBaseline(deformation_images,trainY,train_scaler,testY,test_scaler):
    #Function that fits a linear model to each pixel of the augmented dataset
    #and predicts based on that linear fit.
    
    print('Building Linear Baseline Model ...')
    
    #Do linear fit to the *augmented* dataset (a linear fit of a linear fit essentially)
    #Now that I think about it, I think it would have been the same to use an array
    #of 0,1,2,3,4,5,6,7,8... instead of complete_times_in_int_format which is 0,6,12,18,24,..1788
    complete_times_for_linear_model=list(range(0,len(deformation_images)))
    complete_times_for_linear_model=np.array(complete_times_for_linear_model)
    complete_times_for_linear_model_multiplication=complete_times_for_linear_model
    complete_times_for_linear_model=complete_times_for_linear_model.reshape((-1,1))
    
    
    linear_baseline=np.zeros((np.shape(deformation_images)))
    for i in range(y_pixels):
        for j in range(x_pixels):
            #Get the deformation timeseries
            pixel_timeseries=deformation_images[:,i,j]
            #Fit a line to it
            linear_model=LinearRegression().fit(complete_times_for_linear_model,pixel_timeseries)
            a=linear_model.coef_
            b=linear_model.intercept_
            linear_baseline[:,i,j]=a*complete_times_for_linear_model_multiplication+b
    
    linear_baseline_train,linear_baseline_test = TrainTestSplit(linear_baseline, training_percentage)
    
    #The Y is the one that I care about only.
    linear_baseline_trainX,linear_baseline_trainY = CreateInputOutput(linear_baseline_train, look_back)
    linear_baseline_testX,linear_baseline_testY = CreateInputOutput(linear_baseline_test, look_back)
    #Dropping the extra dimension (function is primarily used to prep dataset for ML)
    linear_baseline_trainY=np.reshape(linear_baseline_trainY,(np.shape(linear_baseline_trainY)[0],np.shape(linear_baseline_trainY)[1],np.shape(linear_baseline_trainY)[2]))
    linear_baseline_testY=np.reshape(linear_baseline_testY,(np.shape(linear_baseline_testY)[0],np.shape(linear_baseline_testY)[1],np.shape(linear_baseline_testY)[2]))
    #Inverse normalize original trainY and testY I guess because I am not doing a function
    OG_trainY_for_baseline=InverseNormalizeDataset(trainY, train_scaler)
    OG_testY_for_baseline=InverseNormalizeDataset(testY, test_scaler)
    
    baseline_train_error,baseline_train_error_std=CalculateError(OG_trainY_for_baseline, linear_baseline_trainY)
    baseline_test_error,baseline_test_error_std=CalculateError(OG_testY_for_baseline, linear_baseline_testY)
    print('Baseline train error to beat is: %.2f \u00B1 %.2f' %(baseline_train_error,baseline_train_error_std))
    print('Baseline test error to beat is: %.2f \u00B1 %.2f' %(baseline_test_error,baseline_test_error_std))
    
    return

def CalculateError(target,prediction):
    #Calculates MSE of each image and averages all MSE's to get one error with a std dev.
    
    mse_list=[]
    for k in range(len(target)):
        mse=mean_squared_error(target[k], prediction[k])
        mse_list.append(mse)
    average_error=np.mean(mse_list)
    std_error=np.std(mse_list)
    
    return average_error,std_error


if __name__ == "__main__":
    
    
    with open('preprocessed_data_dict.pkl', 'rb') as file:
        preprocessed_data_dict = pickle.load(file)
    
    deformation_images_augmented = preprocessed_data_dict['deformation_images_augmented']
    times = preprocessed_data_dict['times']
    complete_times = preprocessed_data_dict['complete_times']
    trainX = preprocessed_data_dict['trainX']
    trainY = preprocessed_data_dict['trainY']
    train_scaler = preprocessed_data_dict['train_scaler']
    testX = preprocessed_data_dict['testX']
    testY = preprocessed_data_dict['testY']
    test_scaler = preprocessed_data_dict['test_scaler']
    train_injection_data = preprocessed_data_dict['train_injection_data']
    test_injection_data = preprocessed_data_dict['test_injection_data']
    train_production_data = preprocessed_data_dict['train_production_data']
    test_production_data = preprocessed_data_dict['test_production_data']
    
    
    LinearBaseline(deformation_images_augmented,trainY,train_scaler,testY,test_scaler)
    
    plot_performance = False
    
    model1, testPredict_A = BuildLSTMCNN(trainY, testY, plot_performance)
    
    # model2 = BuildLSTMCNNWithInjectionData(trainY,testY)
    
    # model3 = BuildLSTMCNNWithProductionData(trainY,testY)
    
    model4, testPredict_B = BuildLSTMCNNWithInjectionAndProductionData(trainY,testY,plot_performance)
    
    PlotScatter(testY, test_scaler, testPredict_A, testPredict_B)
    
    
    
    running_time=datetime.datetime.now()-start
    print('Running time is: ' + str(running_time))
    
    
    
    
    
    
    
    
    
    
    
    
    
    