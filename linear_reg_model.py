#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and loading Data using Pandas Library

# In[20]:


import pandas as pd
import numpy as np
import seaborn as sns;sns.set()
import matplotlib.pyplot as plt

plt.style.use("ggplot")
sns.set(style="ticks")

data = pd.read_csv("/Users/MuditPaliwal/Desktop/Semester 2/FML/Home work/hour.csv")
#importing data using pandas library
data['formatted_date'] = pd.to_datetime(data['dteday'])

data['Week_Number'] = pd.to_datetime(data['dteday']).dt.week


# In[21]:


#Taking out Data info and printing first Few cells of the data
data.head()


# # Creating a Auto - Correlation matrix heatmap for the Data Set

# In[3]:


data = data.drop(['formatted_date',"dteday"],1)
data1 = data-data.mean()
data2 = data1.T@data1
data2 = data2/17379

sd = data.std()
sd1 = np.multiply.outer(sd.to_numpy(),sd.to_numpy())

final = data2/sd1
f, ax = plt.subplots(figsize=(20,15))
ax = sns.heatmap(final,vmin=-0.25, vmax=1,annot=True, fmt="f",linewidths=.5)


# # Plot for Each Feature Vs the count of Total bike rentals

# In[4]:


g = sns.jointplot("instant", "cnt", data=data, kind="reg",color='g')
print("This feature has no predictive power as it's only function is to act as record number of instances i.e Record index.")


# In[5]:


g = sns.jointplot("temp", "cnt", data=data, kind="reg",color='g')
g = sns.jointplot("atemp", "cnt", data=data, kind="reg",color='g')
print ("As seen from the joint point below both TEMPRATURE to usage and ADJUSTED TEMPERATURE to usage have some correlation \nwith the total number of rental bikes but linear fit is'nt the best way to fit the relationship or find the best fit curve.This should be intutuive, as warmer the temperature the more bikes get rented.There is also a dip at the end for the total number of bikes rented.Once again this should make sense as the temperature outside gets too hot, people tend to rent the bike less.As \ncan be seen from the distribution of both graph, they tend to be similar and presents a redundancy to the features selected.So either the two could be combined or we can drop one of the two features. ")


# In[6]:


g = sns.jointplot("hum", "cnt", data=data, kind="reg",color='g')
print("Observing the Jointplot for humidity, we can see there is a negative correlation or inverse realtionship with the total number of registered bikes with a linear fit bieng very close to best curve fit with all the plots except for outliers in low \nhumidity.This can be clearly visualized ad high humidity bieng a deterrent for rent of bike cycles.This situation can also be \nrelated to the weather Categorial plot where heavy rainfall has a direct impact of renting of bike cycles.")


# In[7]:


g = sns.jointplot("windspeed", "cnt", data=data, kind="reg",color='g')
print("As high wind makes it difficult to peddle the bike so we can use this data to help determine the function of usage but after observing the Windspeed visualizations, it shows little interpreation with affect it has on the bike rental usage.The wind \nspeed clearly shows a strong correlation with the total number of bike rentals.")


# In[8]:


sns.catplot(x="weathersit", y="cnt",data=data);
print("1: Clear, Few clouds, Partly cloudy, Partly cloudy\n2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist\n3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds\n4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog\n\nThis attribute will help us capture the effect of weather outlook on rider's behaviour.The weather situation plot is explicitly clear.People tend to rent bikes more on clear weather and decreases as the weather situation worsens.Thic could be correlated \nwith our humidity plot where high humidity tends to lead to a situation of rainfall and acts as a deterrent to rental of bike \ncycles.")


# In[9]:


sns.catplot(x="season", y="cnt", data=data);
print('Season:\n1:spring\n2:summer\n3:fall\n4:winter \n\nThis attribute helped us to visualise the effect of seasonality on the bike rental usage.We observe that the highest rentals has been in the 3rd season(Fall) followed by summer and then winter, which gives us the inutuive belief that riders prefer to ride in warm to pleasant climates in comparison to colder climates.')


# In[10]:


sns.catplot(x="yr", y="cnt", data=data);
print("Looking at the Year Variable , it is observed that the total count of rental bike usage has gone up form Year 1 to Year 2 which could suggest that the system grew in popularity and information about the system was distributed and communicated well.It is important to note that this is only useful when we are making predictions for the year in the range (2011-12), because for fututre dates the algorithm has to be extrapolated significantly and will make this variable a less reliable predictor for a different time period")


# In[11]:


sns.catplot(x="hr", y="cnt", data=data);
print('Now Lets look at the Effects of time on the count of total rental bikes usage.It shows that the lowest usage  is in the late night periods between midnight and 4am with lowest bieng in the interval 4-5am.The peaks are 7-8am and evening 16:00-18:00, which tends to match the office rush hours of typical city life.The fit is far from a linear and some data manipulation is required to so that we change to this to represent the usage rate and find a somewhat linear fit.Having features linearly predict the outcome reduces the need for complex non-linear models.')


# In[12]:


sns.catplot(x="mnth", y="cnt", data=data);
print("This plot is somewhat similar to the hour-count usage plot with again higher usage in the warmer months of fall and summer and lowest use in the month of January.Some datamanipulation technique could be again used to represent the data usage but however, the correlation is not as strong as the Hour versus count time graph.")


# In[13]:


sns.catplot(x="weekday", y="cnt", data=data);
print("This factor is useful in determining the usage type as it helps the model to capture the variation of rental bikes during the weekdays.The main purpose of this attribute here is to identify any class schedules had effect on the demand.")


# In[14]:


sns.catplot(x="holiday", y="cnt", data=data);
print("Holiday : whether that day is holiday or not. This feature helps us to look at the schedule booking pattern. We can see here that people tend to rent bikes more on a working day instead of holiday.This pattern we also observed that most of the rentals bieng happening between working hours of the city in the hour-cnt plot.")


# In[15]:


sns.catplot(x="workingday", y="cnt", data=data);
print("Working-day : if day is neither weekend nor holiday , value is 1. Otherwise 0\nThis shows the availability of more bikes on the days when the day is neither weekend nor holiday.")


# In[16]:


g = sns.jointplot("Week_Number", "cnt", data=data, kind="reg",color="g",height = 10)
print("The date itself provided in the format isnt something that can be used to process anykind of algorithms.From this date we can extract week number for that particular year and use that variable as predictor for the usage count.")


# In[17]:


g = sns.jointplot("casual", "cnt", data=data, kind="reg",color='g')
g = sns.jointplot("registered", "cnt", data=data, kind="reg",color='g')
print("If we see the heatmap the casual usage count is more related with continous variables. It makes sense, if we actually gave a thought about it, then registered users who uses bike to commute to work are much less likely to deterred by \nuncomfortable weather excluding the extreme conditions. Therefore the model should predict the two categories i.e. casual and registered count seperately and add up the final number to find the total count. ")


# # Pre - Processing using the Explorartory Analysis
#
# An important part of the this process is feature engineering, which involves turning the given data into some useful interpretable data or drop the redudntant feature so that model is more accurate and could be used for real time prediction.
#
# 1.Using prior knowledge, we will remove the features that dont add any important information.---Instant---
#
# 2.As we have discussed earlier, the date has been transformed into week number. For that we have first transformed     "dteday" into a formatted date and then convert it into number of weeks. -----dteday----formatted_date----
#
# 3.As we want to avoid misrepresentation of the errors in the response variable-Total number of rental bikes(CNT) and avoid the model with low accuracy we will remove casual and registered features. Though count of of casual riders has a significant change in different seasons, and has to be predicted seperately from the registered users.During on season there is a surge of bicycle users as tourism has a strong effect on the total count. But for now will remove the one feature and predict the total count.-----casual-----
#
# 4."temp" and "atemp" are strongly correalted as can be seen from the heatmap. Hence to prevent redundancy we have dropped "atemp"----atemp-----
#
# 5.Also "season" and "mnth" are strongly corelated with 0.83 value in heatmap above, so we have decided to drop season.-----season----

# # Model Representation and Prediction / Model Performance

# In[30]:


F = data.drop(['dteday','instant','formatted_date',"casual","temp","season","Week_Number"],1)
train = F.sample(frac=0.8)
test = F.drop(train.index)
train


# In[31]:


# MLE estimation of the total count of rental bikes
x_train_array = np.array(train)
x_train = np.delete(x_train_array,[11],1)
#Removing the row of target response variables("cnt")
x_t = x_train_array[:,11]
#Creating The training set into a Numpy array and extracting features and desired outputs

w_MLE = np.linalg.inv(x_train.T@x_train)@x_train.T@x_t
print(w_MLE)
#calculating optimal set of parameters(MLE)

x_test_array = np.array(test)
x_test = np.delete(x_test_array,[11],1)
y_t = x_test_array[:,11]
#Using the optimal set of parameters to test our Model

y_MLE = x_test@w_MLE
error = y_MLE - y_t
#Calculationg the error


# In[32]:


f, ax = plt.subplots(figsize=(30, 24))
ax = sns.regplot(x = y_MLE, y = y_t, color="g",ax=ax)
# Plot of MLE estimation versus True count of total bike in the testing set.


# In[33]:


#Calculating mean and median prediction error
#Calculating R squared value
aMLE = abs(error)
mean = sum(aMLE)/3476
print("Mean Prediction Error using MLE is:",mean)
median = np.median(aMLE)
print("Median Prediction Error using MLE is:",median)
SST = sum((y_t - np.mean(y_t))**2)
SSR = sum((y_MLE-np.mean(y_t))**2)
R_S = SSR/SST
print("R squared value is :",R_S)


# In[34]:


la_c = [0.01,0.1,1,10,100,110,120,130,140,150,160,200,500,1000,1500,5000]
for i in range(len(la_c)):
    w_MAP = np.linalg.inv(x_train.T@x_train + la_c[i]*np.identity(11))@x_train.T@x_t
    y_MAP = x_test@w_MAP
    error2 = y_MAP - y_t
    #calculating optimal set of parameters(MAP)
    aMAP = abs(error2)
    mean = sum(aMAP)/3476
    print("Mean Prediction Error using lambda",la_c[i]," in MAP is :",mean)


# ### As we can see the mean prediction error from different values of Lambda, we tend to choose the value with minimum error. From the iteration above we founf the value between 100 and 200 to give the minimum mean prediction error. Therefore we increase the iteration values between 100 and 200. After increasing the iteration we found the optimal value of lambda to be 140.

# In[35]:


f, ax = plt.subplots(figsize=(30, 24))
ax = sns.regplot(x = y_MAP, y = y_t, color="g",ax=ax)
# Plot of MAP estimation versus True count of total bike in the testing set.


# In[36]:


#calculating optimal set of parameters(MAP)
w_MAP = np.linalg.inv(x_train.T@x_train + 140*np.identity(11))@x_train.T@x_t
y_MAP = x_test@w_MAP
error2 = y_MAP - y_t
    #calculating optimal set of parameters(MAP)
aMAP = abs(error2)
mean = sum(aMAP)/3476
print("Mean Prediction Error using lambda 140 in MAP is :",mean)
median = np.median(aMAP)
print("Median Prediction Error using MAP is:",median)
SSTM = sum((y_t - np.mean(y_t))**2)
SSRM = sum((y_MAP - np.mean(y_t))**2)
R_SM = SSRM/SSTM
print("R squared value is :",R_SM)


# In[37]:


import scipy.stats as stats
stats.probplot(y_MLE, dist="norm", plot=plt)
plt.title("Normal Q-Q plot,y_MLE")
plt.show()
stats.probplot(y_MAP, dist="norm", plot=plt)
plt.title("Normal Q-Q plot,y_MAP")
plt.show()


# ### As we can see from the plots above that MAP and MLE gives similar plots and results regarding model performance . Also here we are using MAP without any prior distribution about the target response variable we are trying to derive.We observe that mean prediction error of both the models comes out to be around 23 and median prediction error to 16. Also the R squared values comes out to 0.94 for both the models. If we look further to investigate and look at Q-Q plots, we see that the prediction of target variable is pretty accurate in a low range data count  upto 450. After that we see a lot of outliers in our predicition towards target variable. This justifies the a difference between our mean and median values.

# ## At the end I would like to choose Maximum Likelihood Estimation for data as subjectively we dont have any information about our prior distribution and in turn the regularization would have affected our model parameters.

# In[ ]:
