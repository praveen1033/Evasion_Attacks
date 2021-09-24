# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 15:32:30 2020

@author: vmcx3
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 01:09:49 2020

@author: vmcx3
"""

import csv
import pandas as pd
import numpy as np
from scipy.stats import hmean
from scipy.special import boxcox, inv_boxcox
from scipy import stats
import random
from sklearn.cluster import KMeans
from pandas.plotting import scatter_matrix
from scipy.stats import norm
from matplotlib import pyplot as plt
import math
#import tensorflow as tf
import seaborn as sns
sns.set()
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def count_values_in_range(series, range_min, range_max):

    # "between" returns a boolean Series equivalent to left <= series <= right.
    # NA values will be treated as False.
    return series.between(left=range_min, right=range_max).sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x):
    return x * (1 - x)



train_df = pd.read_csv("C:\\Users\\vmcx3\\Desktop\\Data sets\\Electricity data\\TOSG_cont\\2015.csv", header = 0)
train_df.use = train_df.use * 1000

"""
for i in range(train_df.use.size):
    if train_df.iloc[i,2] <= 0:
        train_df.iloc[i,2] = 400
"""

more =  train_df[train_df['use'] > 100]
less =  train_df[train_df['use'] <= 100]
less['use'] = 400
train_df3 = more.append(less)

more =  train_df3[train_df3['use'] > 1600]
less =  train_df3[train_df3['use'] <= 1600]
more['use'] = 1600
train_df2 = more.append(less)

#train_df2['use'] = train_df2['use'].fillna(np.mean(train_df2.use))
       
train_df1 = train_df2.copy()


train_df2.use = stats.boxcox(train_df2.use, lmbda=0.25)

final_df = train_df2.use.groupby(train_df.localminute).unique().apply(pd.Series)



hm_avg = []
am_avg = []
sd_avg = []
Q_ratio = [] 

for i in range(60):
    hm=[]
    am=[]
    sd=[]
    for j in range(24):
        hm.append(hmean(final_df.iloc[i*24+j,:].dropna()))
        am.append(np.mean(final_df.iloc[i*24+j,:].dropna()))
        sd.append(np.std(final_df.iloc[i*24+j,:].dropna()))
    hm_avg.append(np.mean(hm))
    am_avg.append(np.mean(am))
    sd_avg.append(np.mean(sd))
    Q_ratio.append(hm_avg[i]/am_avg[i])


##########################   injecting false data    #############
deltaAvg = 500
false_df = train_df1.copy()

rand_50 = pd.DataFrame(np.random.randint(0,50,size=(len(false_df), 1)), columns=['rand'])
 
false_df['use'] = false_df['use'] + deltaAvg + rand_50['rand']

'''
false_df = train_df1.copy()

rand_500 = pd.DataFrame(np.random.randint(0,500,size=(len(false_df), 1)), columns=['rand'])

'''


'''
more =  train_df[false_df['use'] > 0]
less =  train_df[false_df['use'] <= 0]
less['use'] = 100
false_df2 = more.append(less)
'''


false_df.use = stats.boxcox(false_df.use, lmbda=0.25)

false_final_df = false_df.use.groupby(train_df.localminute).unique().apply(pd.Series)
    

mean_correction=[]
for i in range(30):
    mean_correction.append(hm_avg[i] - (am_avg[i] - hm_avg[i]))
    
meterwise_NonBoxcox_original =  train_df.use.groupby(train_df.dataid).unique().apply(pd.Series)
meterwise_NonBoxcox = train_df.use.groupby(train_df.dataid).unique().apply(pd.Series)


meterwise_original = train_df2.use.groupby(train_df.dataid).unique().apply(pd.Series)

meterwise_falsified = false_df.use.groupby(train_df.dataid).unique().apply(pd.Series)



for i in range(150,219):
    for j in range(1440):
        if np.mod(j,2) == 0:
            #if meterwise_original.iloc[i,j] < am_avg[math.floor(j/24)]:
            meterwise_falsified.iloc[i,j] = am_avg[math.floor(j/24)] + sd_avg[math.floor(j/24)] - 0.01
            

        else:
            temp1 = inv_boxcox(meterwise_falsified.iloc[i,j-1], 0.25) - inv_boxcox(meterwise_original.iloc[i,j-1], 0.25)     #false data margin by even point
            temp2 = 2*deltaAvg - temp1                                                                                       #compensating deltaAvg margin
            val = inv_boxcox(meterwise_original.iloc[i,j], 0.25) + temp2                                                                      
            meterwise_NonBoxcox.iloc[i,j] = val
            meterwise_NonBoxcox.iloc[i,j-1] = temp1
            meterwise_falsified.iloc[i,j] =  boxcox(val, 0.25)                                                                                                                                                 





frames = [meterwise_original[:150][:], meterwise_falsified[150:][:]]
meterwise = pd.concat(frames)

temp = np.zeros(shape=(219,720))
theta = pd.DataFrame(temp)
l_original = pd.DataFrame(temp)
x = l_original.copy()
cw = l_original.copy()
w = l_original.copy()

for i in range(219):
    for j in range(720):
        theta.iloc[i,j] = abs(meterwise.iloc[i,j] - am_avg[math.floor(j/24)])
        if theta.iloc[i,j]<sd_avg[math.floor(j/24)]:
            l_original.iloc[i,j]=4
        elif theta.iloc[i,j]<2*sd_avg[math.floor(j/24)]:
            l_original.iloc[i,j]=3
        elif theta.iloc[i,j]<3*sd_avg[math.floor(j/24)]:
            l_original.iloc[i,j]=2
        else:
            l_original.iloc[i,j]=1
    

l1_original = l_original.apply(np.sort, axis = 1)
 
K=4
 
for i in range(219):
    for j in range(720):
        x.iloc[i,j] = 1 + ((K-1)*j)/720
        
temp1 = np.zeros(shape=219)
std_dr = pd.DataFrame(temp1)
   
for i in range(219):
    std_dr.iloc[i] = np.std(l1_original[i])    
    
for i in range(len(std_dr)):
    if int(std_dr.iloc[i]) == 0:
       std_dr.iloc[i] = np.mean(std_dr) 

M_BR = 4

for i in range(219):
    for j in range(720):
        cw.iloc[i,j] = (1/(math.sqrt(2*3.1415)*std_dr.iloc[i,0]))*(math.exp((-1*math.pow((x.iloc[i,j]-M_BR),2))/(2*math.pow(std_dr.iloc[i,0],2))))
        

for i in range(219):
    for j in range(720):
        w.iloc[i,j] = cw.iloc[i,j]/np.sum(cw.iloc[i,:])
        
eeta = 2;
R = np.zeros(shape=219)


for meter in range(219):
    temp2 = np.zeros(shape=(4,720))
    I = pd.DataFrame(temp2)
    

    for j in range(720):
        if l_original.iloc[meter,j] == 1:
            I.iloc[0,j] = 1
        elif l_original.iloc[meter,j] == 2:
            I.iloc[1,j] = 1
        elif l_original.iloc[meter,j] == 3:
            I.iloc[2,j] = 1
        else:
            I.iloc[3,j] = 1
            
    temp3 = np.zeros(shape=4)
    wd = pd.DataFrame(temp3)
    
    for i in range(4):
        for j in range(720):
            wd.iloc[i,0] = wd.iloc[i,0] + I.iloc[i,j]*w.iloc[meter,j]
            
    for i in range(4):
        R[meter] = R[meter] + i*wd.iloc[i,0]

            


TR = np.zeros(shape=219)

for meter in range(219):
    TR[meter] = (1/math.pow(K,eeta))*(math.pow(R[meter],eeta))

'''
kmeans = KMeans(n_clusters=2)
kmeans.fit(TR.reshape(-1,1))
y_kmeans = kmeans.predict(TR.reshape(-1,1))
md = np.sum(y_kmeans[150:]==1)
'''
FA = 0
MD = 0
for i in range(219):
    if i<150:
        if TR[i] < 0.34:
            FA = FA + 1
    else:
        if TR[i] > 0.34:
            MD = MD + 1

FA_percent = FA/150
MD_percent = MD/69
print(MD, FA, MD_percent, FA_percent)

xax = np.zeros(shape=219)

for i in range(219):
    xax[i] = i

plt.hold(True)
for i in range(219):
    if i<150:
        plt.plot(xax[i],TR[i],'b*')
    else:
        plt.plot(xax[i],TR[i],'ro')

plt.title('Attack without Evasion')
plt.xlabel('Smart meter ID')
plt.ylabel('Trust Score')


###################  DISCRIMINATOR


#############################   True Data Samples   ########################

train_setA = meterwise_NonBoxcox_original.iloc[0:150, 0:720]

count_df = pd.DataFrame()

for i in range(30):        
    count_df[i] = train_setA.apply(
            func=lambda row: count_values_in_range(row, (i)*100 + 1, (i+1) * 100), axis=1)


prob_df = (count_df + 1) / 750


#############################   Generating Fake Samples    ########################

train_setB = meterwise_NonBoxcox.iloc[150:, 0:720]

fake_count_df = pd.DataFrame()

for i in range(30):        
    fake_count_df[i] = train_setB.apply(
            func=lambda row: count_values_in_range(row, (i)*100 + 1, (i+1) * 100), axis=1)


fake_prob_df = (fake_count_df + 1) / 750



#############################   Neural Network    ########################


# input dataset
training_inputs = np.concatenate((np.array(prob_df.values), np.array(fake_prob_df.values)))

# output dataset
training_outputs = np.concatenate([np.zeros(150), np.ones(69)]).reshape(1,219).T

# seed random numbers to make calculation
np.random.seed(1)

# initialize weights randomly with mean 0 to create weight matrix, synaptic weights
synaptic_weights = 2 * np.random.random((30,1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

# Iterate 10,000 times
for iteration in range(10000):

    # Define input layer
    input_layer = training_inputs
    # Normalize the product of the input layer with the synaptic weights
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    # how much did we miss?
    error = training_outputs - outputs 

    # multiply how much we missed by the
    # slope of the sigmoid at the values in outputs
    adjustments = error * sigmoid_derivative(outputs) 

    # update weights
    synaptic_weights += np.dot(input_layer.T, adjustments)

print('Synaptic weights after training: ')
print(synaptic_weights)

print("Output After Training:")
print(outputs)

#############################   test Data Samples   ########################
deltaAvg = 400
false_df = train_df1.copy()

rand_50 = pd.DataFrame(np.random.randint(0,50,size=(len(false_df), 1)), columns=['rand'])
 
false_df['use'] = false_df['use'] + deltaAvg + rand_50['rand']

meterwise_NonBoxcox_original =  train_df.use.groupby(train_df.dataid).unique().apply(pd.Series)
meterwise_NonBoxcox = false_df.use.groupby(train_df.dataid).unique().apply(pd.Series)


meterwise_original = train_df2.use.groupby(train_df.dataid).unique().apply(pd.Series)

meterwise_falsified = false_df.use.groupby(train_df.dataid).unique().apply(pd.Series)



train_setA = meterwise_NonBoxcox_original.iloc[0:150, 720:1440]

count_df = pd.DataFrame()

for i in range(30):        
    count_df[i] = train_setA.apply(
            func=lambda row: count_values_in_range(row, (i)*100 + 1, (i+1) * 100), axis=1)


prob_df = (count_df + 1) / 750

train_setB = meterwise_NonBoxcox.iloc[150:, 720:1440]

fake_count_df = pd.DataFrame()

for i in range(30):        
    fake_count_df[i] = train_setB.apply(
            func=lambda row: count_values_in_range(row, (i)*100 + 1, (i+1) * 100), axis=1)


fake_prob_df = (fake_count_df + 1) / 750



# input dataset
test_inputs = np.concatenate((np.array(prob_df.values), np.array(fake_prob_df.values)))

final_outputs = sigmoid(np.dot(test_inputs, synaptic_weights))



for i in range(219):
    if i<150:
        plt.plot(xax[i],final_outputs[i],'b*')
    else:
        plt.plot(xax[i],final_outputs[i],'ro')





'''

#######################################    second
train_df = pd.read_csv("C:\\Users\\vmcx3\\Desktop\\Data set\\Electricity data\\TOSG_cont\\2015.csv", header = 0)
train_df.use = train_df.use * 1000

"""
for i in range(train_df.use.size):
    if train_df.iloc[i,2] <= 0:
        train_df.iloc[i,2] = 400
"""

more =  train_df[train_df['use'] > 100]
less =  train_df[train_df['use'] <= 100]
less['use'] = 400
train_df3 = more.append(less)

more =  train_df3[train_df3['use'] > 1600]
less =  train_df3[train_df3['use'] <= 1600]
more['use'] = 1600
train_df2 = more.append(less)

#train_df2['use'] = train_df2['use'].fillna(np.mean(train_df2.use))
       
train_df1 = train_df2.copy()


train_df2.use = stats.boxcox(train_df2.use, lmbda=0.25)

final_df = train_df2.use.groupby(train_df.localminute).unique().apply(pd.Series)



hm_avg = []
am_avg = []
sd_avg = []
Q_ratio = [] 

for i in range(30):
    hm=[]
    am=[]
    sd=[]
    for j in range(24):
        hm.append(hmean(final_df.iloc[i*30+j,:].dropna()))
        am.append(np.mean(final_df.iloc[i*30+j,:].dropna()))
        sd.append(np.std(final_df.iloc[i*30+j,:].dropna()))
    hm_avg.append(np.mean(hm))
    am_avg.append(np.mean(am))
    sd_avg.append(np.mean(sd))
    Q_ratio.append(hm_avg[i]/am_avg[i])


##########################   injecting false data    #############
#deltaAvg = 500
false_df = train_df1.copy()

rand_50 = pd.DataFrame(np.random.randint(0,50,size=(len(false_df), 1)), columns=['rand'])
 
false_df['use'] = false_df['use'] + deltaAvg + rand_50['rand']

'''
false_df = train_df1.copy()

rand_500 = pd.DataFrame(np.random.randint(0,500,size=(len(false_df), 1)), columns=['rand'])

'''


'''
more =  train_df[false_df['use'] > 0]
less =  train_df[false_df['use'] <= 0]
less['use'] = 100
false_df2 = more.append(less)
'''


false_df.use = stats.boxcox(false_df.use, lmbda=0.25)

false_final_df = false_df.use.groupby(train_df.localminute).unique().apply(pd.Series)
    

mean_correction=[]
for i in range(30):
    mean_correction.append(hm_avg[i] - (am_avg[i] - hm_avg[i]))


meterwise_original = train_df2.use.groupby(train_df.dataid).unique().apply(pd.Series)

meterwise_falsified = false_df.use.groupby(train_df.dataid).unique().apply(pd.Series)



#Generator logic ------- remodifuing the required values (first 720) of trust score based on generator logic.


for i in range(219):
    for j in range(720):
        if np.mod(j,2) == 0:
            #if meterwise_original.iloc[i,j] < am_avg[math.floor(j/24)]:
            meterwise_falsified.iloc[i,j] = am_avg[math.floor(j/24)] + sd_avg[math.floor(j/24)] - 0.01
        else:
            temp1 = inv_boxcox(meterwise_falsified.iloc[i,j-1], 0.25) - inv_boxcox(meterwise_original.iloc[i,j-1], 0.25)     #false data margin by even point
            temp2 = 2*deltaAvg - temp1                                                                                       #compensating deltaAvg margin
            val = inv_boxcox(meterwise_original.iloc[i,j], 0.25) + temp2                                                                       #finding the falsified data
            meterwise_falsified.iloc[i,j] =  boxcox(val, 0.25)                                                                                                                                                  





frames = [meterwise_original[:150][:], meterwise_falsified[150:][:]]
meterwise = pd.concat(frames)

temp = np.zeros(shape=(219,720))
theta = pd.DataFrame(temp)
l_original = pd.DataFrame(temp)
x = l_original.copy()
cw = l_original.copy()
w = l_original.copy()

for i in range(219):
    for j in range(720):
        theta.iloc[i,j] = abs(meterwise.iloc[i,j] - am_avg[math.floor(j/24)])
        if theta.iloc[i,j]<sd_avg[math.floor(j/24)]:
            l_original.iloc[i,j]=4
        elif theta.iloc[i,j]<2*sd_avg[math.floor(j/24)]:
            l_original.iloc[i,j]=3
        elif theta.iloc[i,j]<3*sd_avg[math.floor(j/24)]:
            l_original.iloc[i,j]=2
        else:
            l_original.iloc[i,j]=1
    

l1_original = l_original.apply(np.sort, axis = 1)
 
K=4
 
for i in range(219):
    for j in range(720):
        x.iloc[i,j] = 1 + ((K-1)*j)/720
        
temp1 = np.zeros(shape=219)
std_dr = pd.DataFrame(temp1)
   
for i in range(219):
    std_dr.iloc[i] = np.std(l1_original[i])  
    
for i in range(len(std_dr)):
    if int(std_dr.iloc[i]) == 0:
       std_dr.iloc[i] = np.mean(std_dr) 

M_BR = 4

for i in range(219):
    for j in range(720):
        cw.iloc[i,j] = (1/(math.sqrt(2*3.1415)*std_dr.iloc[i,0]))*(math.exp((-1*math.pow((x.iloc[i,j]-M_BR),2))/(2*math.pow(std_dr.iloc[i,0],2))))
        

for i in range(219):
    for j in range(720):
        w.iloc[i,j] = cw.iloc[i,j]/np.sum(cw.iloc[i,:])
        
eeta = 2;
R1 = np.zeros(shape=219)



for meter in range(219):
    temp2 = np.zeros(shape=(4,720))
    I = pd.DataFrame(temp2)
    

    for j in range(720):
        if l_original.iloc[meter,j] == 1:
            I.iloc[0,j] = 1
        elif l_original.iloc[meter,j] == 2:
            I.iloc[1,j] = 1
        elif l_original.iloc[meter,j] == 3:
            I.iloc[2,j] = 1
        else:
            I.iloc[3,j] = 1
            
    temp3 = np.zeros(shape=4)
    wd = pd.DataFrame(temp3)
    
    for i in range(4):
        for j in range(720):
            wd.iloc[i,0] = wd.iloc[i,0] + I.iloc[i,j]*w.iloc[meter,j]
            
    for i in range(4):
        R1[meter] = R1[meter] + i*wd.iloc[i,0]

            


TR1 = np.zeros(shape=219)

for meter in range(219):
    TR1[meter] = (1/math.pow(K,eeta))*(math.pow(R[meter],eeta))


kmeans1 = KMeans(n_clusters=2)
kmeans1.fit(TR1.reshape(-1,1))
y_kmeans1 = kmeans1.predict(TR1.reshape(-1,1))
md1 = np.sum(y_kmeans1[150:]==1)

plt.hold(True)
for i in range(219):
    if i<150:
        plt.plot(xax[i],TR1[i],'b*')
    else:
        plt.plot(xax[i],TR1[i],'ro')

plt.title('Attack with Evasion')
plt.xlabel('Smart meter ID')
plt.ylabel('Trust Score')

print(md1/69*100 - md/69*100)


'''



'''
false_df.use = stats.boxcox(false_df.use, lmbda=0.25)

false_final_df = false_df.use.groupby(train_df.localminute).unique().apply(pd.Series)



hm_avg = []
am_avg = []
sd_avg = []
Q_ratio = [] 

for i in range(30):
    hm=[]
    am=[]
    sd=[]
    for j in range(24):
        hm.append(hmean(false_final_df.iloc[i*30+j,1:180]))
        am.append(np.mean(false_final_df.iloc[i*30+j,1:180]))
        sd.append(np.std(false_final_df.iloc[i*30+j,1:180]))
    hm_avg.append(np.mean(hm))
    am_avg.append(np.mean(am))
    sd_avg.append(np.mean(sd))
    Q_ratio.append(hm_avg[i]/am_avg[i])






mean_correction=[]
for i in range(30):
    mean_correction.append(hm_avg[i] - (am_avg[i] - hm_avg[i]))


meterwise = false_df.use.groupby(train_df.dataid).unique().apply(pd.Series)

temp = np.zeros(shape=(219,720))
theta = pd.DataFrame(temp)
l = pd.DataFrame(temp)
x = l.copy()
cw = l.copy()
w = l.copy()

for i in range(219):
    for j in range(720):
        theta.iloc[i,j] = abs(meterwise.iloc[i,j] - am_avg[math.floor(j/30)])
        if theta.iloc[i,j]<sd_avg[math.floor(j/30)]:
            l.iloc[i,j]=4
        elif theta.iloc[i,j]<2*sd_avg[math.floor(j/30)]:
            l.iloc[i,j]=3
        elif theta.iloc[i,j]<3*sd_avg[math.floor(j/30)]:
            l.iloc[i,j]=2
        else:
            l.iloc[i,j]=1
    

l1 = l.apply(np.sort, axis = 1)
 
K=4
 
for i in range(219):
    for j in range(720):
        x.iloc[i,j] = 1 + ((K-1)*j)/720
        
temp1 = np.zeros(shape=219)
std_dr = pd.DataFrame(temp1)
   
for i in range(219):
    std_dr.iloc[i] = np.std(l1[i])    

M_BR = 4

for i in range(219):
    for j in range(720):
        cw.iloc[i,j] = (1/(math.sqrt(2*3.1415)*std_dr.iloc[i,0]))*(math.exp((-1*math.pow((x.iloc[i,j]-M_BR),2))/(2*math.pow(std_dr.iloc[i,0],2))))
        

for i in range(219):
    for j in range(720):
        w.iloc[i,j] = cw.iloc[i,j]/np.sum(cw.iloc[i,:])
        
eeta = 2;
tempx = np.zeros(shape=219)
R = pd.DataFrame(tempx)


for meter in range(219):
    temp2 = np.zeros(shape=(4,720))
    I = pd.DataFrame(temp2)
    

    for j in range(720):
        if l.iloc[meter,j] == 1:
            I.iloc[0,j] = 1
        elif l.iloc[meter,j] == 2:
            I.iloc[1,j] = 1
        elif l.iloc[meter,j] == 3:
            I.iloc[2,j] = 1
        else:
            I.iloc[3,j] = 1
            
    temp3 = np.zeros(shape=4)
    wd = pd.DataFrame(temp3)
    
    for i in range(4):
        for j in range(720):
            wd.iloc[i,0] = wd.iloc[i,0] + I.iloc[i,j]*w.iloc[meter,j]
            
    for i in range(4):
        R.iloc[meter,0] = R.iloc[meter,0] + i*wd.iloc[i,0]

'''



