# -*- coding: utf-8 -*-
"""
Created on 20211104 
@author: %(Baorui Hous
@email：houbr1992@gmail.com
"""

import os,time,joblib 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from itertools import chain 
from sklearn.svm import SVR 
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import make_scorer

def TimDat(x,t):
    #Obtain t sec x-input 
    x = np.array([ d[t] for d in x ])                  #t sec x-input 
    return x 

def ConXY(df_,t,Model_ind,TarPar):
    #convert DF to SVR x-input & target-input    
    if Model_ind == 24:
        list_model =  ['CAV','Iv2','Pd','IA','Pa','Pv','BValue','DI']   
    x = np.array([TimDat(df_[x],t) for x in list_model]) 
    x = np.transpose(x,(1,0))
    if Model_ind == 24: 
        x[:,-1] =10**(x[:,-1])                         #10**(DI) 
    x = np.log10(x) 
    y = df_[TarPar]                                    #Target Parameters
    y =np.log10(np.array(y.tolist()))                  #Target input 
    return x.astype('float32'),y.astype('float32')

def MSE(y,pred):
    #Obtained Mean Squared Error
    return np.sqrt(np.mean((y-pred)**2))

def GetBestSVRModel(x0,y0,cv):
    #Obtain Best SVR model based on the paper Cherkasssky 2004 
    d = x0.shape[1]                                    #number of x-input parameters
    n = x0.shape[0]                                    #number of stations  
    C = np.max(((np.mean(y0)-3*np.std(y0,ddof=1)),(np.mean(y1)+3*np.std(y0,ddof=1)))) #C
    p_l = (np.arange(10,60)*0.01)**(1/d)               #p list  
    gam_l = 1./p_l**2                                  #gamma list
    score_threshold = 10                               #set the intial value of score_threshold
    for i in range(len(gam_l)):
        #Obtain epsilon
        MSEScorer = make_scorer(MSE,greater_is_better=False)         #self-defined mse score 
        clf = SVR(kernel='rbf',C=C,gamma=gam_l[i],epsilon=0.)        #SVR Model epsilon = 0. 
        score = np.mean(cross_val_score(clf,x0,y0,scoring=MSEScorer,cv=cv,n_jobs=-1))  #mean mse score 
        epsilon = 3*score*(np.log(n)/n)**(0.5)                       #eplision 
        
        #Obtain a defined model 
        clf = SVR(kernel='rbf',C=C,gamma=gam_l[i],epsilon=epsilon)   #SVR Model epsilon above 
        score = np.mean(cross_val_score(clf,x0,y0,scoring=MSEScorer,cv=cv,n_jobs=-1))  #mean mse score 
        if score <= score_threshold:
            best_params = clf.get_params()                           #Obtain best parameters
            score_threshold =  score                                 #reset score value 
    return best_params 

Model_ind = 24                                         #Model index 
TraFile = './TrainDF'                                  #Train Data Name
Df0 = pd.read_pickle(TraFile)                          #Train DataFrame 
TestFile = './TestDF'                                  #Test Data Name  
Df1 = pd.read_pickle(TestFile)                         #Test DataFrame 
TarPar = 'PGV'                                         #Target Parameter Name 
cv = 10                                                #cross validation value cv 
t = 2                                                  #P-wave time window - 1 second 

x0,y0 = ConXY(Df0,t,Model_ind,TarPar)                  #Train Dataset  
x1,y1 = ConXY(Df1,t,Model_ind,TarPar)                  #Test Dataset 
best_params = GetBestSVRModel(x0,y0,cv)                #Obtain the best parameters for SVM regression  
clf = SVR(kernel='rbf',C=best_params['C'],gamma=best_params['gamma'],epsilon=best_params['epsilon'])
clf.fit(x0,y0)                                         #fitting the SVM model 

path_out = './model'                                   #model folder 
if not os.path.exists(path_out):                 
    os.makerdirs(path_out) 
model_name = '{}/SVM-{}_Model_{}_Time_{}sec.pkl'.format(path_out,TarPar,Model_ind,t+1)
joblib.dump(clf,model_name)                            #Save model  
