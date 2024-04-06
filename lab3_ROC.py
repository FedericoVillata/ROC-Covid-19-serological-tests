#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:47:09 2023

@author: augustacastelli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score

def sens_spec_thresh(test,swab):
    
    thresh = np.sort(np.array(test)) #vector of thresholds
    spec = np.zeros(len(thresh))
    sens = np.zeros(len(thresh))
    for i in range(len(thresh)):
        x0=test[swab==0] # test results for healthy patients
        x1=test[swab==1] # test results for ill patients
        Np=np.sum(swab==1) # number of ill patients
        Nn=np.sum(swab==0) # number of healthy patients
        n1=np.sum(x1>thresh[i]) # number of true positives for the given thresh 
        sens[i]=n1/Np # sensitivity
        n0=np.sum(x0<thresh[i]) # number of true negatives
        spec[i]=n0/Nn # specificity
    return sens, spec, thresh    

def roc_auc_trapezoidal(fpr, tpr):
    n = len(fpr)
    area = 0.0

    for i in range(1, n):
        width = fpr[i] - fpr[i-1]
        avg_height = (tpr[i] + tpr[i-1]) / 2
        area += width * avg_height

    return area

def find_opt_thresh(sens,spec,thresh):
    opt_thresh=0
    min_different=2
    for i in range(len(thresh)):
        difference=abs(sens[i]-spec[i])
        if difference<min_different:
            min_different=difference
            opt_thresh=thresh[i]
    return opt_thresh

                     
if __name__=='__main__':

    plt.close('all')
    xx=pd.read_csv("covid_serological_results.csv")
    xx=xx[xx.COVID_swab_res!=1]# remove rows with swab test = 1
    xx.COVID_swab_res[xx.COVID_swab_res==2]=1# replace swab test = 2 with swab test = 1
    swab=xx.COVID_swab_res.values
    Test1=xx.IgG_Test1_titre.values  #from 2.5 to 314
    Test2=xx.IgG_Test2_titre.values  #from 0 to 9.71
    
    #Perform the usual data analysis
    
    # Display summary statistics
    xx.describe()
    
    # Create a scatter matrix
    pd.plotting.scatter_matrix(xx[['COVID_swab_res', 'IgG_Test1_titre', 'IgG_Test2_titre']], figsize=(10, 10))
    plt.suptitle('Scatter Matrix of COVID Swab Results and IgG Levels')
    plt.show()
    
    #Test2
    
    x2=Test2
    y=swab
    
    sens2, spec2, thresh2 = sens_spec_thresh(x2,y)
    FPR2=1-sens2
    FNR2=1-spec2
    
    #Test1
    
    x1=Test1
    
    sens1, spec1, thresh1 = sens_spec_thresh(x1,y)
    FPR1=1-sens1
    FNR1=1-spec1
    
    #Plot sens e spec vs. thresholds
    
    #test2
    plt.figure()
    plt.grid()
    plt.title('Test2')
    plt.plot(thresh2, sens2, label='sens')
    plt.plot(thresh2, FNR2, label='FNR')
    plt.legend()
    plt.xlabel('Threshold')  # Aggiungi questa linea per etichettare l'asse x
    plt.show()
    
    #test1
    plt.figure()
    plt.grid()
    plt.title('Test1')
    plt.plot(thresh1, sens1, label='sens')
    plt.plot(thresh1, FNR1, label='FNR')
    plt.legend()
    plt.xlabel('Threshold')  # Aggiungi questa linea per etichettare l'asse x
    plt.show()
    
    #Plot of ROC curve TPR vs. FPR
    
    plt.figure()
    plt.grid()
    plt.plot(FNR2, sens2)
    plt.title('Test2-ROC curve')
    plt.xlabel('FPR') 
    plt.ylabel('TPR') 
    plt.show()
    
    plt.figure()
    plt.grid()
    plt.plot(FNR1, sens1)
    plt.title('Test1-ROC curve')
    plt.xlabel('FPR') 
    plt.ylabel('TPR') 
    plt.show()
    
    # Find Area Under the (ROC) Curve Test2
    
    #trapezoidal method
    AUC_trapezoid_Test2 = roc_auc_trapezoidal(FPR2, spec2)
    print(f"Test2 : The AUC of the ROC computed with trapezoidal method is: {AUC_trapezoid_Test2}")
    
    #skl method
    #fpr2_skl, tpr2_skl, thresh2_skl=metrics.roc_curve(y,x2,pos_label=1)
    AUC_Test2 = roc_auc_score(y,x2)
    print(f"Test2 : The AUC of the ROC computed with skl is: {AUC_Test2}\n")
    
    
    # Find Area Under the (ROC) Curve Test2
    
    #trapezoidal method
    AUC_trapezoid_Test1 = roc_auc_trapezoidal(FPR1, spec1)
    print(f"Test1 : The AUC of the ROC computed with trapezoidal method is: {AUC_trapezoid_Test1}")
    
    #skl method
    
    #fpr1_skl, tpr1_skl, thresh1_skl=metrics.roc_curve(y,x1,pos_label=1)
    AUC_Test1 = roc_auc_score(y,x1)
    print(f"Test1 : The AUC of the ROC computed with skl is: {AUC_Test1}")
    
    #set the thresholds
    
    opt_thresh2=find_opt_thresh(sens2,spec2,thresh2)
    print(f"The optimal threshold for test2 is : {opt_thresh2}\n")
    
    opt_thresh1=find_opt_thresh(sens1,spec1,thresh1)
    print(f"The optimal threshold for test1 is : {opt_thresh1}\n")
    
    
    
    
    
