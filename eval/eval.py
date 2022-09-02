# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 08:37:11 2021

@author: Narmin Ghaffari Laleh
"""

##############################################################################

import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import numpy as np
import utils.utils as utils
from shutil import copyfile
import torch
from tqdm import tqdm
from utils.data_utils import DatasetLoader
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

##############################################################################

def CalculatePatientWiseAUC(resultCSVPath, args, reportFile, eps = None, epoch = None, val = True, micro = True):
    
    data = pd.read_csv(resultCSVPath)
    patients = list(set(data['PATIENT']))
    keys = list(args.target_labelDict.keys())
    yProbDict = {}    
    for index, key in enumerate(keys):
        patientsList = []
        yTrueList = []
        yTrueLabelList = []
        yProbList = []         
        keys_temp = keys.copy()
        keys_temp.remove(key)
        for patient in patients:
            patientsList.append(patient)
            data_temp = data.loc[data['PATIENT'] == patient]                        
            data_temp = data_temp.reset_index()            
            yTrueList.append(data_temp['yTrue'][0])
            yTrueLabelList.append(utils.get_key_from_value(args.target_labelDict, data_temp['yTrue'][0]))                        
            
            dl_pred = np.where(data_temp[keys_temp].lt(data_temp[key], axis=0).all(axis=1), True, False)
            dl_pred = list(dl_pred)
            true_count = dl_pred.count(True)            
            yProbList.append(true_count / len(dl_pred)) 
                   
        fpr, tpr, thresholds = metrics.roc_curve(yTrueList, yProbList, pos_label = args.target_labelDict[key])
                
        name = 'TEST_RESULT_PATIENT_BASED_EPSILON_'
        path = os.path.join(args.result_dir, name + str(eps) + '.csv')
        #print('\nAUC FOR TARGET {} IN THIS DATA SET WITH EPS {} IS: {} '.format(key, eps, np.round(metrics.auc(fpr, tpr), 3)))
        #reportFile.write('AUC FOR TARGET {} IN THIS DATA SET WITH EPS {} IS: {} '.format(key, eps, np.round(metrics.auc(fpr, tpr), 3)) + '\n')            
        yProbDict[key] = yProbList
            
    lb = LabelBinarizer()  
    y_true = yTrueList
    lb.fit(y_true)
    y = lb.transform(y_true)
    y_score = np.array((yProbDict['ccRCC'], yProbDict['chRCC'], yProbDict['papRCC'])).T
    #y_score = np.array((yProbDict['diffuse'], yProbDict['intestinal'], yProbDict['mixed'])).T
    fpr, tpr, _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc_micro = auc(fpr, tpr)      
    
    print('\nMicro AUC IN THIS DATA SET WITH EPS {} IS: {} '.format(eps, np.round(roc_auc_micro, 3)))
    reportFile.write('\nMicro AUC IN THIS DATA SET WITH EPS {} IS: {}'.format(eps, np.round(roc_auc_micro, 3)) + '\n')                    
    
    yProbDict = pd.DataFrame.from_dict(yProbDict)
    df = pd.DataFrame(list(zip(patientsList, yTrueList, yTrueLabelList)), columns =['PATIENT', 'yTrue', 'yTrueLabel'])
    df = pd.concat([df, yProbDict], axis=1)    
    df.to_csv(path, index = False)
    return path, np.round(metrics.auc(fpr, tpr), 3)
                        
##############################################################################

def PlotTrainingLossAcc(train_loss_history, train_acc_history):
    
    plt.figure()
    plt.plot(range(len(train_loss_history)), train_loss_history)
    plt.xlabel('Epochs', fontsize = 30)  
    plt.xlabel('Train_Loss', fontsize = 30)

    plt.figure()
    plt.plot(range(len(train_acc_history)), train_acc_history)
    plt.xlabel('Epochs', fontsize = 30)  
    plt.xlabel('Train_Accuracy', fontsize = 30)

##############################################################################

def PlotBoxPlot(y_true, y_pred):
    fig, ax = plt.subplots()
    sns.boxplot(x = y_true, y = y_pred)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.xlabel('CLASSES', fontsize = 30)  
    plt.ylabel('SCORES', fontsize = 30)  
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=30)

##############################################################################

def PlotROCCurve(y_true, y_pred):

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    fig, ax = plt.subplots()
    plt.title('Receiver Operating Characteristic', fontsize = 30)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate', fontsize = 30)
    plt.xlabel('False Positive Rate', fontsize = 30)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=30)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

##############################################################################

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

##############################################################################

def CalculateTotalROC(resultsPath, results, target_labelDict, reportFile):
    
    totalData = []    
    for item in results:
        data = pd.read_csv(os.path.join(resultsPath, item))
        totalData.append(data)        
    totalData = pd.concat(totalData)
    y_true = list(totalData['yTrue'])
    keys = list(target_labelDict.keys())
    
    for key in keys:
        y_pred = totalData[key]
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label = target_labelDict[key])
        print('-' * 30)        
        print('TOTAL AUC FOR target {} IN THIS DATASET IS : {} '.format(key, np.round(metrics.auc(fpr, tpr), 3)))
        reportFile.write('-' * 30 + '\n')
        reportFile.write('TOTAL AUC FOR target {} IN THIS DATASET IS : {} '.format(key, np.round(metrics.auc(fpr, tpr), 3)) + '\n')
        auc_values = []
        nsamples = 1000
        rng = np.random.RandomState(666)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        for i in range(nsamples):
            indices = rng.randint(0, len(y_pred), len(y_pred))
            if len(np.unique(y_pred[indices])) < 2 or np.sum(y_true[indices]) == 0:
                continue    
            fpr, tpr, thresholds = metrics.roc_curve(y_true[indices], y_pred[indices], pos_label = target_labelDict[key])
            auc_values.append(metrics.auc(fpr, tpr))
        
        auc_values = np.array(auc_values)
        auc_values.sort()
        reportFile.write('Lower Confidebnce Interval For Target {}: {}'.format(key, np.round(auc_values[int(0.025 * len(auc_values))], 3)) + '\n')
        reportFile.write('Higher Confidebnce Interval For Target {} : {}'.format(key, np.round(auc_values[int(0.975 * len(auc_values))], 3)) + '\n')     
        print('Lower Confidebnce Interval For Target {}: {}'.format(key, np.round(auc_values[int(0.025 * len(auc_values))], 3)))        
        print('Higher Confidebnce Interval For Target {} : {}'.format(key, np.round(auc_values[int(0.975 * len(auc_values))], 3)))
        
    totalData.to_csv(os.path.join(resultsPath, 'TEST_RESULT_PATIENT_BASED_TOTAL.csv'), index = False)    

##############################################################################

def find_closes(array, value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return array[idx]

##############################################################################

def MergeResultCSV(resultsPath, results, milClam = False):
    
    totalData = []    
    for item in results:
        data = pd.read_csv(os.path.join(resultsPath, item))
        totalData.append(data)
    totalData = pd.concat(totalData)
    if milClam:
        totalData.to_csv(os.path.join(resultsPath, 'TEST_RESULT_SLIDE_BASED_TOTAL.csv'), index = False)
    else:            
        totalData.to_csv(os.path.join(resultsPath, 'TEST_RESULT_TILE_BASED_TOTAL.csv'), index = False)

##############################################################################

def GenerateHighScoreTiles(totalPatientResultPath, totalResultPath, numHighScorePetients, numHighScoreTiles, 
                           target_labelDict, savePath, attack = None, eps = 0, bns = False, input_size = None):
                       
    patientData = pd.read_csv(totalPatientResultPath)
    tileData = pd.read_csv(totalResultPath)
    
    keys = list(target_labelDict.keys())
    
    for key in keys:
        dataTemp = patientData.loc[patientData['yTrueLabel'] == key]
        dataTemp = dataTemp.sort_values(by = [key], ascending = False)
        
        highScorePosPatients = list(dataTemp['PATIENT'][0 : numHighScorePetients])
                 
        fig = plt.figure(figsize=(10,10))
        i = 1
        
        path = os.path.join(savePath, key)
        os.makedirs(path, exist_ok = True)
        
        for index, patient in enumerate(highScorePosPatients):            
            dataTemp = tileData.loc[tileData['PATIENT'] == patient]
            dataTemp = dataTemp.sort_values(by = [key], ascending = False)
            highScorePosTiles = list(dataTemp['TilePath'][0:numHighScoreTiles])                        
            for tile in highScorePosTiles:  
                
                params = {'batch_size': 1,
                          'shuffle': False,
                          'num_workers': 0}
                test_set = DatasetLoader([tile], [target_labelDict[key]], transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
                testGenerator = torch.utils.data.DataLoader(test_set, **params)
        
                img = Image.open(tile)
                if not attack == None:
                    with torch.enable_grad():
                        for inputs, labels in testGenerator:
                            inputs = inputs.to(device)
                            labels = labels.to(device)
                            img = attack(inputs, labels)
                            # img = attack.perturb(original_images = inputs, labels = labels, 
                            #                     reduction4loss = 'mean', random_start = False, bns = bns, 
                            #                     exclusive = False)
                        temp = img.cpu().detach().numpy()
                        data_img = (temp.squeeze()*255).astype(np.uint8)  
                        data_img = np.moveaxis(data_img, -1, 0)
                        data_img = np.moveaxis(data_img, -1, -2)
                        data_img = np.moveaxis(data_img, -2, -3)
                        img = Image.fromarray(data_img, mode='RGB')
                        img.save(os.path.join(path, tile.split('\\')[-1]))
                else:
                    for inputs, labels in testGenerator:
                        inputs = inputs.to(device)
                        temp = inputs.cpu().detach().numpy()
                        data_img = (temp.squeeze()*255).astype(np.uint8)  
                        data_img = np.moveaxis(data_img, -1, 0)
                        data_img = np.moveaxis(data_img, -1, -2)
                        data_img = np.moveaxis(data_img, -2, -3)
                        img = Image.fromarray(data_img, mode='RGB')
                        img.save(os.path.join(path, tile.split('\\')[-1]))
                    #copyfile(tile, os.path.join(path, tile.split('\\')[-1]))        
                ax = plt.subplot(numHighScorePetients, numHighScoreTiles, i)
                ax.set_axis_off()
                plt.imshow(img)
                i += 1 
            
        plt.savefig(os.path.join(path,  key + '.png'))
        plt.close()

##############################################################################






































