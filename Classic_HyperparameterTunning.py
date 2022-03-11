# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 08:48:16 2021

@author: nghaffarilal
"""

import utils.utils as utils
import argparse
import warnings
import torch
from utils.data_utils import ConcatCohorts_Classic, DatasetLoader, GetTiles
from utils.core_utils import Train_model
import torch.nn as nn
import torchvision
import pandas as pd
import os
import random
from sklearn import preprocessing
from attack import FastGradientSignUntargeted
import numpy as np
from torch_lr_finder import LRFinder

###############################################################################

parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
parser.add_argument('--adressExp', type = str, default = r"J:\Adversarial Project\TCGA_RN50_HyperParameterTunning.txt", help = 'Adress to the experiment File')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
print('\nTORCH Detected: {}\n'.format(device))
args = utils.ReadExperimentFile(args) 
torch.cuda.set_device(args.gpuNo)

freezRatios = [0.25, 0.5, 0.75]

additionalData = list(pd.read_csv(r"U:\WholeData.csv")['0'])
targetLabels = args.target_labels
for targetLabel in targetLabels:
    args.target_label = targetLabel        
    random.seed(args.seed)
    args.projectFolder = utils.CreateProjectFolder(ExName = args.project_name, ExAdr = args.adressExp, targetLabel = targetLabel,
                                                     model_name = args.model_name)
    print('-' * 30 + '\n')
    print(args.projectFolder)
    if os.path.exists(args.projectFolder):
        continue
    else:
        os.mkdir(args.projectFolder) 

    args.result_dir = os.path.join(args.projectFolder, 'RESULTS')
    os.makedirs(args.result_dir, exist_ok = True)
    args.split_dir = os.path.join(args.projectFolder, 'SPLITS')
    os.makedirs(args.split_dir, exist_ok = True)
       
    reportFile  = open(os.path.join(args.projectFolder,'Report.txt'), 'a', encoding="utf-8")
    reportFile.write('-' * 30 + '\n')
    reportFile.write(str(args))
    reportFile.write('-' * 30 + '\n')

    print('\nLOAD THE DATASET FOR TRAINING...\n')     
    patientsList, labelsList, args.csvFile = ConcatCohorts_Classic(imagesPath = args.datadir_train, 
                                                              cliniTablePath = args.clini_dir, slideTablePath = args.slide_dir,
                                                              label = targetLabel, minNumberOfTiles = args.minNumBlocks,
                                                              outputPath = args.projectFolder, reportFile = reportFile, csvName = args.csv_name,
                                                              patientNumber = args.numPatientToUse, additionalData = additionalData)                        
    labelsList = utils.CheckForTargetType(labelsList)            
    le = preprocessing.LabelEncoder()
    labelsList = le.fit_transform(labelsList)            
    args.num_classes = len(set(labelsList))
    args.target_labelDict = dict(zip(le.classes_, range(len(le.classes_))))                  
    utils.Summarize(args, list(labelsList), reportFile)        
    
    tempData  = pd.DataFrame(list(zip(patientsList, labelsList)), columns = ['PATIENT', 'LABEL'])
    v_data = tempData.groupby('LABEL', group_keys = False).apply(lambda x: x.sample(frac = 0.3)) 
    t_data = tempData[~tempData['PATIENT'].isin(v_data['PATIENT'])]

    print('GENERATE NEW TILES...')                            
    train_data = GetTiles(csvFile = args.csvFile, label = targetLabel, target_labelDict = args.target_labelDict, 
                      maxBlockNum = args.maxBlockNum, test = False, filterPatients = list(t_data['PATIENT']))                
    train_x = list(train_data['TilePath'])
    train_y = list(train_data['yTrue'])  
    train_data.to_csv(os.path.join(args.split_dir, 'TrainSplit.csv'), index = False)
    
    val_data = GetTiles(csvFile = args.csvFile, label = targetLabel, target_labelDict = args.target_labelDict, 
                      maxBlockNum = args.maxBlockNum, test = True, filterPatients = list(v_data['PATIENT'])) 
    
    val_x = list(val_data['TilePath']) 
    val_y = list(val_data['yTrue'])                  
    val_data.to_csv(os.path.join(args.split_dir, 'ValSplit.csv'), index = False)        
    print()
    print('-' * 30)        
    model, input_size = utils.Initialize_model(model_name = args.model_name, num_classes = args.num_classes, feature_extract = False, use_pretrained = True)
    model = model.to(device) 

    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': 16,
              'pin_memory' : False}

    train_set = DatasetLoader(train_x, train_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
    trainGenerator = torch.utils.data.DataLoader(train_set, **params)
    params = {'batch_size': args.batch_size,
              'shuffle': False,
              'num_workers': 16,
              'pin_memory' : False}
    val_set = DatasetLoader(val_x, val_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
    valGenerator = torch.utils.data.DataLoader(val_set, **params)
    for fr in freezRatios:  
                
        model, input_size = utils.Initialize_model(model_name = args.model_name, num_classes = args.num_classes, feature_extract = False, use_pretrained = True)
        model = model.to(device) 
            
        noOfLayers = 0
        for name, child in model.named_children():
            noOfLayers += 1            
        cut = int (fr * noOfLayers)                
        ct = 0
        for name, child in model.named_children():
            if ct < cut:
                for name2, params in child.named_parameters():
                    params.requires_grad = False
            ct += 1

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()
        lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
        lr_finder.range_test(trainGenerator, end_lr=100, num_iter=200)
        lr_finder.plot()
        lr_finder.reset()

        model, input_size = utils.Initialize_model(model_name = args.model_name, num_classes = args.num_classes, feature_extract = False, use_pretrained = True)
        model = model.to(device) 
            
        noOfLayers = 0
        for name, child in model.named_children():
            noOfLayers += 1            
        cut = int (fr * noOfLayers)                
        ct = 0
        for name, child in model.named_children():
            if ct < cut:
                for name2, params in child.named_parameters():
                    params.requires_grad = False
            ct += 1
            
        min_grad_idx = (np.gradient(np.array(lr_finder.history['loss']))).argmin()
        optimizer = torch.optim.Adam(model.parameters(), lr=np.round(lr_finder.history['lr'][min_grad_idx], 6), 
                                     weight_decay=1e-2)
        
        if args.adv_train:
            attack = FastGradientSignUntargeted(model = model, 
                                                epsilon = args.epsilon, 
                                                alpha = args.alpha, 
                                                min_val = 0, 
                                                max_val = 1, 
                                                max_iters = args.maxNoIteration, 
                                                _type=args.perturbationType)
        else:
            attack = None
                    
        model, train_loss_history, train_acc_history, val_acc_history, val_loss_history, spentTime = Train_model(model = model,
                                         trainLoaders = trainGenerator, valLoaders = valGenerator,
                                         criterion = criterion, optimizer = optimizer, args = args, reportFile = reportFile,
                                         advTrain = args.adv_train, attack = attack, additionalName =  '_' + str(fr))            
        print('-' * 30)
                
        torch.save(model.state_dict(), os.path.join(args.projectFolder, 'RESULTS', 'finalModel' + '_' + str(fr)))                
        history = pd.DataFrame(list(zip(train_loss_history, train_acc_history, val_loss_history, val_acc_history)), 
                          columns =['train_loss', 'train_acc', 'val_loss', 'val_acc'])                
        history.to_csv(os.path.join(args.result_dir, 'TRAIN_HISTORY_FULL' + str(fr) + '.csv'))
        reportFile.write('Training for ' + str(fr) + 'completed in ' + spentTime + '\n')
        reportFile.write('Optimum LR in Freeze Ratio of {} is {} \n'.format(fr, np.round(lr_finder.history['lr'][min_grad_idx], 6)))                
    reportFile.close()

























   
