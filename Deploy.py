# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 13:45:05 2021

@author: Narmin Ghaffari Laleh
"""
##############################################################################

import utils.utils as utils
from utils.core_utils import Validate_model
from utils.data_utils import ConcatCohorts_Classic, DatasetLoader, GetTiles
from eval.eval import CalculatePatientWiseAUC, GenerateHighScoreTiles
import torch.nn as nn
import torchvision
import pandas as pd
import argparse
import torch
import os
import random
from sklearn import preprocessing
from models.resnetdsbn import *
from attack import FastGradientSignUntargeted
import torchattacks
import shutil
import torchattacks

##############################################################################

if __name__ == '__main__':   
        
    for i in range(5):
        parser = argparse.ArgumentParser(description = 'Main Script to Run deployment')
        p = r""
        expTemp =r"".format(i+1)
        if not i == 0:
            shutil.copy(p, expTemp)
        modelPath = r"".format(i+1)
        parser.add_argument('--adressExp', type = str, 
                            default = expTemp, 
                            help = 'Adress to the experiment File')
        parser.add_argument('--modelAdr', type = str, 
                            default = modelPath,
                            help = 'Adress to the selected model')
        args = parser.parse_args()    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('\nTORCH Detected: {}\n'.format(device))
        
        useCSV = True    
        print(args.modelAdr)  
        
    ##############################################################################
 
        epsilons = [0.0, 0.25, 0.75,1.5]
        epsilons = [i * 0.001 for i in epsilons]
        additionalData = list(pd.read_csv(r"U:\WholeData.csv")['0'])  
         
        args = utils.ReadExperimentFile(args, deploy = True)    
        args.batch_size  = 16
        torch.cuda.set_device(args.gpuNo)
        random.seed(args.seed)        
        args.target_label = args.target_labels[0]  
        args.projectFolder = utils.CreateProjectFolder(ExName = args.project_name, ExAdr = args.adressExp, targetLabel = args.target_label,
                                                       model_name = args.model_name)
        args.useCSV = useCSV
        print('-' * 30 + '\n')
        print(args.projectFolder)
        if os.path.exists(args.projectFolder):
            continue
        else:
            os.makedirs(args.projectFolder, exist_ok = True)
            
        args.result_dir = os.path.join(args.projectFolder, 'RESULTS')
        os.makedirs(args.result_dir, exist_ok = True)
        args.split_dir = os.path.join(args.projectFolder, 'SPLITS')
        os.makedirs(args.split_dir, exist_ok = True)
           
        reportFile  = open(os.path.join(args.projectFolder,'Report.txt'), 'a', encoding="utf-8")
        reportFile.write('-' * 30 + '\n')
        reportFile.write(str(args))
        reportFile.write('-' * 30 + '\n')
        
        print('\nLOAD THE DATASET FOR TESTING...\n')     
        patientsList, labelsList, args.csvFile = ConcatCohorts_Classic(imagesPath = args.datadir_test, 
                                                                      cliniTablePath = args.clini_dir, slideTablePath = args.slide_dir,
                                                                      label = args.target_label, minNumberOfTiles = args.minNumBlocks,
                                                                      outputPath = args.projectFolder, reportFile = reportFile, csvName = args.csv_name,
                                                                      patientNumber = args.numPatientToUse,additionalData = additionalData)                        
        labelsList = utils.CheckForTargetType(labelsList)
        
        le = preprocessing.LabelEncoder()
        labelsList = le.fit_transform(labelsList)
        
        args.num_classes = len(set(labelsList))
        args.target_labelDict = dict(zip(le.classes_, range(len(le.classes_))))        
          
        utils.Summarize(args, list(labelsList), reportFile)
        print('-' * 30)
        print('IT IS A DEPLOYMENT FOR ' + args.target_label + '!')            
        print('GENERATE NEW TILES...') 
         
        if args.useCSV == True:
            path = r''.format(i+1)
            test_data = pd.read_csv(path)
            test_x = list(test_data['TilePath'])
            test_y = list(test_data['yTrue'])                
            test_data.to_csv(os.path.join(args.split_dir, 'TestSplit.csv'), index = False)                      
            print()
            print('-' * 30)
        
        else:                          
            test_data = GetTiles(csvFile = args.csvFile, label = args.target_label, target_labelDict = args.target_labelDict,
                                 maxBlockNum = args.maxBlockNum, test = True, seed = i)                
            test_x = list(test_data['TilePath'])
            test_y = list(test_data['yTrue'])                
            test_data.to_csv(os.path.join(args.split_dir, 'TestSplit.csv'), index = False)                      
            print()
            print('-' * 30)
                
        _, input_size = utils.Initialize_model(model_name = args.model_name, num_classes = args.num_classes,
                                                   feature_extract = False, use_pretrained = True)    
        
        params = {'batch_size': args.batch_size,
                  'shuffle': False,
                  'num_workers': 8}
            
        test_set = DatasetLoader(test_x, test_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)           
        testGenerator = torch.utils.data.DataLoader(test_set, **params)
        
        criterion = nn.CrossEntropyLoss()
        args.exclusive = False
        
        for eps in epsilons:    
            print('EPSILON  = {}'.format(eps)) 
            
            if args.model_name == 'bns':              
                model1 = resnet50dsbn(pretrained =True, widefactor=1)
                model1.fc = nn.Linear(model1.fc.in_features, args.num_classes)
                args.bns = True
            else:
                model1, input_size = utils.Initialize_model(model_name = args.model_name, num_classes = args.num_classes,                              
                                                          feature_extract = False, use_pretrained = True)   
   
                args.bns = False
                
            model1.eval()
            model1 = model1.to(device)
            if eps == 0.0:
                attack = None
            else:   
                attack = FastGradientSignUntargeted(model = model1, 
                                    epsilon = eps, 
                                    alpha = eps / 2,
                                    min_val = 0, 
                                    max_val = 1, 
                                    max_iters = args.maxNoIteration, 
                                    _type=args.perturbationType)
                print('MAX ITERATION = {}'.format(args.maxNoIteration))
                print('perturbationType = {}'.format(args.perturbationType))
                
            model1.load_state_dict(torch.load(args.modelAdr, map_location=lambda storage, loc: storage))   
            
            if eps == 0.0:
                attackFlag = False
            else:
                attackFlag = True
            probsList = Validate_model(model = model1, dataloaders = testGenerator, criterion = criterion, attackFlag = attackFlag,
                                         bns = args.bns, attack = attack, exclusive = args.exclusive)
            
            probs = {}
            for key in list(args.target_labelDict.keys()):
                probs[key] = []
                for item in probsList:
                    probs[key].append(item[utils.get_value_from_key(args.target_labelDict, key)])
        
            probs = pd.DataFrame.from_dict(probs)
            testResults = pd.concat([test_data, probs], axis = 1)
            
            testResultsPath = os.path.join(args.result_dir, 'TEST_RESULT_TILE_SCORES_FULL_'+ str(eps).replace('0.', '') + '.csv')
            testResults.to_csv(testResultsPath, index = False)
            totalPatientResultPath,_ = CalculatePatientWiseAUC(resultCSVPath = testResultsPath, args = args, eps = str(eps) , reportFile = reportFile, val = False)
            highScoreTilePath = os.path.join(args.result_dir, 'eps' + str(eps))
            os.makedirs(highScoreTilePath, exist_ok = True)
            GenerateHighScoreTiles(totalPatientResultPath = totalPatientResultPath, totalResultPath = testResultsPath, 
                                   numHighScorePetients = args.numHighScorePatients, numHighScoreTiles = args.numHighScorePatients,
                                   target_labelDict = args.target_labelDict, savePath = highScoreTilePath, attack = attack, eps = eps, bns = args.bns, input_size = input_size)                                        
        
        reportFile.write('-' * 100 + '\n')
        print('\n')
        print('-' * 30)
        reportFile.close()

                                    







                