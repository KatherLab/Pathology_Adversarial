# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:30:07 2021

@author: Narmin Ghaffari Laleh
"""

##############################################################################  

from attack.fast_gradient_sign_untargeted import FastGradientSignUntargeted
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import time
import torch
import os
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################
    
class EarlyStopping:
    def __init__(self, patience = 20, stop_epoch = 50, verbose=False):

        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score >= self.best_score:
            self.counter += 1
            print(f'\nEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter > self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss    
        

##############################################################################    
    
def Train_model(model, trainLoaders, args, valLoaders = [], criterion = None, 
                        optimizer = None, reportFile = None, advTrain = False, attack = None, additionalName = None):    
    
    since = time.time()    
    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []  


    if args.early_stopping:
        early_stopping = EarlyStopping(patience = args.patience, stop_epoch = args.minEpochToTrain, verbose = True)    
    for epoch in range(args.max_epochs):
        phase = 'train'
        print('Epoch {}/{}\n'.format(epoch, args.max_epochs - 1))
        print('\nTRAINING...\n')        
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(trainLoaders):
            model.train()
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                inputs = inputs.to(device)
                labels = labels.to(device)
                if args.adv_train == True and args.model_name == 'bns':
                    adv_images = attack.perturb(original_images = inputs, labels = labels, 
                                                reduction4loss = 'mean', random_start = True, bns = True, 
                                                exclusive = False)
                    model.train()
                    output_ = model(adv_images, [1])
                    output = model(inputs, [0])
                    labels = labels.type(torch.long)
                    loss = criterion(output, labels)
                    loss_ = criterion(output_, labels)
                    loss_t = loss + loss_ 
                elif args.adv_train == False and args.model_name == 'bns':
                    model.train()
                    output = model(inputs, [0])
                    labels = labels.type(torch.long)
                    loss_t = criterion(output, labels)
                elif args.adv_train == True and (not args.model_name == 'bns'):
                    adv_images = attack.perturb(original_images = inputs, labels = labels, 
                            reduction4loss = 'mean', random_start = True, bns = False, 
                            exclusive = False)
                    model.train()
                    output = model(adv_images)
                    labels = labels.type(torch.long)
                    loss_t = criterion(output, labels)
                elif args.adv_train == False and (not args.model_name == 'bns'):
                    output = model(inputs)
                    labels = labels.type(torch.long)
                    loss_t = criterion(output, labels)                
                loss_t.backward() 
                optimizer.step()                                    
                _, y_hat = torch.max(output, 1)                        
                running_loss += loss_t.item() * inputs.size(0)
                running_corrects += torch.sum(y_hat == labels.data)
        epoch_loss = running_loss / len(trainLoaders.dataset)
        epoch_acc = running_corrects.double() / len(trainLoaders.dataset)        
        train_acc_history.append(epoch_acc.item())
        train_loss_history.append(epoch_loss)   
        
        print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        print()  
        if valLoaders:
            print('VALIDATION...\n')
            phase = 'val'    
            model.eval()        
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in tqdm(valLoaders):
                inputs = inputs.to(device)
                labels = labels.to(device)        
                with torch.set_grad_enabled(phase == 'train'):            
                    if args.model_name == 'bns':
                        output = model(inputs, [0])
                        labels = labels.type(torch.long)
                        loss_t = criterion(output, labels)
                    else:
                        output = model(inputs)
                        loss_t = criterion(output, labels)
                    _, y_hat = torch.max(output, 1)  
                    running_loss += loss_t.item() * inputs.size(0)
                    running_corrects += torch.sum(y_hat == labels.data)                    
            val_loss = running_loss / len(valLoaders.dataset)
            val_acc = running_corrects.double() / len(valLoaders.dataset) 
                        
            val_acc_history.append(val_acc.item())
            val_loss_history.append(val_loss)
            print('\n{} Loss: {:.4f} Acc: {:.4f} '.format(phase, val_loss, val_acc)) 
            ckpt_name = os.path.join(args.result_dir, "bestModel")
            if not additionalName == None:
                ckpt_name = ckpt_name +  additionalName
            early_stopping(epoch, val_loss, model, ckpt_name = ckpt_name)
            if early_stopping.early_stop:
                print('-' * 30)
                print("The Validation Loss Didn't Decrease, Early Stopping!!")
                print('-' * 30)
                break
            print('-' * 30)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    spentTime = str(time_elapsed // 60) + ' M ' + str(time_elapsed % 60) + ' S'
    return model, train_loss_history, train_acc_history, val_acc_history, val_loss_history, spentTime 
    
    
##############################################################################    
    
def Validate_model(model, dataloaders, criterion, attackFlag = False, bns = False, attack = None, exclusive = False):
    
    phase = 'test'
    model.eval()
    predList = []                                            
    for inputs, labels in tqdm(dataloaders):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if bns:
            if attackFlag:
                with torch.enable_grad():
                    images = attack.perturb(original_images = inputs, labels = labels, 
                                            reduction4loss = 'mean', random_start = False, bns = True, 
                                            exclusive = False)
                model.eval()
                outputs = nn.Softmax(dim=1)(model(images, [1]))                
            else:
                images = inputs
                with torch.set_grad_enabled(phase == 'train'): 
                    model.eval()
                    outputs = nn.Softmax(dim=1)(model(images, [0])) 
        else:
              if attackFlag:
                  with torch.enable_grad():
                       images= attack.perturb(original_images = inputs, labels = labels, 
                                             reduction4loss = 'mean', random_start = False, bns = False, 
                                             exclusive = False)
                      #images = attack(inputs, labels)
              else:
                images = inputs
                
              with torch.set_grad_enabled(phase == 'train'):   
                model.eval()
                outputs = nn.Softmax(dim=-1)(model(images)) 

        predList = predList + outputs.tolist()
    return predList 
        
##############################################################################    
    

    
    
    
    
    
    
    
    
    
    