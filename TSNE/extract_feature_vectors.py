# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 

@author: dcifci
"""


import os
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from fastai.vision.all import *
from PIL import Image
import numpy
import pandas as pd
from dataclasses import dataclass
from pytorch_pretrained_vit import ViT
from attack import FastGradientSignUntargeted
from utils.data_utils import DatasetLoader
import torchvision
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print("No weight could be loaded..")
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    return model

@dataclass
class FeatureExtraction():
    
    spreadsheet_path: str
    name_project: str
    self_supervised: bool = False
    deep_med_model: str = None
    vit: bool = False
    attack: bool = False
    
    def transform_image(self, image_path: str) -> torch.Tensor:
        scaler = transforms.Scale((224, 224)) 
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]) 
        to_tensor = transforms.ToTensor() 
        img = Image.open(image_path) 
        if img.mode != "RGB":
            img = img.convert("RGB") 
        t_img = Variable(normalize(to_tensor(scaler(img)).to(device)).unsqueeze(0)) ##
        return t_img
    
    def get_feature_vectors(self, trans_img: torch.Tensor) -> torch.Tensor:
        """
        Extracts the feature vector of the given image from the avgpool layer of ResNet-18 and
        returns the feature vector as a tensor
        
        
        Parameters: 
        image_path (str): The absolute path to the image itself
        
        Returns: 
        embedding (torch.Tensor)
        
        """
        
        # embedding = torch.zeros(512) #512
        # #def copy_data(m, i, o):
        #  #   embedding.copy_(o.data.reshape(o.data.size(1)))
        # #h = self.layer.register_forward_hook(copy_data)
        # self.model.fc = nn.Sequential()
        # embedding = self.model(trans_img)
        # embedding = embedding.data.reshape(embedding.data.size(1))
        
        # return embedding
        self.model.eval()
        embedding = self.model(trans_img)
        return embedding
        
        
    def get_selfsup_feature_vectors(self, trans_img: torch.Tensor) -> torch.Tensor:
        out = self.model(trans_img)
        embedding = torch.squeeze(out)
        return embedding
        
    def extract_and_save_feature_vectors(self):
        """
        Extracts and saves the feature vectors of all images in the given spreadsheet and
        saves the feature vector and the labels of images as a tensor
        
        Parameters: 
        image_path (str): The absolute path to the image itself
        """
        # self.model = models.resnet50(pretrained=True).to(device)
        # layer = self.model._modules.get('avgpool')
        # self.model.eval()
        
        def get_label(x):
            i_image = df[df["file_names"] == x].index.values[0]
            label = df["labels"][i_image]
            if label.isnumeric():
                return torch.tensor(float(df["labels"][i_image])), float(df["labels"][i_image])
            else:
                unique_labels = sorted(df["labels"].unique())
                label_dict = dict()
                for i in range(len(unique_labels)):
                    label_dict[unique_labels[i]] = i
                # print(label_dict)
                return torch.tensor(float(label_dict[label])), float(label_dict[label])
        
        df = pd.read_csv(self.spreadsheet_path, dtype=str) #, engine='openpyxl'
        all_features = []
        labels = []
        image_names = []

        if self.vit:
            model = ViT('B_16_imagenet1k', pretrained = True, image_size = 224).to(device)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 3)   
            model.load_state_dict(torch.load(r"J:\Adversarial Project\TCGA_VIT_TRAINFULL_1\RESULTS\bestModel", map_location=lambda storage, loc: storage))   
            model.fc = nn.Identity()
            for p in model.parameters():
                p.requires_grad = False            
        else:
            model = models.resnet50(pretrained = True).to(device)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 3)            
            model.load_state_dict(torch.load(r"J:\Adversarial Project\TCGA_RN50_TRAINFULL_1\RESULTS\bestModel", map_location=lambda storage, loc: storage))   
            modules = list(model.children())[:-1]
            model = nn.Sequential(*modules)
            for p in model.parameters():
                p.requires_grad = False
                
        self.model = model
        self.model =  self.model.eval()
        if self.vit:
            modelATTACK = ViT('B_16_imagenet1k', pretrained = True, image_size = 224).to(device)
            num_ftrs = modelATTACK.fc.in_features
            modelATTACK.fc = nn.Linear(num_ftrs, 3)   
        else:
            modelATTACK = models.resnet50(pretrained = True).to(device)
            num_ftrs = modelATTACK.fc.in_features
            modelATTACK.fc = nn.Linear(num_ftrs, 3)   
        modelATTACK.eval()
        modelATTACK = modelATTACK.to(device)
         
        for im_name in tqdm(df["file_names"]):
            image_path = im_name
            #trans_im = self.transform_image(image_path)       
            label, labelAtatck = get_label(im_name)
            eps = 0.05
            params = {'batch_size': 1,
                  'shuffle': False,
                  'num_workers': 0}
            
            test_set = DatasetLoader([image_path], [labelAtatck], transform = torchvision.transforms.ToTensor, target_patch_size = 224)           
            testGenerator = torch.utils.data.DataLoader(test_set, **params)                    
            if self.attack:
                eps = 0.05
                attack = FastGradientSignUntargeted(model = modelATTACK, 
                                    epsilon = eps, 
                                    alpha = eps / 2,
                                    min_val = 0, 
                                    max_val = 1, 
                                    max_iters = 10, 
                                    _type='linf')  
                with torch.enable_grad():
                    for inputs, lbs in testGenerator :
                        inputs = inputs.to(device)
                        lbs = lbs.to(device)
                        # lbs = lbs.unsqueeze_(dim=1)
                        # lbs = lbs.unsqueeze_(dim=0)
                        trans_im = attack.perturb(original_images = inputs, labels = lbs, 
                                            reduction4loss = 'mean', random_start = False, bns = False, 
                                            exclusive = False)
            else:
                for inputs, lbs in testGenerator:
                    trans_im = inputs.to(device)
                        
            if self.self_supervised == False:
                image_vector = self.get_feature_vectors(trans_im)
                image_vector = image_vector.cpu().detach().numpy()[0]
            else:
                image_vector = self.get_selfsup_feature_vectors(trans_im)
           
            all_features.append(image_vector)
            labels.append(label)
            image_names.append(image_path)
        
        #all_features = torch.stack((all_features), dim=0)
        if self.self_supervised == False:
            np_all_features = all_features
        else:
            np_all_features = all_features.cpu().detach().numpy()
            
        labels = torch.stack((labels), dim=0)
        labels = labels.reshape(1,-1).t()
        np_labels = labels.numpy()
        numpy.save(self.name_project + "/feature_vectors.npy", np_all_features)
        numpy.save(self.name_project + "/labels.npy", np_labels)
        
        df_im = pd.DataFrame()
        df_im["im_names"] = image_names
        df_im.head()
        df_im.to_csv(self.name_project + "/images_names.csv", index=False)
        
    