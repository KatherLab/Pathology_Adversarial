# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 2021

@author: dcifci
"""

import os
import argparse
from TSNE.extract_feature_vectors import FeatureExtraction 
from TSNE.clustering import DataClustering


def Run_TSNE(path_to_spreadsheet, name_project, model):
           
    v_features = "./data/feature_vectors_" + name_project + ".npy"
    v_labels = "./data/labels_" + name_project + ".npy"
    df_images = "./data/images_names_" + name_project + ".csv"
    
    if os.path.exists(v_features) and os.path.exists(v_labels) and os.path.exists(df_images): 
        if not os.path.exists("./out_visuals"):
            os.mkdir("./out_visuals")
        dc = DataClustering(name_project, v_features, v_labels, df_images)
        dc.cluster_tSNE(); dc.plot_scatter(); dc.plot_imgs()
        
    else:
        if not os.path.exists("./data"):
            os.mkdir("./data")
            

    fc = FeatureExtraction(path_to_spreadsheet, name_project, deep_med_model = None)
    fc.extract_and_save_feature_vectors()
    
    dc = DataClustering(name_project, v_features, v_labels, df_images)
    dc.cluster_PCA_tSNE()
    dc.plot_scatter()
    
    
    
    dc.plot_imgs()
            
        
