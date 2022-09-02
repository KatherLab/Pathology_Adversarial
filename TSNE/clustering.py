# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 2021

@author: dcifci
"""

import numpy as np
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
from dataclasses import dataclass
from sklearn.decomposition import PCA

@dataclass
class DataClustering():
    """A class for holding clustering objects"""
    name_project: str 
    npy_feature_vectors: str 
    npy_labels: str 
    ss_im_names: str 
    
    
    def load_data(self):
        self.df = pd.read_csv(self.ss_im_names)
        self.X = np.load(self.npy_feature_vectors)
        self.y = np.load(self.npy_labels)
    
    def cluster_tSNE(self):
        self.load_data()
        tsne = TSNE(n_components=2)
        X_2d = tsne.fit_transform(self.X)
        self.df["labels"] = self.y.T.reshape(-1)
        self.df["comp-1"] = X_2d[:,0]
        self.df["comp-2"] = X_2d[:,1]
        
    def cluster_PCA_tSNE(self):
        self.load_data()
        
        pca = PCA(n_components=50)
        pca_result = pca.fit_transform(self.X)
        
        tsne = TSNE(n_components=2, perplexity=5)
        X_2d = tsne.fit_transform(pca_result)
        self.df["labels"] = self.y.T.reshape(-1)
        self.df["comp-1"] = X_2d[:,0]
        self.df["comp-2"] = X_2d[:,1]
        
    def plot_scatter(self):
        plt.figure(figsize=(16,10)) 
        n_classes = len(self.df["labels"].unique())
        sns.scatterplot(x="comp-1", y="comp-2", hue=self.df.labels.tolist(), 
                        palette=sns.color_palette("hls", n_classes), 
                        data=self.df).set(title="T-SNE projection")
        #plt.show()
        plt.savefig("./out_visuals/out_scatter_dots_" + self.name_project + ".png")
        
    def plot_imgs(self):
        fig, ax = plt.subplots()
        plt.figure(figsize=(16,10))
        x_min = int(self.df["comp-1"].min())
        x_max = int(self.df["comp-1"].max())
        y_min = int(self.df["comp-2"].min())
        y_max = int(self.df["comp-2"].max())
        
        for im in self.df["im_names"]:
            with mpl.cbook.get_sample_data(im) as file: 
                arr_image = plt.imread(file)
            
            ax.set_xlim([x_min-30, x_max+30])
            ax.set_ylim([y_min-30, y_max+30])
            ax.set_title("t-SNE projection with images")
            ax.set_xlabel('comp-1')
            ax.set_ylabel('comp-2') 
            ax.lines = [] 
            #ax.axis("off") 
            #ax.set_visible(False)
            idx = self.df[self.df["im_names"] == im].index.values[0] 
            x = self.df["comp-1"][idx] 
            y = self.df["comp-2"][idx] 
            axin = ax.inset_axes([x,y,2,2],transform=ax.transData) 
            axin.imshow(arr_image, cmap="gray") 
            axin.axis("off")
        #plt.show()
        
        # fig.set_size_inches(20, 20)
        fig.savefig("./out_visuals/out_imgs_" + self.name_project + ".tiff", format="tiff", dpi=1000)
        
    