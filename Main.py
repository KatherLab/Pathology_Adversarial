# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:08:57 2021

@author: Narmin Ghaffari Laleh
"""

###############################################################################

import utils.utils as utils
import warnings
import argparse
import torch
from Training import Training

###############################################################################

    
parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
parser.add_argument('--adressExp', type = str,
default = r"",  help = 'Adress to the experiment File')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
print('\nTORCH Detected: {}\n'.format(device))

###############################################################################

if __name__ == '__main__':
        
    args = utils.ReadExperimentFile(args)    
    torch.cuda.set_device(args.gpuNo)
    Training(args)                

        
        
        
         
        
        
        
        
        
        
        
        
        
        
        
        