# Pathology_Adversarial

## Overview

This repository contains the Python version of a general workflow for Adversarial attacks and attack-proof artificial intelligence models in computational pathology.
It is based on workflows which were previously described in [Kather et al., Nature Medicine 2019](https://www.nature.com/articles/s41591-019-0462-y) and 
[Ghaffari Laleh et al. 2021](https://www.biorxiv.org/content/10.1101/2021.08.09.455633v1.full.pdf).
The objective is to predict a given *label* directly from digitized histological whole slide images (WSI) under multiple adversarial attacks. 
The *label* is defined on the level of *patients*, not on the level of pixels in a given WSI. The AI models can be trained normaly or robustly to adversarial attack, by defining the advTrain flag in the experiment file. Also the strength of attack during training can be defined by the value of epsilon in the experiment file. 

- Classical resnet-based training (similar to [Kather et al., Nature Medicine 2019](https://www.nature.com/articles/s41591-019-0462-y))
- Vision transformers (inspired by  Dosovitskiy et al., conference paper at ICLR 2021)(https://arxiv.org/abs/2010.11929)
- Dual batch training (inspired by HAn et al., Nature cimmunication at 2021)(https://www.nature.com/articles/s41467-021-24464-3)

This is important to notice that this repository follows exactly the same procedure as https://github.com/KatherLab/HIA. So it is recommended to check the initial repository, to be able to use the pathology_adversarial scripts. 

## Example data

The example dataset which has been pre-processed based on the modifications explained in this study, can be found in https://zenodo.org/record/5337009. This tiles are derived from the TCGA-BRCA breast cancer histology dataset at https://portal.gdc.cancer.gov/ (please check this website for the original data license). 

## System requirements

### Hardware requirements

The code in this repository requires a CUDA-enabled NVIDIA GPU for a fast and convenient training. However, it will detect automatically the present of the GPU in the system and run the rest of the codes correspondingly.

### Software requirements

#### OS Dependencies
The scripts in this repository have been developed on windoes server 2019 (version 1809).

#### Pathon Dependencies
This repository mainly depends on the following packages:

````
Pytorch 
Scikit-learn
Numpy
Pandas
OpenCV
pytorch_pretrained_vit
pickle
efficientnet_pytorch
torchvision
````
## Installation guide

To use this repository, it is recommended to check the https://github.com/KatherLab/HIA. The structure of the experiment file is same as it is been explained in HIA repository. Please install Python (e.g. Anaconda) on your windows system, and run the experiemnt file, filled with all the required information in the Main.py script. No installation required. The training and deployment time can vary based on the dataset size and the computational power of the system. 

## Demo and Instructions for use

To start training, you need to download the code in this repository and fill the experiment file, with the adresses to your training and in case to your validation set. It is necessary to define the target label which should be present in the clinical table. The number of epochs, batch size, learning rate can be defined in the experiment file. You can set the value of AdvTrain if you want to train your model adversarially robust and set it to False if you want to train it with the normal model. For all the othe details in experiement file, we would recomment to check the https://github.com/KatherLab/HIA. Here is the example of experiment file:

````
{
    "projectDetails":"This is the demo for adversarially, 3 fold-cross validation training"!",
    "dataDir_train":["D:\\Path to the folder containing the training data set."], # This folder contains subfolder for each WSI with the extracted tiles. 
    "dataDir_test":["E:\\ PAth to the folder contaning the test data set. "], # This is only required if you run the deployment script.

    "targetLabels":["RCC subtyping"], # This is the name of column in the clinical table which we want to use as a prediction label.
    "trainFull":"False", # Set to True, if you want to use all the data to train a model and then use this trained model for deployement. 
    "numPatientToUse" : "ALL", # You can use a portion of the patient for the training. When you set it to 'ALL', it will use all the patients present in the clinical table. 

    "maxNumBlocks":100, # Maximum number of tiles to select for each WSI. 
    "minNumBlocks" : 8, # Minimum number of tiles to which needs to be present in the WSI.

    "epochs":50, # Maximum number of epochs to train the model. If earlyStop set to be True, the training can be stopped before reaching this number. 
    "batchSize":128,
    "freezeRatio" : 0.5,
    "repeatExperiment" : 5, #The rxperiemnt will be repeated 5 times with different random seeds. 
     
    "modelName":"vit", 
    "opt":"adam",
    "lr":0.0001,
    "reg":0.00001,
    "gpuNo":1,

    "earlyStop":"True",
    "minEpochToTrain":10, 
    "patience":5,

    "advTrain" : "True", #If set to be True, then the model will be trained robustly. 
    "epsilon" : 0.005,
    "alpha" : 0.0025,
    "maxNoIteration" : 10,
    "perturbationType" : "linf"
}

````

## Expected output
This repository mainly gives the area under the curve values as a output during the deployment. If you use k-fold cross entropy option, it will print the AUC values after each fold for the test set and at the end, it will concatenate the result of all folds. For a single deployment, it will print the AUC values at the end of deployment. It also saves all the results in a report.txt file with the confidence intervals for the calculated AUC values.  

## License
This project is covered under the MIT license. 
