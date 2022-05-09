# Pathology_Adversarial

## Overview

This repository contains the Python version of a general workflow for Adversarial attacks and attack-proof artificial intelligence models in computational patholog.
It is based on workflows which were previously described in [Kather et al., Nature Medicine 2019](https://www.nature.com/articles/s41591-019-0462-y) and 
[Ghaffari Laleh et al. 2021](https://www.biorxiv.org/content/10.1101/2021.08.09.455633v1.full.pdf).
The objective is to predict a given *label* directly from digitized histological whole slide images (WSI) under multiple adversarial attacks. 
The *label* is defined on the level of *patients*, not on the level of pixels in a given WSI. The AI models can be trained normaly or robustly to adversarial attack, by defining the
the advTrain flag in the experiment file. Also the strength of attack during training can be defined by the value of epsilon in the experiment file. 

- Classical resnet-based training (similar to [Kather et al., Nature Medicine 2019](https://www.nature.com/articles/s41591-019-0462-y))
- Vision transformers (inspired by  Dosovitskiy et al., conference paper at ICLR 2021)(https://arxiv.org/abs/2010.11929)
- Dual batch training (inspired by HAn et al., Nature cimmunication at 2021)(https://www.nature.com/articles/s41467-021-24464-3)

This is important to notice that this repository follows exactly the same procedure as https://github.com/KatherLab/HIA. So it is recommended to check the initial repository, to be able to use the pathology_adversarial scripts. 

## Example data

The example dataset which has been pre-processed based on the modifications explained in this study, can be found in https://zenodo.org/record/5337009. This tiles are derived from the TCGA-BRCA breast cancer histology dataset at https://portal.gdc.cancer.gov/ (please check this website for the original data license). 

## System requirements

The code in this repository has been developed on Windows Server 2019 and it requires a CUDA-enabled NVIDIA GPU and a Python installation of at least version 3.8. 

## Installation guide

To use this repository, it is recommended to check the https://github.com/KatherLab/HIA. The structure of the experiment file is same as it is been explained in HIA repository. Please install Python (e.g. Anaconda) on your windows system, and run the experiemnt file, filled with all the required information. No installation required. The training and deployment time can vary based on the dataset size and the computational power of the system. 

## Demo and Instructions for use

To start training, you need to download the code in this repository and fill the experiment file, with the adresses to your training and in case to your validation set. It is necessary to define the target label which should be present in the clinical table. The number of epochs, batch size, learning rate can be defined in the experiment file. You can set the value of AdvTrain if you want to train your model adversarially robust and set it to False if you want to train it with the normal model. For all the othe details in experiement file, we would recomment to check the https://github.com/KatherLab/HIA. 
