# Pathology_Adversarial

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

