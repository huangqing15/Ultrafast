# Ultrafast 3D segmentation of brain-wide optical neuronal volume

## Overview:

This study aimed to design an ultrafast method, Simplified Deep-Layer Aggregation Supervision Network (SDASN), to segment complex neuronal morphology from large-scale micro-optical image. SDASN is a highly direct integrated streamlined network accelerated by TensorRT. It realized an unattainable inference speed (~0.3s on a 300¡Á300¡Á300 volume while 7.3s for 3D UNet), and outperformed current novel methods on different low SNR images prediction. Neurons in TB-sized brain images were segmented by SDASN with few manual annotations and a sparse data reduction strategy in several hours using one computer effectively and generalized, further promoting neuron tracing and analysis. We also published our 3D neuronal training dataset to encourage deep learning usage in large-scale neuronal volume processing.

## System Requirements

### Hardware Requirements:

The deep learning algorithm requires enough RAM and GPU to support the calculation. For optimal performace, we recommenda computer with the following specs:

- RAM: 16+GB
- CPU: Intel i5 or better
- GPU:  2080Ti or better

### Environment Requirements:

- Nvidia GPU corresponding driver
- CUDA: cuda 9.0
- cudnn: cudnn 7
- Python: 3.6
- pytorch:0.4.1 
- visdom:0.1.8.5
- Numpy: 1.14.5
- tifffile: 0.15.1
- Scikit-image:0.13.1
- tensorRT

## Functions:

For interactive demos of the functions, please give the file paths that include the training and testing images via the Train or Predict python fileS. You can also adjust some paramters for better training or testing in your own computer. The python file config.py is used for configuration of the packages.  Paths and training or testing parameters can be adjusted via this file.

## three main functions:

Fast_MyNet5.py:  The network architecture of SDASN for fast segmentation.
Train_Supervise.py: Realizing the training process for a new training or transfer learning for fine tuning.
Predict.py: Loading the trained model for predcting of new testing image.

## Models:
We provide some trained models named Fast_MyNet5_Re300_epoch_X.ckpt in the 'checkpoints' file.

## Test Dataset:

We also include 6 testing images for testing  in the 'image' file under the 'test_dataset' file. 
The datasets can be accessed via: https://pan.baidu.com/s/1_7TP1-p5KLkiOFiG3iGk-A. 
Its extraction code is: dbhq.



## 3D Training datasets for Brain-scale:

We also publish our typical training datasets with 300 neuronal images from brain-scale neuronal dataset to encourage the study of neuron reconstruction. 
The datasets can be accessed via: https://pan.baidu.com/s/16rp-YSyM3ttitus5h3jvMA.  The code for the datasets and more detailed informations can be provided by email: hustsy2008@163.com.
