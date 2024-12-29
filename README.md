# A CNN Model With Temporal Attention For Coronary Artery Disease Detection (CAD)

## Overview
This repository contains a deep-learning model for predicting Coronary Artery Disease (CAD) using cardiac MRI images from the public Sunnybrook Dataset. The model utilizes a Convolutional Neural Network (CNN) with a Temporal Attention mechanism to analyze medical imaging data and predict the likelihood of CAD.

## Features
1. Utilizes the Sunnybrook Cardiac MRI dataset
2. Implements a custom CNN architecture with Temporal Attention
3. Processes DICOM images for model input
4. Includes data augmentation and normalization techniques
5. Provides evaluation metrics including accuracy, precision, and recall

## Requirements
numpy==1.21.0
torch==1.9.0
torchvision==0.10.0
pydicom==2.2.2
Pillow==8.3.1
kagglehub==0.1.0
scikit-learn==0.24.2


## Results and Analysis
From our last training cycle, our model reached an Accuracy of 97.87%, Precision of 100%, Recall of 95.65%, and an Area Under the Curve (AUC) of 99.82%. The near perfect AUC means that the model can distinguish between patients with/without CAD, while the high recall, accuracy, and precision shows the model can accurately predict if a patient has CAD. 
