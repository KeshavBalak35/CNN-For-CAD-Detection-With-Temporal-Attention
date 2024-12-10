# A CNN Model With Temporal Attention For Coronary Artery Disease Detection (CAD)

## Overview
This repository contains a deep-learning model for predicting Coronary Artery Disease (CAD) using cardiac MRI images. The model utilizes a Convolutional Neural Network (CNN) with a Temporal Attention mechanism to analyze medical imaging data and predict the likelihood of CAD.

## Features
1. Utilizes the Sunnybrook Cardiac MRI dataset
2. Implements a custom CNN architecture with Temporal Attention
3. Processes DICOM images for model input
4. Includes data augmentation and normalization techniques
5. Provides evaluation metrics including accuracy, precision, and recall

## Requirements
1. Python 3.7+
2. PyTorch
3. torchvision
4. numpy
5. pydicom
6. Pillow
7. scikit-learn
8. kagglehub
9. wfdb

## Results and Analysis
From our last training cycle, our model reached an Accuracy of 97.87%, Precision of 100%, Recall of 95.65%, and an Area Under the Curve (AUC) of 99.82%. The High AUC means that the model can distinguish between patients with/without CAD, while the high recall, accuracy, and precision shows the model can accurately predict if a patient has CAD. 





