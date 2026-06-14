# SEM Image Classification and Explainable AI Analysis

## Overview

This repository contains the source code used for image augmentation, classification, ensemble learning, and explainable artificial intelligence (XAI) analysis of scanning electron microscopy (SEM) images.

## Repository Structure

* `albumentations/` – Image augmentation pipeline implemented using Albumentations.
* `classification_xai/` – Classification, ensemble learning, and explainable AI analysis.
* `dataset/` – Dataset access information.

## Dataset

The original dataset can be accessed through the link provided in the `dataset` folder.

## Methodology

1. Original SEM images were augmented using Albumentations.
2. Classification models were developed using:

   * VGG19
   * ResNet50
   * InceptionV3
3. Ensemble learning was performed using the predictions from the individual models using Voting Mechanism.
4. Explainable AI techniques were applied to interpret model decisions.

## Requirements

* Python 3.x
* TensorFlow / Keras
* NumPy
* OpenCV
* Albumentations
* Matplotlib
* Scikit-learn
* LIME
* SHAP
* Alibi

## Usage

1. Download the dataset.
2. Run the augmentation pipeline.
3. Run the classification and XAI pipeline.
4. Review generated metrics, visualizations, and explanations.
