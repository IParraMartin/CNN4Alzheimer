# Alzheimer Image Classification with CNNs 🧠

<p align="center">
  <img src="resources/img_readme.png" width="350" title="Example of a processed image">
</p>

Welcome to the Alzheimer Image Classification repository! This project aims to provide an educational example of a multiclass image classification problem using convolutional neural networks (CNNs).

## Project Overview 🔖
This repository contains the necessary code and guidance to build, train, and evaluate a CNN model that classifies brain scan images into four categories of dementia: non-demented, very mild, mild, and moderate dementia.

## Learning Objectives 👨🏽‍💻
By working through this project, you will learn how to:

- Use torchvision for image processing tasks
- Create a ```.csv``` file mapping images to their labels
- Convert images to tensors and vice versa
- Build and train a CNN model for multiclass classification
- Implement a compartmentalized training loop for efficient model training and evaluation

## Dataset 📁
The dataset used in this project is available on [Kaggle](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images). It includes brain scan images categorized into four classes:

- Non-Demented
- Very Mild Demented
- Mild Demented
- Moderate Demented

## Repository Contents 📦
This repository includes the following components:

- .csv annotations file builder (```annotate.py```)
- Dataset class: Handles data loading and preprocessing (```dataset.py```)
- Model definition: Defines the CNN architecture (```model.py```)
- Training script: Contains the training loop and model evaluation (```train.py```)

## How To Get Annotations File: Example (CLI)
You can pass the following arguments to get the .csv file with the image annotations and labels to use in the ```__getitem__``` method.
```
cd CNNMentia
python3 utils/annotate.py \
--root_dir ROOT_DIR_WITH_CLASS_SUBFOLDERS
--out_dir OUT_DIR
--col_names Images Labels
```
