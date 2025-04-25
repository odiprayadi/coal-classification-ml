# 🪨 Coal Classification using Transfer Learning (MobileNetV2)

This project uses **Transfer Learning with MobileNetV2** to classify coal images into four types: **Anthracite**, **Bituminous**, **Lignite**, and **Peat**. It includes data preprocessing, model training, evaluation, inference on custom images, and similarity search using deep features.

## 📁 Dataset
The dataset consists of coal images, each labeled according to its coal type:
- Anthracite
- Bituminous
- Lignite
- Peat

> 📌 Note: Due to copyright and file size limitations, the dataset is not included in this repository. You may request the dataset privately or prepare your own using similar folder structures.

## 🔧 Features
- Data preprocessing & organization
- Image augmentation
- Transfer Learning using MobileNetV2
- Class balancing with computed class weights
- Confusion matrix & classification report
- Custom image inference with confidence scores
- Similar image retrieval using cosine similarity
- Visualization of example predictions

## 🧠 Model Architecture
- **Backbone**: MobileNetV2 (pre-trained on ImageNet, frozen during training)
- **Classifier Head**:
  - GlobalAveragePooling2D
  - Dense (ReLU)
  - Dropout
  - Dense (Softmax)

## 🧪 Performance
- Achieved high validation accuracy with balanced results across classes.
- Includes visual reports and confusion matrix for model evaluation.

## 🖼️ Example Inference
![Prediction Example]([https://github.com/odiprayadi/coal-classification-ml/blob/main/results/prediction_result.png])

## 🔍 Find Similar Images
Given a custom input image, the model can retrieve top-N similar images from the training set based on cosine similarity of deep features.

## 📦 Requirements
Install dependencies using:
```bash
pip install -r requirements.txt

