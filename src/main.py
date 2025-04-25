import os
from src.data_preprocessing import organize_dataset, load_data
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.inference import predict_image

# Define paths
raw_path = '/content/coal_data_set/Coal Classification'
organized_path = '/content/coal_data_set/Coal Classification Organized'

# Step 1: Organize dataset
organize_dataset(raw_path, organized_path)

# Step 2: Load dataset
train, val = load_data(organized_path)

# Step 3: Build the model
model = build_model(train.num_classes)

# Step 4: Train the model
history = train_model(model, train, val, train.classes)

# Step 5: Evaluate the model
evaluate_model(model, val)

# Step 6: Predict a custom image
img_path = '/content/testt.jpeg'
pred_class, confidence = predict_image(model, img_path, class_indices=val.class_indices)
print(f"Predicted class: {pred_class} with confidence: {confidence:.2f}%")
