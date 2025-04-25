from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

def predict_image(model, img_path, img_size=(224, 224), class_indices=None):
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    pred_class = np.argmax(pred)
    confidence = np.max(pred) * 100

    # Label mapping
    id_to_label = {v: k for k, v in class_indices.items()}

    plt.imshow(load_img(img_path))
    plt.axis('off')
    plt.title(f"Prediction: {id_to_label[pred_class]} ({confidence:.2f}%)")
    plt.show()

    return id_to_label[pred_class], confidence
