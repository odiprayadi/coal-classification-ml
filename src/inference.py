import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

def predict_image(model, img_path, img_size=(224, 224)):
    """Predict class for a single image."""
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return np.argmax(prediction), np.max(prediction)

if __name__ == "__main__":
    # Load model
    model = load_model('coal_transfer_model.h5')

    # Make prediction
    img_path = 'path/to/your/image.jpg'
    class_index, confidence = predict_image(model, img_path)
    print(f"Predicted class: {class_index}, Confidence: {confidence * 100:.2f}%")

