import glob
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from collections import Counter
import random
import matplotlib.pyplot as plt

def find_similar_images(input_img_path, dataset_path, base_model, img_size=(224, 224), top_k=3):
    input_img = load_img(input_img_path, target_size=img_size)
    input_array = img_to_array(input_img)
    input_array = preprocess_input(input_array)
    input_array = np.expand_dims(input_array, axis=0)
    input_feature = base_model.predict(input_array)

    similarities = []
    image_paths = glob.glob(os.path.join(dataset_path, "*", "*.jpg"))

    for img_file in image_paths:
        img = load_img(img_file, target_size=img_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        feat = base_model.predict(img_array)
        sim = cosine_similarity(input_feature.reshape(1, -1), feat.reshape(1, -1))[0][0]
        similarities.append((img_file, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    for i in range(top_k):
        print(f"Top-{i+1} match: {similarities[i][0]}, similarity: {similarities[i][1]:.4f}")
        plt.imshow(load_img(similarities[i][0]))
        plt.title(f"Match {i+1}")
        plt.axis('off')
        plt.show()

def show_examples_from_predicted_class(predicted_label, dataset_path, max_samples=3):
    folder_path = os.path.join(dataset_path, predicted_label)
    if not os.path.exists(folder_path):
        print(f"Folder for class {predicted_label} not found in {dataset_path}")
        return

    sample_images = glob.glob(os.path.join(folder_path, "*.jpg"))
    if len(sample_images) == 0:
        print(f"No images found in the folder {folder_path}")
        return

    sample_images = random.sample(sample_images, min(len(sample_images), max_samples))
    for img_file in sample_images:
        img = load_img(img_file)
        plt.imshow(img)
        plt.title(f"Example from Class {predicted_label}")
        plt.axis('off')
        plt.show()
