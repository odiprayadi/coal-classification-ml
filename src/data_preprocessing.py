import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from collections import Counter

def organize_dataset(raw_path, organized_path):
    print("Dataset berada di:", raw_path)
    print("Isi dari direktori awal:")
    print(os.listdir(raw_path))

    checkpoints_dir = os.path.join(raw_path, '.ipynb_checkpoints')
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
        print("Folder '.ipynb_checkpoints' telah dihapus.")

    os.makedirs(organized_path, exist_ok=True)

    for folder_name in os.listdir(raw_path):
        folder_path = os.path.join(raw_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        label = folder_name.capitalize()
        class_dir = os.path.join(organized_path, label)
        os.makedirs(class_dir, exist_ok=True)

        for fname in os.listdir(folder_path):
            file_path = os.path.join(folder_path, fname)
            if os.path.isfile(file_path):
                shutil.copy(file_path, os.path.join(class_dir, fname))

    print("\nOrganisasi selesai. Folder hasil klasifikasi per jenis batubara:")
    print(os.listdir(organized_path))

def load_data(organized_path, img_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        rotation_range=10,
        zoom_range=0.1,
        shear_range=0.1,
        horizontal_flip=True
    )

    train = datagen.flow_from_directory(
        organized_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='training',
        shuffle=True
    )

    val = datagen.flow_from_directory(
        organized_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )

    # Optional: check class distribution
    y_train = train.classes
    print("Train class distribution:", Counter(y_train))
    
    return train, val
