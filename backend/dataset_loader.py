import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = "dataset"
IMG_SIZE = 224

images = []
labels = []

for animal in os.listdir(DATASET_PATH):
    animal_path = os.path.join(DATASET_PATH, animal)
    if not os.path.isdir(animal_path):
        continue

    for body_part in os.listdir(animal_path):
        body_part_path = os.path.join(animal_path, body_part)
        if not os.path.isdir(body_part_path):
            continue

        # CASE 1: images directly inside body_part (e.g. dental)
        for item in os.listdir(body_part_path):
            item_path = os.path.join(body_part_path, item)

            if os.path.isfile(item_path):
                try:
                    img = cv2.imread(item_path)
                    if img is None:
                        continue

                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0

                    images.append(img)
                    labels.append(f"{animal}_{body_part}")

                except Exception as e:
                    print("Error:", item_path)
                continue

            # CASE 2: disease folder exists
            if os.path.isdir(item_path):
                disease = item
                for img_name in os.listdir(item_path):
                    img_path = os.path.join(item_path, img_name)

                    if not os.path.isfile(img_path):
                        continue

                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            continue

                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        img = img / 255.0

                        images.append(img)
                        labels.append(f"{animal}_{body_part}_{disease}")

                    except Exception as e:
                        print("Error:", img_path)

X = np.array(images)
y = np.array(labels)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\n✅ DATASET LOADED SUCCESSFULLY")
print("Total images :", X.shape[0])
print("Image shape  :", X.shape[1:])
print("Total classes:", len(encoder.classes_))
print("Classes:")
for c in encoder.classes_:
    print("-", c)
