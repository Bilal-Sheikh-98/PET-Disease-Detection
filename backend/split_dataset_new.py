import os
import shutil
import random

SOURCE_DIR = "dataset"
DEST_DIR = "dataset_split"
SPLIT_RATIO = 0.8

random.seed(42)

def split_folder(src, train_dst, val_dst):
    files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    if len(files) == 0:
        return

    random.shuffle(files)
    split = int(len(files) * SPLIT_RATIO)

    train_files = files[:split]
    val_files = files[split:]

    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(val_dst, exist_ok=True)

    for f in train_files:
        shutil.copy(os.path.join(src, f), os.path.join(train_dst, f))
    for f in val_files:
        shutil.copy(os.path.join(src, f), os.path.join(val_dst, f))

for animal in os.listdir(SOURCE_DIR):
    animal_path = os.path.join(SOURCE_DIR, animal)
    if not os.path.isdir(animal_path):
        continue

    for disease in os.listdir(animal_path):
        disease_path = os.path.join(animal_path, disease)
        if not os.path.isdir(disease_path):
            continue

        split_folder(
            disease_path,
            os.path.join(DEST_DIR, "train", animal, disease),
            os.path.join(DEST_DIR, "val", animal, disease)
        )

print("✅ New dataset split completed successfully")
