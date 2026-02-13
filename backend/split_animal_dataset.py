import os, shutil, random

SRC = "dataset_animal"
DST = "dataset_animal_split"
RATIO = 0.8
random.seed(42)

for cls in ["dog", "cat"]:
    files = os.listdir(os.path.join(SRC, cls))
    random.shuffle(files)
    split = int(len(files) * RATIO)

    for f in files[:split]:
        os.makedirs(os.path.join(DST, "train", cls), exist_ok=True)
        shutil.copy(os.path.join(SRC, cls, f), os.path.join(DST, "train", cls, f))

    for f in files[split:]:
        os.makedirs(os.path.join(DST, "val", cls), exist_ok=True)
        shutil.copy(os.path.join(SRC, cls, f), os.path.join(DST, "val", cls, f))

print("✅ Animal dataset split done")
