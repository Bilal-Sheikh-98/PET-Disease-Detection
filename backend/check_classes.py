from tensorflow.keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator()

# ===== CHECK DOG CLASSES =====
dog_data = gen.flow_from_directory("dataset_split/train/dog")
print("DOG disease class indices:", dog_data.class_indices)

# ===== CHECK CAT CLASSES =====
cat_data = gen.flow_from_directory("dataset_split/train/cat")
print("CAT disease class indices:", cat_data.class_indices)
