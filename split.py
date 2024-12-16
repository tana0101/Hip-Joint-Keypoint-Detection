import os
import shutil
import random

image_folder = "dataset/xray_IHDI_2/images"
annotation_folder = "dataset/xray_IHDI_2/annotations"
output_folder = "data"

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_folder, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, split, 'annotations'), exist_ok=True)

images = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
annotations = sorted([f for f in os.listdir(annotation_folder) if f.endswith('.csv')])

assert len(images) == len(annotations)

# Split data into train, val, and test sets
train_split = 0.7
val_split = 0.15
test_split = 0.15

data = list(zip(images, annotations))
random.shuffle(data)

train_size = int(train_split * len(data))
val_size = int(val_split * len(data))
test_size = len(data) - train_size - val_size

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

for split, dataset in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
    for image, annotation in dataset:
        shutil.copy(os.path.join(image_folder, image), os.path.join(output_folder, split, 'images', image))
        shutil.copy(os.path.join(annotation_folder, annotation), os.path.join(output_folder, split, 'annotations', annotation))

print("Data split successfully!")
