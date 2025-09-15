import tensorflow as tf
from tensorflow.keras import layers, models
import os

train_dir = r"C:\Users\DELL\Desktop\projects\grape disease detection\data\train"  
val_dir   = r"C:\Users\DELL\Desktop\projects\grape disease detection\data\test"   

img_size = (128, 128)
batch_size = 32
epochs = 15

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Leaf Disease Classes:", class_names)

train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds   = val_ds.map(lambda x, y: (x / 255.0, y))

def random_brightness(x):
    return tf.image.random_brightness(x, max_delta=0.2)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.Lambda(random_brightness),   
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
])

model = models.Sequential([
    layers.Input(shape=(*img_size, 3)),
    data_augmentation,

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training Leaf-Only Grape Disease Model...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

os.makedirs("models", exist_ok=True)
model.save("models/unified_leaf_model.keras")
print("Leaf-Only Model Saved: models/unified_leaf_model.keras")
