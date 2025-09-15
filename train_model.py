import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")


def prepare_datasets(dataset_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(dataset_dir, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(dataset_dir, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    class_names = train_ds.class_names
    return train_ds, val_ds, class_names


def prepare(ds, shuffle=False, augment=False):
    if shuffle:
        ds = ds.shuffle(1000)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def build_model(num_classes):
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    base = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg")
    base.trainable = False  

    x = layers.Dropout(0.3)(base.output)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_model(dataset_dir, save_path):
    print("Training Stage-1 (Leaf vs Fruit)")

    train_ds, val_ds, class_names = prepare_datasets(dataset_dir)

    train_ds = prepare(train_ds, shuffle=True, augment=True)
    val_ds = prepare(val_ds)

    model = build_model(len(class_names))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )

    model.save(save_path)
    return model, class_names


if __name__ == "__main__":
    DATASET_STAGE1 = "stage1"
    os.makedirs("saved_models", exist_ok=True)
    stage1_model, stage1_classes = train_model(DATASET_STAGE1, "saved_models/stage1.keras")
    print("Training complete. Classes:", stage1_classes)
