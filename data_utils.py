from pathlib import Path
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (224, 224)

def load_datasets(data_dir, batch_size=32, seed=42):
    """Load class-folder data with deterministic train/validation/test splits."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    common = dict(
        directory=str(data_dir),
        labels="inferred",
        label_mode="categorical",
        class_names=sorted([p.name for p in data_dir.iterdir() if p.is_dir()]),
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        seed=seed,
        validation_split=0.2,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        subset="training", **common
    )
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        subset="validation", **common
    )
    # Split the 20% holdout evenly into validation and test.
    holdout_batches = list(val_test_ds.unbatch().batch(batch_size))
    if len(holdout_batches) < 2:
        raise ValueError("Not enough images for a separate validation and test split.")
    midpoint = max(1, len(holdout_batches) // 2)
    val_ds = tf.data.Dataset.from_tensor_slices(holdout_batches[:midpoint]).unbatch().batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices(holdout_batches[midpoint:]).unbatch().batch(batch_size)

    class_names = train_ds.class_names
    train_ds = train_ds.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y),
                            num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y),
                        num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y),
                          num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names
