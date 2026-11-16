import tensorflow as tf

def build_model(num_classes, image_size=(224, 224), train_base=False):
    inputs = tf.keras.Input(shape=(*image_size, 3), name="leaf_image")
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3), include_top=False,
        weights="imagenet", pooling=None
    )
    base.trainable = train_base
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
