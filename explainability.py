import numpy as np
import tensorflow as tf
import cv2

def make_gradcam(model, preprocessed_batch, last_conv_layer_name="out_relu", class_index=None):
    """Return a normalized Grad-CAM heatmap for one preprocessed image batch."""
    base = model.get_layer("mobilenetv2_1.00_224")
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[base.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(preprocessed_batch, training=False)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_score = predictions[:, class_index]
    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_value + tf.keras.backend.epsilon())
    return heatmap.numpy()

def overlay_heatmap(rgb_image, heatmap, alpha=0.38):
    h, w = rgb_image.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip((1-alpha)*rgb_image + alpha*colored, 0, 255))
