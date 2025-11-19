import numpy as np
import cv2
import tensorflow as tf
import gc

# ============================================================
# 🔹 Preprocesamiento
# ============================================================
def preprocess_bgr_to_model(img_bgr, size=224):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (size, size))
    x = img_rgb.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0)


# ============================================================
# 🔹 Buscar última capa convolucional
# ============================================================
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except:
            continue
    return None  # Mejor que devolver algo incorrecto


# ============================================================
# 🔹 Grad-CAM con soporte para class_idx
# ============================================================
def make_gradcam_heatmap(model, img_array, class_idx=None, last_conv_layer_name=None):

    # Si no se especifica capa, encontrar la última conv
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    if last_conv_layer_name is None:
        raise ValueError("❌ No se encontró ninguna capa convolucional en el modelo.")

    # Crear modelo parcial (conv + output)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            model.get_layer(last_conv_layer_name).output,
            model.output if not isinstance(model.output, (list, tuple)) else model.output[0]
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Asegurar que predictions es un vector
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        predictions = tf.reshape(predictions, [-1])

        # Elegir la clase:
        if class_idx is None:
            class_idx = int(tf.argmax(predictions))
        else:
            class_idx = int(class_idx)

        loss = predictions[class_idx]

    # Calcular gradientes
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap.numpy(), 0)
    maxv = np.max(heatmap) if np.max(heatmap) > 0 else 1e-6
    heatmap /= maxv

    return heatmap, class_idx, predictions.numpy()


# ============================================================
# 🔹 Superponer mapa de calor
# ============================================================
def overlay_heatmap_on_image(orig_bgr, heatmap, alpha=0.45):
    h, w = orig_bgr.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(orig_bgr, 1 - alpha, heatmap_color, alpha, 0)


# ============================================================
# 🔹 Liberar memoria
# ============================================================
def release_tf_memory(model=None):
    try:
        tf.keras.backend.clear_session()
        del model
        gc.collect()
    except:
        pass

