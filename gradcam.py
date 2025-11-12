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
# 🔹 Buscar la última capa convolucional
# ============================================================
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    return "Conv_1"


# ============================================================
# 🔹 Grad-CAM
# ============================================================
def make_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    # Asegura que el modelo sea compatible (multioutput → solo primera salida)
    model_output = model.output
    if isinstance(model_output, (list, tuple)):
        model_output = model_output[0]

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model_output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Si predictions es lista o tupla → tomar primer tensor
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        # Asegurar vector correcto
        predictions = tf.reshape(predictions, [-1])
        class_idx = tf.argmax(predictions)
        loss = predictions[class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap.numpy(), 0)
    maxv = np.max(heatmap) if np.max(heatmap) > 0 else 1e-7
    heatmap /= maxv

    # Convertir class_idx a entero puro
    class_idx = int(class_idx.numpy()) if hasattr(class_idx, "numpy") else int(class_idx)

    # Pasar predicciones a numpy (vector)
    preds_np = predictions.numpy() if hasattr(predictions, "numpy") else np.array(predictions)
    preds_np = np.squeeze(preds_np)

    return heatmap, class_idx, preds_np


# ============================================================
# 🔹 Superponer mapa de calor
# ============================================================
def overlay_heatmap_on_image(orig_bgr, heatmap, alpha=0.4):
    h, w = orig_bgr.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(orig_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
    return blended


# ============================================================
# 🔹 Liberar memoria
# ============================================================
def release_tf_memory(model=None):
    try:
        tf.keras.backend.clear_session()
        del model
        gc.collect()
    except Exception:
        pass
