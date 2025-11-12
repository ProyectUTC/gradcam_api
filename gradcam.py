import numpy as np
import cv2
import tensorflow as tf
import json, gc

# ===============================================================
# 🧩 PREPROCESAMIENTO DE IMAGEN
# ===============================================================

def preprocess_bgr_to_model(img_bgr, size=224):
    """Convierte una imagen BGR (OpenCV) al formato de entrada del modelo."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (size, size))
    x = img_rgb.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0)  # (1,224,224,3)


# ===============================================================
# 🔍 BUSCAR LA ÚLTIMA CAPA CONVOLUCIONAL
# ===============================================================

def find_last_conv_layer(model):
    """Devuelve el nombre de la última capa convolucional para Grad-CAM."""
    for layer in reversed(model.layers):
        try:
            out_shape = layer.output.shape
            if len(out_shape) == 4:
                return layer.name
        except:
            continue
    # Fallback común (MobileNetV2)
    return "Conv_1"


# ===============================================================
# 🔥 GENERAR HEATMAP (Grad-CAM)
# ===============================================================

def make_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    """Calcula el mapa de calor Grad-CAM para una imagen."""
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)  # (1,h,w,c)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (c,)

    conv_outputs = conv_outputs[0]  # (h,w,c)
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)  # (h,w)

    heatmap = np.maximum(heatmap.numpy(), 0)
    maxv = np.max(heatmap) if np.max(heatmap) > 0 else 1e-7
    heatmap /= maxv
    return heatmap, int(class_idx), predictions.numpy()[0]


# ===============================================================
# 🎨 SUPERPONER HEATMAP SOBRE LA IMAGEN ORIGINAL
# ===============================================================

def overlay_heatmap_on_image(orig_bgr, heatmap, alpha=0.4):
    """Combina la imagen original con el mapa de calor Grad-CAM."""
    h, w = orig_bgr.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(orig_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
    return blended


# ===============================================================
# 🧹 LIMPIEZA DE MEMORIA (para Render Free)
# ===============================================================

def release_tf_memory(model=None):
    """Libera memoria de TensorFlow para evitar OOM en Render Free."""
    try:
        if model is not None:
            del model
        gc.collect()
        tf.keras.backend.clear_session()
    except Exception as e:
        print(f"[WARN] Error al liberar memoria: {e}")
