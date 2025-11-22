import numpy as np
import cv2
import tensorflow as tf
import gc


# ============================================================
# 🔹 Preprocesamiento multicanal
# ============================================================

def preprocess_multi_inputs(img_bgr, size=224):

    # --- Color ---
    color = cv2.resize(img_bgr, (size, size)).astype(np.float32) / 255.0

    # --- Grayscale ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gray = cv2.resize(gray, (size, size)).astype(np.float32) / 255.0

    # --- Segmentación rápida ---
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 40, 40])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    seg = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    seg = cv2.resize(seg, (size, size)).astype(np.float32) / 255.0

    return (
        np.expand_dims(color, 0),
        np.expand_dims(gray, 0),
        np.expand_dims(seg, 0)
    )


# ============================================================
# 🔹 GradCAM para modelos multicanal
# ============================================================

def make_gradcam_multibranch(model, inputs, class_idx):

    # Obtener la última capa convolucional de cualquiera de las ramas
    last_conv_name = None
    for layer in model.layers[::-1]:
        if len(layer.output.shape) == 4:
            last_conv_name = layer.name
            break

    if not last_conv_name:
        raise ValueError("No se encontró capa conv final.")

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outs, preds = grad_model(inputs)
        preds = preds[0]
        loss = preds[class_idx]

    grads = tape.gradient(loss, conv_outs)[0]
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    conv = conv_outs[0]

    heatmap = tf.reduce_mean(conv * pooled, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap + 1e-7)

    return heatmap


# ============================================================
# 🔹 Superponer heatmap
# ============================================================

def overlay_heatmap_on_image(orig_bgr, heatmap, alpha=0.45):
    h, w = orig_bgr.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(orig_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return blended


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
