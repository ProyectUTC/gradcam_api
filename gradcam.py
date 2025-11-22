import numpy as np
import cv2
import tensorflow as tf
import gc

# ============================================================
# 🔹 Preprocesamiento multicanal
# ============================================================
def preprocess_inputs_multichannel(img_bgr, size=224):

    # ---- COLOR ----
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size))
    x_color = rgb.astype(np.float32) / 255.0

    # ---- GRAY ----
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_3c = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    gray_3c = cv2.resize(gray_3c, (size, size))
    x_gray = gray_3c.astype(np.float32) / 255.0

    # ---- SEGMENTACIÓN SIMPLE ----
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 40, 20])   # rango verde
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    seg = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    seg = cv2.resize(seg, (size, size))
    x_seg = seg.astype(np.float32) / 255.0

    return (
        np.expand_dims(x_color, 0),
        np.expand_dims(x_gray, 0),
        np.expand_dims(x_seg, 0),
    )


# ============================================================
# 🔹 Buscar última conv
# ============================================================
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except:
            pass
    return None


# ============================================================
# 🔹 Grad-CAM sobre la rama COLOR
# ============================================================
def make_gradcam_heatmap(model, x_color, class_idx):

    last_conv = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            model.get_layer(last_conv).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model([x_color, x_color, x_color])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]

    heatmap = tf.reduce_mean(conv_out * pooled, axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0)
    heatmap /= (np.max(heatmap) + 1e-7)

    return heatmap


# ============================================================
# 🔹 Mezclar mapa con imagen
# ============================================================
def overlay_heatmap_on_image(img_bgr, heatmap, alpha=0.45):
    h, w = img_bgr.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)


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

