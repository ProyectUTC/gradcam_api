import numpy as np
import cv2
import tensorflow as tf

def preprocess_bgr_to_model(img_bgr, size=224):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (size, size))
    x = img_rgb.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0)  # (1,224,224,3)

def find_last_conv_layer(model):
    # Buscar la última capa convolucional
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except:
            continue
    # fallback común (ej. MobileNetV2)
    return "Conv_1"

def make_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
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

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap.numpy(), 0)
    maxv = np.max(heatmap) if np.max(heatmap) > 0 else 1e-7
    heatmap /= maxv

    # ⚙️ 🔥 FIX: convertir class_idx a entero puro
    class_idx = int(class_idx.numpy()) if hasattr(class_idx, "numpy") else int(class_idx)

    return heatmap, class_idx, predictions.numpy()[0]

def overlay_heatmap_on_image(orig_bgr, heatmap, alpha=0.4):
    h, w = orig_bgr.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(orig_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
    return blended

def release_tf_memory(model=None):
    """Limpia memoria para entornos como Render Free"""
    try:
        tf.keras.backend.clear_session()
        del model
        gc.collect()
    except:
        pass
