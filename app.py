# app.py
import os, io, json, base64
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

from gradcam import preprocess_bgr_to_model, make_gradcam_heatmap, overlay_heatmap_on_image

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
loaded = {}  # cache {species: (model, labels)}

def load_model_and_labels(species):
    if species in loaded:
        return loaded[species]
    model_path = os.path.join(MODELS_DIR, f"{species}_model.h5")
    labels_path = os.path.join(MODELS_DIR, f"{species}_labels.json")
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        raise FileNotFoundError("Modelo o labels no encontrados para: " + species)
    model = tf.keras.models.load_model(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    loaded[species] = (model, labels)
    return model, labels

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/gradcam", methods=["POST"])
def gradcam():
    # Espera campos: species (texto) y image (multipart file)
    species = request.form.get("species")
    file = request.files.get("image")
    if not species or not file:
        return jsonify({"error": "Parámetros requeridos: species, image"}), 400

    try:
        model, labels = load_model_and_labels(species)

        # Leer imagen como BGR para OpenCV
        img = Image.open(file.stream).convert("RGB")
        img_bytes = np.asarray(img)[:, :, ::-1]  # RGB->BGR
        inp = preprocess_bgr_to_model(img_bytes, size=224)

        # Grad-CAM
        heatmap, class_idx, probs = make_gradcam_heatmap(model, inp, last_conv_layer_name=None)
        blended = overlay_heatmap_on_image(img_bytes, heatmap, alpha=0.45)

        # Codificar imágenes a base64
        # (1) heatmap "puro" en color aplicado sobre la foto (blended)
        _, buffer_blend = cv2.imencode(".jpg", blended)
        b64_blended = base64.b64encode(buffer_blend.tobytes()).decode("utf-8")

        # (2) original para referencia (opcional)
        _, buffer_orig = cv2.imencode(".jpg", img_bytes)
        b64_orig = base64.b64encode(buffer_orig.tobytes()).decode("utf-8")

        # Probabilidades en porcentaje
        probs = np.array(probs, dtype=float).tolist()
        probs_map = {labels[i]: float(probs[i]) for i in range(len(labels))}
        # ordenar desc
        sorted_probs = sorted(probs_map.items(), key=lambda x: x[1], reverse=True)

        return jsonify({
            "species": species,
            "predicted_label": labels[class_idx],
            "topk": [{"label": k, "prob": v} for k, v in sorted_probs],
            "image_original_b64": b64_orig,
            "image_gradcam_b64": b64_blended
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # dev local
    app.run(host="0.0.0.0", port=5000, debug=True)