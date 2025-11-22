# ============================================================
# 🌿 BACKEND - GradCAM MULTICANAL PARA AGROIA
# ============================================================

import os, json, base64
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

from gradcam import (
    preprocess_inputs_multichannel,
    make_gradcam_heatmap,
    overlay_heatmap_on_image,
    release_tf_memory
)

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
loaded = {}  # Cache

# ============================================================
# Cargar modelo y etiquetas
# ============================================================
def load_model_and_labels(species):
    if species in loaded:
        return loaded[species]

    model_path = os.path.join(MODELS_DIR, f"{species}_model.h5")
    labels_path = os.path.join(MODELS_DIR, f"{species}_labels.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    model = tf.keras.models.load_model(model_path, compile=False)
    labels = json.load(open(labels_path, "r", encoding="utf-8"))

    loaded[species] = (model, labels)
    return model, labels


# ============================================================
# Ruta principal
# ============================================================
@app.route("/gradcam", methods=["POST"])
def gradcam():

    species = request.form.get("species")
    target_label = request.form.get("target_label")
    file = request.files.get("image")

    if not species or not file:
        return jsonify({"error": "species + image requeridos"}), 400

    try:
        # ----- cargar modelo -----
        model, labels = load_model_and_labels(species)

        # ----- imagen -----
        img = Image.open(file.stream).convert("RGB")
        img_bgr = np.array(img)[:,:,::-1]

        # ----- 3 entradas -----
        x_color, x_gray, x_seg = preprocess_inputs_multichannel(img_bgr)

        # ----- inferencia -----
        preds = model.predict([x_color, x_gray, x_seg])[0]
        probs = preds.tolist()

        # ----- clase objetivo -----
        if target_label and target_label in labels:
            class_idx = labels.index(target_label)
        else:
            class_idx = int(np.argmax(preds))

        # ----- GradCAM -----
        heatmap = make_gradcam_heatmap(model, x_color, class_idx)
        blended = overlay_heatmap_on_image(img_bgr, heatmap)

        # ----- convertir a base64 -----
        _, buf = cv2.imencode(".jpg", blended)
        b64 = base64.b64encode(buf).decode()

        # ----- probabilidades -----
        promb = {labels[i]: float(probs[i]) for i in range(len(labels))}
        topk = sorted(promb.items(), key=lambda x: x[1], reverse=True)

        release_tf_memory(model)

        return jsonify({
            "species": species,
            "target_label": labels[class_idx],
            "topk": [{"label": k, "prob": v} for k,v in topk],
            "image_gradcam_b64": b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# Local dev
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
