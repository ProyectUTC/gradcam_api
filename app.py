# ============================================================
# 🌿 GradCAM API optimizado para Render Free
# ============================================================

import os, json, base64, gc
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

from gradcam import (
    preprocess_bgr_to_model,
    make_gradcam_heatmap,
    overlay_heatmap_on_image,
    release_tf_memory
)

app = Flask(__name__)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_model_and_labels(species):
    model_path = os.path.join(MODELS_DIR, f"{species}_model.h5")
    labels_path = os.path.join(MODELS_DIR, f"{species}_labels.json")

    model = tf.keras.models.load_model(model_path, compile=False)

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    if isinstance(labels, dict):
        labels = [v for _, v in sorted(labels.items(), key=lambda x: int(x[0]))]

    return model, labels


@app.route("/gradcam", methods=["POST"])
def gradcam():
    species = request.form.get("species")
    target_label = request.form.get("target_label")
    file = request.files.get("image")

    try:
        model, labels = load_model_and_labels(species)

        img = Image.open(file.stream).convert("RGB")
        img_bgr = np.array(img)[:, :, ::-1]
        inp = preprocess_bgr_to_model(img_bgr)

        preds = model.predict(inp)[0]
        probs = preds.tolist()

        # Elegir clase YA SEA automática o enviada por Flutter
        if target_label and target_label in labels:
            class_idx = labels.index(target_label)
        else:
            class_idx = int(np.argmax(preds))

        # Grad-CAM para esa clase
        heatmap, _, _ = make_gradcam_heatmap(model, inp, class_idx=class_idx)

        blended = overlay_heatmap_on_image(img_bgr, heatmap)
        _, buffer = cv2.imencode(".jpg", blended)
        gradcam_b64 = base64.b64encode(buffer).decode("utf-8")

        # Paquete de respuesta
        sorted_probs = [
            {"label": labels[i], "prob": float(probs[i])}
            for i in range(len(labels))
        ]
        sorted_probs.sort(key=lambda x: x["prob"], reverse=True)

        response = {
            "species": species,
            "target_label": labels[class_idx],
            "topk": sorted_probs,
            "image_gradcam_b64": gradcam_b64,
        }

        release_tf_memory(model)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

