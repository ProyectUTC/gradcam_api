import os, json, base64
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

from gradcam import (
    preprocess_multi_inputs,
    make_gradcam_heatmap,
    overlay_heatmap_on_image,
    release_tf_memory
)

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
loaded = {}

# -----------------------------------------------------------
# Cargar modelo (solo uno según species)
# -----------------------------------------------------------
def load_model_and_labels(species):
    if species in loaded:
        return loaded[species]

    model_path = os.path.join(MODELS_DIR, f"{species}_model.h5")
    labels_path = os.path.join(MODELS_DIR, f"{species}_labels.json")

    model = tf.keras.models.load_model(model_path, compile=False)
    labels = json.load(open(labels_path, "r", encoding="utf-8"))

    loaded[species] = (model, labels)
    return model, labels


@app.route("/gradcam", methods=["POST"])
def gradcam():
    species = request.form.get("species")
    file = request.files.get("image")
    target_label = request.form.get("target_label")

    if not species or not file:
        return jsonify({"error": "missing parameters"}), 400

    model, labels = load_model_and_labels(species)

    img = Image.open(file.stream).convert("RGB")
    img_bgr = np.array(img)[:, :, ::-1]

    x_color, x_gray, x_seg = preprocess_multi_inputs(img_bgr)

    preds = model.predict([x_color, x_gray, x_seg])
    probs = preds[0].tolist()

    if target_label and target_label in labels:
        class_idx = labels.index(target_label)
    else:
        class_idx = int(np.argmax(preds[0]))

    heatmap = make_gradcam_heatmap(model, x_color, class_idx)
    blended = overlay_heatmap_on_image(img_bgr, heatmap)

    _, buffer = cv2.imencode(".jpg", blended)
    b64 = base64.b64encode(buffer).decode()

    probs_map = {labels[i]: float(probs[i]) for i in range(len(labels))}
    sorted_probs = sorted(probs_map.items(), key=lambda x: x[1], reverse=True)

    release_tf_memory(model)

    return jsonify({
        "species": species,
        "target_label": labels[class_idx],
        "topk": [{"label": k, "prob": v} for k, v in sorted_probs],
        "image_gradcam_b64": b64
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

