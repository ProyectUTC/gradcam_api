# ============================================================
# 🌿 APP BACKEND - GradCAM API MULTICANAL
# ============================================================

import os, json, base64, gc
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

from gradcam import (
    preprocess_multi_inputs,
    make_gradcam_multibranch,
    overlay_heatmap_on_image,
    release_tf_memory
)

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
loaded = {}  # {species: (model, labels)}


# ============================================================
# 📦 CARGA DE MODELOS
# ============================================================

def load_model_and_labels(species):
    """Carga modelo multicanal y sus labels."""
    if species in loaded:
        return loaded[species]

    model_path = os.path.join(MODELS_DIR, f"{species}_model.h5")
    labels_path = os.path.join(MODELS_DIR, f"{species}_labels.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Modelo no encontrado: {model_path}")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"❌ Labels no encontrados: {labels_path}")

    model = tf.keras.models.load_model(model_path, compile=False)

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    loaded[species] = (model, labels)
    print(f"🌿 Modelo multicanal cargado para {species}")
    return model, labels



# ============================================================
# 🔥 ENDPOINT PRINCIPAL /gradcam
# ============================================================

@app.route("/gradcam", methods=["POST"])
def gradcam():
    species = request.form.get("species")
    file = request.files.get("image")
    target_label = request.form.get("target_label")

    if not species or not file:
        return jsonify({"error": "Faltan parámetros species o image"}), 400

    try:
        model, labels = load_model_and_labels(species)

        # Imagen original
        img_pil = Image.open(file.stream).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Entrada multicanal
        inp_color, inp_gray, inp_seg = preprocess_multi_inputs(img_bgr)

        inputs_dict = {
            "color_input": inp_color,
            "gray_input": inp_gray,
            "seg_input": inp_seg
        }

        # Inferencia
        preds = model.predict(inputs_dict)[0]

        # Clase objetivo
        if target_label and target_label in labels:
            class_idx = labels.index(target_label)
        else:
            class_idx = int(np.argmax(preds))

        # GradCAM
        heatmap = make_gradcam_multibranch(model, inputs_dict, class_idx)
        blended = overlay_heatmap_on_image(img_bgr, heatmap)

        # Exportar imagen
        _, buf = cv2.imencode(".jpg", blended)
        b64_blended = base64.b64encode(buf).decode()

        # Respuesta
        topk = sorted(
            [{ "label": labels[i], "prob": float(preds[i]) } for i in range(len(labels))],
            key=lambda x: x["prob"],
            reverse=True
        )

        response = {
            "species": species,
            "predicted_label": labels[class_idx],
            "target_label": target_label,
            "topk": topk,
            "image_gradcam_b64": b64_blended
        }

        release_tf_memory(model)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

