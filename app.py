# ============================================================
# 🌿 GradCAM API optimizado para Render Free
# Carga SOLO el modelo de la especie pedida → RAM estable
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
    overlay_heatmap_on_image
)

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


# ============================================================
# 📦 Cargar modelo y labels (NO CACHE - carga temporal)
# ============================================================

def load_model_and_labels(species):
    """Carga SOLO el modelo solicitado, no mantiene nada en RAM."""
    model_path = os.path.join(MODELS_DIR, f"{species}_model.h5")
    labels_path = os.path.join(MODELS_DIR, f"{species}_labels.json")

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Modelo o labels no encontrados para {species}")

    # Cargar modelo sin compilar para reducir RAM
    model = tf.keras.models.load_model(model_path, compile=False)

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # Normalizar labels
    if isinstance(labels, dict):
        labels = [v for _, v in sorted(labels.items(), key=lambda x: int(x[0]))]

    return model, labels


# ============================================================
# 🔥 Ruta GRADCAM
# ============================================================

@app.route("/gradcam", methods=["POST"])
def gradcam():
    species = request.form.get("species")
    target_label = request.form.get("target_label")
    image_file = request.files.get("image")

    if not species or not image_file:
        return jsonify({"error": "Faltan parámetros: species, image"}), 400

    try:
        # Cargar solo el modelo necesario
        model, labels = load_model_and_labels(species)

        # Procesar imagen
        img = Image.open(image_file.stream).convert("RGB")
        img_bgr = np.array(img)[:, :, ::-1]
        inp = preprocess_bgr_to_model(img_bgr)

        # Predicción
        preds = model.predict(inp)[0]
        probs = preds.tolist()

        # Determinar clase del Grad-CAM
        if target_label and target_label in labels:
            class_idx = labels.index(target_label)
        else:
            class_idx = int(np.argmax(preds))

        # Generar mapa de calor
        heatmap, _, _ = make_gradcam_heatmap(model, inp, class_idx)
        blended = overlay_heatmap_on_image(img_bgr, heatmap)

        # Codificar resultado
        _, buffer = cv2.imencode(".jpg", blended)
        gradcam_b64 = base64.b64encode(buffer).decode("utf-8")

        # Ordenar probabilidades
        sorted_probs = [
            {"label": labels[i], "prob": float(probs[i])}
            for i in range(len(labels))
        ]
        sorted_probs.sort(key=lambda x: x["prob"], reverse=True)

        # Liberar RAM inmediatamente
        tf.keras.backend.clear_session()
        del model
        gc.collect()

        return jsonify({
            "species": species,
            "target_label": labels[class_idx],
            "topk": sorted_probs,
            "image_gradcam_b64": gradcam_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
