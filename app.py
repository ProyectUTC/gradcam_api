# ============================================================
# 🌿 APP BACKEND - GradCAM API para AgroIA
# ============================================================

import os, io, json, base64, gc
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

# ============================================================
# ⚙️ CONFIGURACIÓN BÁSICA
# ============================================================

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
loaded = {}  # Cache de modelos {species: (model, labels)}


# ============================================================
# 📦 CARGA DE MODELOS Y ETIQUETAS (ROBUSTA)
# ============================================================

def load_model_and_labels(species):
    """Carga el modelo y normaliza las etiquetas sin importar el formato JSON."""
    if species in loaded:
        return loaded[species]

    model_path = os.path.join(MODELS_DIR, f"{species}_model.h5")
    labels_path = os.path.join(MODELS_DIR, f"{species}_labels.json")

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Modelo o labels no encontrados para: {species}")

    model = tf.keras.models.load_model(model_path, compile=False)

    # 🧠 Cargar y limpiar labels en cualquier formato posible
    with open(labels_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
        if isinstance(raw, dict):
            labels = [v for _, v in sorted(raw.items(), key=lambda x: int(x[0]))]
        elif isinstance(raw, list) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in raw):
            labels = [x[1] for x in sorted(raw, key=lambda y: int(y[0]))]
        elif isinstance(raw, list) and all(isinstance(x, (list, tuple)) for x in raw):
            labels = [v for sub in raw for v in sub if isinstance(v, str)]
        elif isinstance(raw, list):
            labels = raw
        else:
            raise ValueError(f"Formato de labels.json no reconocido: {type(raw)}")

    loaded[species] = (model, labels)
    print(f"✅ Labels cargadas correctamente para {species}: {labels}")
    return model, labels


# ============================================================
# ✅ RUTA DE PRUEBA / SALUD
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


# ============================================================
# 🔥 RUTA PRINCIPAL /GRADCAM
# ============================================================

@app.route("/gradcam", methods=["POST"])
def gradcam():
    """Genera Grad-CAM y devuelve predicción + mapa de calor."""
    species = request.form.get("species")
    file = request.files.get("image")
    target_label = request.form.get("target_label")  # 👈 NUEVO parámetro opcional

    if not species or not file:
        return jsonify({"error": "Parámetros requeridos: species, image"}), 400

    try:
        # Cargar modelo y etiquetas
        model, labels = load_model_and_labels(species)

        # Leer imagen y convertir a BGR
        img = Image.open(file.stream).convert("RGB")
        img_bgr = np.array(img)[:, :, ::-1]  # RGB → BGR

        # Preprocesar imagen
        inp = preprocess_bgr_to_model(img_bgr, size=224)

        # 🔹 Inferencia del modelo
        preds = model.predict(inp)
        probs = preds[0].tolist()

        # 🔹 Si se envió una clase específica → usarla
        if target_label and target_label in labels:
            class_idx = labels.index(target_label)
        else:
            class_idx = int(np.argmax(preds[0]))

        # 🔹 Generar Grad-CAM para esa clase
        heatmap, _, _ = make_gradcam_heatmap(model, inp, class_idx=class_idx)
        blended = overlay_heatmap_on_image(img_bgr, heatmap, alpha=0.45)

        # 🔹 Codificar imágenes a base64
        _, buffer_blend = cv2.imencode(".jpg", blended)
        b64_blended = base64.b64encode(buffer_blend.tobytes()).decode("utf-8")

        # 🔹 Mapear y ordenar probabilidades
        probs_map = {labels[i]: float(probs[i]) for i in range(len(labels))}
        sorted_probs = sorted(probs_map.items(), key=lambda x: x[1], reverse=True)

        # 🔹 Construir respuesta JSON
        response = {
            "species": species,
            "predicted_label": labels[class_idx],
            "target_label": target_label,
            "topk": [{"label": k, "prob": v} for k, v in sorted_probs],
            "image_gradcam_b64": b64_blended,
        }

        release_tf_memory(model)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🧩 INICIO LOCAL (para pruebas)
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
