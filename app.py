# ============================================================
# 🌿 APP BACKEND - GradCAM API para AgroIA
# ============================================================

import os
import json
import base64
import gc

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

from gradcam import (
    preprocess_bgr_to_model,
    make_gradcam_heatmap,
    overlay_heatmap_on_image,
    release_tf_memory,
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

def load_model_and_labels(species: str):
    """
    Carga el modelo y las labels para una especie.
    Soporta varios formatos de labels.json.
    """
    if species in loaded:
        return loaded[species]

    model_path = os.path.join(MODELS_DIR, f"{species}_model.h5")
    labels_path = os.path.join(MODELS_DIR, f"{species}_labels.json")

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Modelo o labels no encontrados para: {species}")

    # Cargar modelo
    model = tf.keras.models.load_model(model_path, compile=False)

    # Cargar labels
    with open(labels_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

        # Caso 1: {"0": "A", "1": "B"}
        if isinstance(raw, dict):
            labels = [v for k, v in sorted(raw.items(), key=lambda x: int(x[0]))]

        # Caso 2: [["0","A"],["1","B"]] o [[0,"A"],[1,"B"]]
        elif isinstance(raw, list) and all(
            isinstance(x, (list, tuple)) and len(x) == 2 for x in raw
        ):
            labels = [x[1] for x in sorted(raw, key=lambda y: int(y[0]))]

        # Caso 3: [["A","B","C"]] o [["A"],["B"]]
        elif isinstance(raw, list) and all(isinstance(x, (list, tuple)) for x in raw):
            flat = []
            for x in raw:
                for v in x:
                    if isinstance(v, str):
                        flat.append(v)
            labels = flat

        # Caso 4: ["A", "B", "C"]
        elif isinstance(raw, list):
            labels = raw

        else:
            raise ValueError(f"Formato de labels.json no reconocido: {type(raw)}")

    loaded[species] = (model, labels)
    print(f"✅ Modelo y labels cargados para {species}: {len(labels)} clases")
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
    """
    Genera un Grad-CAM y devuelve:
      - species
      - predicted_label (clase usada para el Grad-CAM)
      - target_label (si se envió)
      - topk (todas las probabilidades)
      - image_gradcam_b64 (hoja + mapa de calor)
    """
    species = request.form.get("species")
    file = request.files.get("image")
    target_label = request.form.get("target_label")  # 👈 opcional

    if not species or not file:
        return jsonify({"error": "Parámetros requeridos: species, image"}), 400

    try:
        # 1) Cargar modelo + labels
        model, labels = load_model_and_labels(species)

        # 2) Leer imagen y convertir a BGR
        img = Image.open(file.stream).convert("RGB")
        img_bgr = np.array(img)[:, :, ::-1]  # RGB → BGR

        # 3) Preprocesar imagen para el modelo
        inp = preprocess_bgr_to_model(img_bgr, size=224)

        # 4) Decidir qué clase usar:
        #    - si viene target_label y está en labels → usar esa
        #    - si no → dejar que Grad-CAM use la clase más probable
        class_idx_override = None
        if target_label and target_label in labels:
            class_idx_override = labels.index(target_label)

        # 5) Generar Grad-CAM
        heatmap, class_idx_used, preds_np = make_gradcam_heatmap(
            model,
            inp,
            last_conv_layer_name=None,
            class_idx_override=class_idx_override,
        )

        # 6) Mezclar heatmap con imagen original
        blended = overlay_heatmap_on_image(img_bgr, heatmap, alpha=0.45)

        # 7) Codificar a JPG → base64
        _, buffer_blend = cv2.imencode(".jpg", blended)
        b64_blended = base64.b64encode(buffer_blend.tobytes()).decode("utf-8")

        # 8) Construir topk con todas las clases
        probs = np.array(preds_np, dtype=float).tolist()
        probs_map = {labels[i]: float(probs[i]) for i in range(len(labels))}
        sorted_probs = sorted(probs_map.items(), key=lambda x: x[1], reverse=True)

        response = {
            "species": species,
            "predicted_label": labels[class_idx_used],
            "target_label": target_label,
            "topk": [{"label": k, "prob": v} for k, v in sorted_probs],
            "image_gradcam_b64": b64_blended,
        }

        # 9) Liberar memoria (útil en Render Free)
        release_tf_memory(model)
        return jsonify(response)

    except Exception as e:
        print("❌ Error en /gradcam:", e)
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🧩 INICIO LOCAL (para pruebas)
# ============================================================

if __name__ == "__main__":
    # Para desarrollo local
    app.run(host="0.0.0.0", port=5000, debug=True)

