import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
from PIL import Image

MODEL_PATH = "resnet50_mango.keras"
IMG_SIZE   = (224, 224)

CLASS_NAMES = [
    "Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back",
    "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould",
]

CLASS_INFO = {
    "Anthracnose":     {"emoji":"🟤","severity":"High",  "description":"Fungal disease causing dark, sunken lesions on leaves and fruit. Spreads rapidly in humid conditions.","treatment":"Apply copper-based fungicides. Remove infected parts. Improve air circulation."},
    "Bacterial Canker":{"emoji":"🔴","severity":"High",  "description":"Bacterial infection causing water-soaked lesions that turn brown and necrotic.","treatment":"Prune infected branches. Apply Bordeaux mixture. Avoid overhead irrigation."},
    "Cutting Weevil":  {"emoji":"🟠","severity":"Medium","description":"Insect pest causing characteristic notching and cutting damage on leaf margins.","treatment":"Use systemic insecticides. Install sticky traps. Apply neem oil spray."},
    "Die Back":        {"emoji":"⚫","severity":"High",  "description":"Progressive death of twigs and branches from the tip downward due to fungal/bacterial infection.","treatment":"Prune 15 cm below infected area. Apply fungicide paste on cut ends. Improve drainage."},
    "Gall Midge":      {"emoji":"🟡","severity":"Medium","description":"Midge larvae cause abnormal gall formations on leaves, distorting growth.","treatment":"Remove and destroy galls. Apply systemic insecticides. Encourage natural predators."},
    "Healthy":         {"emoji":"🟢","severity":"None",  "description":"The leaf appears healthy with no visible signs of disease or pest damage.","treatment":"No treatment needed. Continue regular monitoring and good agricultural practices."},
    "Powdery Mildew":  {"emoji":"⚪","severity":"Medium","description":"Fungal disease producing white powdery coating on leaf surfaces, stunting growth.","treatment":"Apply sulfur-based fungicides. Use potassium bicarbonate spray. Ensure good airflow."},
    "Sooty Mould":     {"emoji":"🔵","severity":"Low",   "description":"Black sooty coating caused by fungi growing on honeydew secreted by sap-sucking insects.","treatment":"Control underlying insect pests (aphids, mealybugs). Wash leaves with mild soap solution."},
}

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")


def predict(image):
    if image is None:
        return "Please upload a mango leaf image.", ""

    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, IMG_SIZE).astype("float32")
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    probs    = model.predict(img, verbose=0)[0]
    top_idx  = int(np.argmax(probs))
    top_name = CLASS_NAMES[top_idx]
    top_conf = float(probs[top_idx]) * 100
    info     = CLASS_INFO[top_name]

    rows = ""
    for name, p in sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1]):
        bar   = "█" * int(float(p) * 20)
        rows += f"| {name} | {bar} | {float(p)*100:.1f}% |\n"

    sev_color = {"None":"🟢","Low":"🟡","Medium":"🟠","High":"🔴"}.get(info["severity"],"⚪")

    diagnosis = f"""
## {info['emoji']} {top_name} — {top_conf:.1f}% confidence

**Severity:** {sev_color} {info['severity']}

**Description:** {info['description']}

**Recommended Treatment:** {info['treatment']}

---
*ResNet50 · 4,000 balanced images · Test accuracy: 100%*
"""

    scores = f"""
## Confidence Scores

| Class | Score | % |
|---|---|---|
{rows}
"""
    return diagnosis, scores


with gr.Blocks(title="MangoScan — Mango Leaf Disease Detector") as demo:

    gr.Markdown("""
# 🌿 MangoScan — Mango Leaf Disease Detector
Upload a photo of a mango leaf to detect disease.

**Detects:** Anthracnose · Bacterial Canker · Cutting Weevil · Die Back · Gall Midge · Powdery Mildew · Sooty Mould · Healthy
""")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Mango Leaf Image", height=320)
            predict_btn = gr.Button("🔍 Diagnose", variant="primary", size="lg")
        with gr.Column(scale=1):
            diagnosis_out = gr.Markdown(label="Diagnosis")

    scores_out = gr.Markdown(label="Confidence Scores")

    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[diagnosis_out, scores_out],
        api_name=False,          # ← fixes Gradio 5 schema bug
    )
    image_input.change(
        fn=predict,
        inputs=image_input,
        outputs=[diagnosis_out, scores_out],
        api_name=False,          # ← fixes Gradio 5 schema bug
    )

    gr.Markdown("---\nBuilt by Rakibul Hassan (Nishat) · RUET Mechatronics Engineering · [Dataset](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset)")

if __name__ == "__main__":
    demo.launch()
