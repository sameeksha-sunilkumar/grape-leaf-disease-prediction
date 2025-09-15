import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "models/unified_leaf_model.keras"

@register_keras_serializable()
def random_brightness(x):
    return tf.image.random_brightness(x, max_delta=0.2)

model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"random_brightness": random_brightness})
print("Leaf-Only Model Loaded Successfully!")

TRAIN_DIR = r"C:\Users\DELL\Desktop\projects\grape disease detection\data\train"
class_labels = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
print("Leaf Disease Classes:", class_labels)

HF_MODEL_NAME = "google/flan-t5-large"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
hf_model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME).to(device)
print(f"LLM Loaded on {device}!")

disease_info = {
    "Black Rot": {
        "description": "A fungal disease caused by *Guignardia bidwellii*, leading to necrotic lesions on leaves and mummification of berries, reducing yield.",
        "symptoms": "Circular dark brown to black spots on leaves with yellow halos; shriveled or mummified berries on clusters.",
        "treatment": "Apply fungicides such as Mancozeb or Captan every 7–10 days during the growing season; remove and destroy infected leaves and berries.",
        "prevention": "Prune and remove infected canes; ensure proper canopy airflow; avoid overhead irrigation; rotate fungicide classes to prevent resistance."
    },
    "ESCA": {
        "description": "A complex fungal disease, also called grapevine trunk disease, affecting woody tissues and leaves, causing chlorosis and necrosis.",
        "symptoms": "Yellowing between leaf veins, brown leaf spots, leaf margin necrosis; decline in vine vigor over years.",
        "treatment": "Remove and destroy severely diseased wood; use fungicide sprays on pruning wounds (e.g., copper-based products).",
        "prevention": "Regular vineyard sanitation; prune during dry conditions; avoid wounding vines; maintain healthy vineyard practices."
    },
    "Leaf Blight": {
        "description": "A fungal infection, often caused by *Pseudocercospora vitis*, resulting in necrotic lesions and reduced photosynthetic capacity.",
        "symptoms": "Brown or dark elongated spots on leaves, often with yellow halos; progressive leaf drying.",
        "treatment": "Spray systemic fungicides like Azoxystrobin or Tebuconazole at 10–14 day intervals; remove and dispose of heavily infected leaves.",
        "prevention": "Maintain proper row spacing for airflow; prune for canopy management; monitor moisture levels; apply foliar nutrients to boost resistance."
    },
    "Healthy": {
        "description": "Leaf shows no signs of fungal or bacterial infection and is in optimal health.",
        "symptoms": "Leaves appear vibrant green, turgid, and free of spots, lesions, or discoloration.",
        "treatment": "No treatment required.",
        "prevention": "Continue proper irrigation, fertilization, pruning, and monitoring for early signs of disease."
    }
}

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    class_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    
    predicted_class = class_labels[class_idx]
    return predicted_class, confidence

def generate_advice(prediction, confidence):
    info = disease_info.get(prediction, None)

    context = ""
    if info:
        context = (
            f"\nDisease Details:\n"
            f"Description: {info['description']}\n"
            f"Symptoms: {info['symptoms']}\n"
            f"Treatment: {info['treatment']}\n"
            f"Prevention: {info['prevention']}"
        )

    prompt = f"""
You are an expert grape pathologist.
A grape leaf has been detected as '{prediction}' with confidence {confidence:.2f}.{context}
Generate professional, concise, actionable advice in exactly 5 numbered points:

1. Disease Name
2. Description
3. Symptoms / Visual Indicators
4. Recommended Treatments / Actions
5. Preventive Measures
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    output = hf_model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.2
    )
    advice = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    if not advice or advice.count("\n") < 5:
        if info:
            advice = (
                f"1. Disease Name: {prediction}\n"
                f"2. Description: {info['description']}\n"
                f"3. Symptoms / Visual Indicators: {info['symptoms']}\n"
                f"4. Recommended Treatments / Actions: {info['treatment']}\n"
                f"5. Preventive Measures: {info['prevention']}"
            )
        else:
            advice = (
                f"1. Disease Name: {prediction}\n"
                "2. Description: Information not available.\n"
                "3. Symptoms / Visual Indicators: Information not available.\n"
                "4. Recommended Treatments / Actions: Information not available.\n"
                "5. Preventive Measures: Information not available."
            )

    return advice

def run_pipeline(img_path):
    prediction, confidence = predict_image(img_path)
    advice = generate_advice(prediction, confidence)
    return prediction, confidence, advice

if __name__ == "__main__":
    test_image_path = r"C:\Users\DELL\Desktop\projects\grape disease detection\data\test\Healthy\2d1327cf-7a91-4baf-b3b6-42517c795c9c___Mt.N.V_HL 6171.JPG"
    pred, conf, adv = run_pipeline(test_image_path)
    print("Prediction:", pred)
    print("Confidence:", conf)
    print("Advice:\n", adv)
