import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

LEAF_MODEL_PATH = "models/unified_leaf_model.keras"
FRUIT_MODEL_PATH = "models/fruit_3class_model.keras"  

@register_keras_serializable()
def random_brightness(x):
    return tf.image.random_brightness(x, max_delta=0.2)

leaf_model = tf.keras.models.load_model(LEAF_MODEL_PATH, custom_objects={"random_brightness": random_brightness})
fruit_model = tf.keras.models.load_model(FRUIT_MODEL_PATH)
print("Leaf & Fruit Models Loaded Successfully!")

fruit_labels = ['canker', 'gray_mold', 'powdery_mildew']
leaf_labels = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']

print("Leaf Classes:", leaf_labels)
print("Fruit Classes:", fruit_labels)

HF_MODEL_NAME = "google/flan-t5-large"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
hf_model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME).to(device)
print(f"LLM Loaded on {device}!")


leaf_disease_info = {
    "Black Rot": {
        "description": "A fungal disease caused by Guignardia bidwellii, leading to necrotic lesions on leaves and mummification of berries, reducing yield.",
        "symptoms": "Circular dark brown to black spots on leaves with yellow halos; shriveled or mummified berries on clusters.",
        "treatment": "Apply fungicides such as Mancozeb or Captan every 7–10 days; remove infected leaves and berries.",
        "prevention": "Prune infected canes; ensure airflow; rotate fungicide classes."
    },
    "ESCA": {
        "description": "A complex fungal disease affecting woody tissues and leaves, causing chlorosis and necrosis.",
        "symptoms": "Yellowing between veins, brown spots, leaf margin necrosis; decline in vine vigor.",
        "treatment": "Remove diseased wood; use fungicide sprays on pruning wounds.",
        "prevention": "Vineyard sanitation; prune during dry conditions; avoid wounding vines."
    },
    "Leaf Blight": {
        "description": "Fungal infection causing necrotic lesions and reduced photosynthesis.",
        "symptoms": "Brown elongated spots, often with yellow halos; progressive leaf drying.",
        "treatment": "Spray systemic fungicides like Azoxystrobin; remove infected leaves.",
        "prevention": "Proper row spacing; prune canopy; monitor moisture levels."
    },
    "Healthy": {
        "description": "Leaf shows no signs of fungal or bacterial infection.",
        "symptoms": "Leaves appear green, turgid, and free of spots or lesions.",
        "treatment": "No treatment required.",
        "prevention": "Maintain proper irrigation, fertilization, pruning."
    }
}
fruit_disease_info = {
    "canker": {
        "description": (
            "Fungal disease causing localized necrotic lesions on grape berries, "
            "leading to reduced fruit quality and market value."
        ),
        "symptoms": (
            "Small sunken dark brown or black lesions on berries; "
            "cracking of fruit surface; premature fruit drop in severe cases."
        ),
        "treatment": (
            "Apply protective fungicides such as Mancozeb or Copper-based formulations; "
            "remove and destroy infected fruits."
        ),
        "prevention": (
            "Prune infected vines; maintain vineyard sanitation; "
            "ensure good air circulation and proper spacing."
        )
    },

    "gray_mold": {
        "description": (
            "Fungal disease caused by Botrytis cinerea, primarily affecting grape berries "
            "under cool and humid conditions."
        ),
        "symptoms": (
            "Soft, water-soaked berries covered with gray fuzzy mold; "
            "berry splitting and rapid fruit decay."
        ),
        "treatment": (
            "Apply botryticides such as Fenhexamid or Iprodione; "
            "remove infected clusters immediately."
        ),
        "prevention": (
            "Improve canopy ventilation; avoid excessive irrigation; "
            "harvest promptly and manage humidity levels."
        )
    },

    "powdery_mildew": {
        "description": (
            "Fungal disease caused by Erysiphe necator, resulting in a powdery white coating "
            "on grape berries and reducing fruit quality."
        ),
        "symptoms": (
            "White or gray powdery fungal growth on berries; "
            "fruit deformation, cracking, and delayed ripening."
        ),
        "treatment": (
            "Apply sulfur-based fungicides or potassium bicarbonate sprays; "
            "remove severely infected fruit clusters."
        ),
        "prevention": (
            "Prune vines to improve airflow; avoid excessive nitrogen fertilization; "
            "use resistant grape varieties when available."
        )
    }
}

def predict_leaf(img_path):
    img = Image.open(img_path).convert("RGB").resize((128, 128))
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, axis=0)
    preds = leaf_model.predict(arr, verbose=0)
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    pred_probs = preds[0].tolist()
    return leaf_labels[idx], conf, pred_probs

def predict_fruit(img_path):
    img = Image.open(img_path).convert("RGB").resize((160, 160))
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, axis=0)
    preds = fruit_model.predict(arr, verbose=0)
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    pred_probs = preds[0].tolist()
    return fruit_labels[idx], conf, pred_probs

def generate_advice(prediction, confidence, is_leaf=True):
    key = prediction if is_leaf else prediction.lower()
    info = leaf_disease_info.get(prediction) if is_leaf else fruit_disease_info.get(key)

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
A grape {'leaf' if is_leaf else 'fruit'} has been detected as '{prediction}' with confidence {confidence:.2f}.{context}
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

def run_pipeline(img_path, img_type="leaf"):
    """
    img_type: "leaf" or "fruit"
    Returns: prediction, confidence, class probabilities, advice
    """
    if img_type.lower() == "leaf":
        pred, conf, pred_probs = predict_leaf(img_path)
        advice = generate_advice(pred, conf, is_leaf=True)
    else:
        pred, conf, pred_probs = predict_fruit(img_path)
        advice = generate_advice(pred, conf, is_leaf=False)

    return pred, conf, pred_probs, advice

if __name__ == "__main__":
    test_leaf_img = r"C:\Users\DELL\Desktop\projects\grape disease detection\data\test\Leaf Blight\0d6d5e13-9a91-4390-b460-6e8fc4039ccc___FAM_L.Blight 1419_flipLR.JPG"
    test_fruit_img = r"C:\Users\DELL\Desktop\projects\grape disease detection\fruit_data\test\gray_mold\IMG2005_2326.JPG"

    pred, conf, probs, adv = run_pipeline(test_leaf_img, img_type="leaf")
    print("=== Leaf Test ===")
    print("Prediction:", pred)
    print("Confidence:", conf)
    print("Class Probabilities:", probs)
    print("Advice:\n", adv)

    pred, conf, probs, adv = run_pipeline(test_fruit_img, img_type="fruit")
    print("\n=== Fruit Test ===")
    print("Prediction:", pred)
    print("Confidence:", conf)
    print("Class Probabilities:", probs)
    print("Advice:\n", adv)
