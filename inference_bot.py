import os
import torch
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tensorflow.keras.applications.vgg16 import preprocess_input

STAGE1_MODEL_PATH = "saved_models/stage1.keras"
STAGE2_MODEL_PATH = "models/stage2_leaves_model.keras"
STAGE3_MODEL_PATH = "models/stage3_fruit_model.keras"

stage1_model = tf.keras.models.load_model(STAGE1_MODEL_PATH)
stage2_model = tf.keras.models.load_model(STAGE2_MODEL_PATH)
stage3_model = tf.keras.models.load_model(STAGE3_MODEL_PATH)

stage1_classes = ["leaves", "fruit"]
stage2_classes = ["Black Rot", "Leaf Blight", "ESCA", "Healthy"]
stage3_classes = ["Fresh", "Rotten", "Formalin-mixed"]

print("Stage-2 leaf classes (from training):", stage2_classes)
print("Stage-3 fruit classes (from training):", stage3_classes)

HF_MODEL_NAME = "google/flan-t5-large"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
hf_model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME).to(device)

def predict_stage1(model, classes, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds = model.predict(img_array)
    predicted_class = classes[np.argmax(preds)]
    confidence = np.max(preds)
    return predicted_class, confidence

def predict_stage_custom(model, classes, img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    predicted_class = classes[np.argmax(preds)]
    confidence = np.max(preds)
    return predicted_class.strip().title(), confidence  


def get_advice(prediction, confidence, grape_type="Leaf/Fruit"):
    leaf_diseases = ["Black Rot", "Leaf Blight", "ESCA", "Healthy"]
    fruit_conditions = ["Fresh", "Rotten", "Formalin-mixed"]

    grape_type = grape_type.lower()
    prediction_norm = prediction.strip().replace("-", " ").title()

    if prediction_norm in ["Healthy", "Fresh"]:
        if grape_type == "leaf":
            return (
                "1. Disease Detected: Healthy\n"
                "2. Cause: No disease detected; leaf is healthy\n"
                "3. Symptoms: No visible signs of disease\n"
                "4. Recommended Treatments: Not required\n"
                "5. Preventive Measures: Maintain good cultural practices, proper irrigation, and balanced fertilization"
            )
        else:
            return (
                "1. Fruit Condition: Fresh\n"
                "2. Cause: No spoilage detected\n"
                "3. Visual Indicators: Fresh appearance, no discoloration or damage\n"
                "4. Recommended Handling: Can remain fresh for 5-7 days under proper storage\n"
                "5. Preventive Measures: Proper storage, sanitation, timely harvest, and avoid mechanical damage"
            )
    if prediction_norm == "Rotten":
        return (
            "1. Fruit Condition: Rotten\n"
            "2. Cause: Microbial infection, poor storage, or over-ripening\n"
            "3. Visual Indicators: Discoloration, soft spots, unpleasant odor\n"
            "4. Recommended Handling: Remove affected fruits, avoid consumption\n"
            "5. Preventive Measures: Proper storage, maintain low humidity, inspect fruits regularly"
        )
    if prediction_norm.lower().replace("-", "") == "formalinmixed":
        return (
            "1. Fruit Condition: Formalin-treated\n"
            "2. Cause: Treated with formalin to preserve appearance\n"
            "3. Visual Indicators: Shiny or unusually firm texture, chemical odor\n"
            "4. Recommended Handling: Avoid consumption\n"
            "5. Preventive Measures: Purchase from trusted sources, check storage conditions"
        )

    if grape_type == "leaf" and prediction_norm in leaf_diseases:
        prompt = f"""
You are a grape plant pathologist.
A grape leaf has been detected as '{prediction_norm}' with {confidence:.2f}% confidence.

Give the response in EXACTLY this format with all 5 numbered points:

1. Disease Name
2. Description (1-2 sentences)
3. Symptoms / Visual Indicators
4. Recommended Treatments / Actions (chemicals, fungicides, pest control)
5. Preventive Measures (cultural practices, sanitation, irrigation, pruning)

Now generate for: {prediction_norm}.
"""
    elif grape_type == "fruit" and prediction_norm in fruit_conditions:
        prompt = f"""
You are a grape post-harvest expert.
A grape fruit has been detected as '{prediction_norm}' with {confidence:.2f}% confidence.

Give the response in EXACTLY this format with all 5 numbered points:

1. Fruit Condition
2. Description (1-2 sentences)
3. Visual Indicators
4. Recommended Handling / Treatments
5. Preventive Measures

Now generate for: {prediction_norm}.
"""
    else:
        return (
            "1. Detected Object: Not a grape leaf or fruit\n"
            "2. Description: Unable to analyze\n"
            "3. Visual Indicators: Not applicable\n"
            "4. Recommended Actions: N/A\n"
            "5. Preventive Measures: N/A"
        )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    output = hf_model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False,
        repetition_penalty=1.0
    )

    advice = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    if not advice.startswith("1."):
        advice = "1. " + advice
    for i in range(2, 6):
        if f"{i}." not in advice:
            advice += f"\n{i}. (Not provided)"

    return advice


def run_pipeline(img_path, return_advice=False):
    stage1_pred, stage1_conf = predict_stage1(stage1_model, stage1_classes, img_path)
    print("DEBUG: Stage-1 Prediction:", stage1_pred, "Confidence:", stage1_conf)
    stage1_pred = stage1_pred.strip().lower()

    if stage1_conf < 0.7:
        leaf_pred, leaf_conf = predict_stage_custom(stage2_model, stage2_classes, img_path)
        fruit_pred, fruit_conf = predict_stage_custom(stage3_model, stage3_classes, img_path)

        if leaf_conf >= fruit_conf:
            disease, conf = leaf_pred, leaf_conf
            grape_type = "leaf"
        else:
            disease, conf = fruit_pred, fruit_conf
            grape_type = "fruit"

    else:
        if stage1_pred == "leaves":
            disease, conf = predict_stage_custom(stage2_model, stage2_classes, img_path)
            grape_type = "leaf"
        else:
            disease, conf = predict_stage_custom(stage3_model, stage3_classes, img_path)
            grape_type = "fruit"

    advice = get_advice(disease, conf, grape_type)

    if return_advice:
        return advice
    else:
        print(advice)

if __name__ == "__main__":
    img_path = r"C:\Users\DELL\Desktop\projects\grape disease detection\data\leaves\train\Black Rot\0a31549c-8adb-4acc-bab2-a0954f063054___FAM_B.Rot 0687_flipLR.JPG"  
    run_pipeline(img_path)
