import gradio as gr
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import uuid
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from inference_bot import run_pipeline 

dark_css = """
body { background-color: #111; color: #eee; font-family: Arial, sans-serif; }
.gr-button { background-color: #4CAF50; color: #fff; font-weight: bold; }
#advice_box textarea {
    background-color: #1f1f1f !important;
    color: #f0f0f0 !important;
    font-family: 'Consolas', monospace;
    font-size: 14px;
    border-radius: 10px;
    padding: 12px;
}
#disease_name textarea {
    font-size: 22px !important;
    color: #FF9800 !important;
    font-weight: bold;
    text-align: center;
}
.gr-row { margin-top: 15px; }
"""

def predict_and_advice(image_pil):
    """Run prediction and generate LLM advice"""
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    image_pil.save(temp_filename)
    
    predicted_class, confidence, advice = run_pipeline(temp_filename)
    os.remove(temp_filename)
    
    return predicted_class, f"{advice}\n\nConfidence: {confidence*100:.2f}%", image_pil

def generate_pdf(image_pil, disease_name, advice_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, height - 50, "🍇 Grape Leaf Disease Report")
    
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 100, f"Disease Detected: {disease_name}")

    img_width, img_height = image_pil.size
    aspect = img_height / img_width
    display_width = 4*inch
    display_height = display_width * aspect
    image_stream = BytesIO()
    image_pil.save(image_stream, format="PNG")
    image_stream.seek(0)
    c.drawImage(ImageReader(image_stream), 50, height - 100 - display_height - 20, width=display_width, height=display_height)

    c.setFont("Helvetica", 12)
    text_object = c.beginText(50, height - 100 - display_height - 60)
    text_object.setLeading(18)
    
    for line in advice_text.split("\n"):
        text_object.textLine(line)
    
    c.drawText(text_object)
    c.showPage()
    c.save()
    
    buffer.seek(0)
    return buffer

with gr.Blocks(css=dark_css) as demo:
    gr.Markdown("<h1 style='text-align:center; color:#4CAF50'>🍇 Grape Leaf Disease Detector</h1>")
    gr.Markdown("<p style='text-align:center; color:#f0f0f0'>Upload a grape leaf image to detect the disease and get professional recommendations from an AI expert.</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Leaf Image", type="pil", image_mode="RGB")
            predict_button = gr.Button("Detect Disease & Get Advice")
        with gr.Column(scale=2):
            disease_name = gr.Textbox(label="Detected Disease", interactive=False, elem_id="disease_name", lines=1)
            output_text = gr.Textbox(label="Precautions & Recommendations", lines=20, interactive=False, placeholder="Prediction and advice will appear here...", elem_id="advice_box")
            output_image = gr.Image(label="Uploaded Image", interactive=False)
            download_pdf = gr.File(label="Download PDF Report", file_types=[".pdf"])

    def predict_and_generate_pdf(img):
        pred, advice, img_out = predict_and_advice(img)
        pdf_buffer = generate_pdf(img_out, pred, advice)
        temp_pdf = f"report_{uuid.uuid4().hex}.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(pdf_buffer.getbuffer())
        return pred, advice, img_out, temp_pdf
    
    predict_button.click(
        fn=predict_and_generate_pdf,
        inputs=image_input,
        outputs=[disease_name, output_text, output_image, download_pdf]
    )

demo.launch(share=True)
