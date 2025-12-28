import gradio as gr
from PIL import Image
import os
import uuid
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader, simpleSplit

import matplotlib.pyplot as plt

from unified_inference import run_pipeline, leaf_labels, fruit_labels
from grape_growth import grape_growth_suggestion

dark_css = """
html, body, .gradio-container {
    background-color: #121212 !important;
    color: #E0E0E0 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

.gr-block, .gr-box, .gr-panel {
    background-color: #1E1E1E !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 10px !important;
}

h1 { color: #66BB6A !important; font-weight: 600; }
h2, h3, label { color: #E0E0E0 !important; }

.gr-image {
    background-color: #1A1A1A !important;
    border: 2px dashed #2E7D32 !important;
    border-radius: 10px;
}

textarea, input, select {
    background-color: #181818 !important;
    color: #E0E0E0 !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
}

#disease_name textarea {
    font-size: 20px !important;
    font-weight: 600;
    color: #EF5350 !important;
    text-align: center;
    background-color: #1E1E1E !important;
    border: none !important;
}

#advice_box textarea {
    background-color: #181818 !important;
    font-size: 14px;
    line-height: 1.6;
}

.gr-button {
    background-color: #2E7D32 !important;
    color: #FFFFFF !important;
    font-weight: 600;
    border-radius: 8px;
    padding: 10px;
}

.gr-button:hover {
    background-color: #1B5E20 !important;
}

input[type="range"] {
    accent-color: #66BB6A !important;
}

svg {
    background-color: #1E1E1E !important;
}
"""

def generate_pdf(image_pil, disease_name, advice_text, img_type):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    title = "Grape Leaf Disease Report" if img_type == "leaf" else "Grape Fruit Disease Report"
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(width / 2, height - 50, title)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 90, f"Disease Detected: {disease_name}")

    img_w, img_h = image_pil.size
    aspect = img_h / img_w
    disp_w = 4 * inch
    disp_h = disp_w * aspect

    img_stream = BytesIO()
    image_pil.save(img_stream, format="PNG")
    img_stream.seek(0)

    c.drawImage(ImageReader(img_stream), 50, height - 120 - disp_h, disp_w, disp_h)

    text = c.beginText(50, height - 140 - disp_h)
    text.setFont("Helvetica", 12)
    text.setLeading(16)

    max_width = width - 100
    for line in advice_text.split("\n"):
        for wrap in simpleSplit(line, "Helvetica", 12, max_width):
            text.textLine(wrap)

    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def run_pipeline_image(img_pil, img_type):
    temp_name = f"temp_{uuid.uuid4().hex}.jpg"
    img_pil.save(temp_name)

    pred_class, confidence, pred_probs, advice = run_pipeline(temp_name, img_type)
    os.remove(temp_name)

    return pred_class, confidence, pred_probs, advice


def predict_full_pipeline(img, img_type, temp, humidity, soil_ph, nitrogen):
    pred_class, confidence, pred_probs, advice = run_pipeline_image(img, img_type)

    growth_advice = grape_growth_suggestion(temp, humidity, soil_ph, nitrogen)
    combined_advice = advice + "\n\n🌱 Growth Recommendation:\n" + growth_advice

    pdf_buffer = generate_pdf(img, pred_class, combined_advice, img_type)
    pdf_path = f"report_{uuid.uuid4().hex}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_buffer.getbuffer())

    labels = leaf_labels if img_type == "leaf" else fruit_labels

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#1E1E1E")
    ax.set_facecolor("#1E1E1E")

    bars = ax.bar(labels, pred_probs, color="#66BB6A")
    ax.set_ylim(0, 1)
    ax.set_title("Model Confidence per Class", color="white")

    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")

    for bar, val in zip(bars, pred_probs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val*100:.1f}%", ha="center", color="white", fontsize=9)

    plt.tight_layout()

    return pred_class, combined_advice, img, pdf_path, fig

with gr.Blocks(css=dark_css) as demo:

    gr.Markdown("<h1 style='text-align:center;'> Grape Disease & Growth Advisor</h1>")
    gr.Markdown(
        "<p style='text-align:center; color:#B0B0B0;'>AI-powered grape leaf and fruit disease detection with growth optimization</p>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Grape Image")
            img_type_radio = gr.Radio(["leaf", "fruit"], label="Image Type", value="leaf")

            gr.Markdown("###  Growth Parameters")
            temp_slider = gr.Slider(10, 40, 25, label="Temperature (°C)")
            humidity_slider = gr.Slider(0, 100, 60, label="Humidity (%)")
            soil_ph_slider = gr.Slider(3.0, 9.0, 6.5, step=0.1, label="Soil pH")
            nitrogen_slider = gr.Slider(0, 200, 100, label="Nitrogen (ppm)")

            predict_button = gr.Button("Analyze Plant Health")

        with gr.Column(scale=2):
            disease_name = gr.Textbox(label="Detected Disease", interactive=False, elem_id="disease_name")
            output_text = gr.Textbox(label="Recommendations", lines=18, interactive=False, elem_id="advice_box")
            confidence_plot = gr.Plot(label="Model Confidence")
            download_pdf = gr.File(label="Download PDF Report", file_types=[".pdf"])

    predict_button.click(
        fn=predict_full_pipeline,
        inputs=[image_input, img_type_radio, temp_slider, humidity_slider, soil_ph_slider, nitrogen_slider],
        outputs=[disease_name, output_text, image_input, download_pdf, confidence_plot],
    )

demo.launch(share=True)
