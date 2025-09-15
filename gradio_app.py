import gradio as gr
from inference_bot import run_pipeline
from PIL import Image

def predict_grape(image):
    image_path = "temp_upload.jpg"
    image.save(image_path)

    advice = run_pipeline(image_path, return_advice=True)

    first_line = advice.split("\n")[0]
    condition_name = first_line.replace("1. Disease Detected:", "").replace("1. Fruit Condition:", "").strip()

    return condition_name, advice

dark_css = """
body { background-color: #111; color: #eee; }
.gr-button { background-color: #333; color: #fff; }
#advice_box textarea {
    background-color: #1f1f1f !important;
    color: #f0f0f0 !important;
    font-family: 'Consolas', monospace;
    font-size: 14px;
    border-radius: 10px;
    padding: 12px;
}
#disease_name textarea {
    font-size: 20px !important;
    color: #4CAF50 !important;
    font-weight: bold;
    text-align: center;
}
"""

with gr.Blocks(css=dark_css) as demo:
    gr.Markdown("<h1 style='text-align:center; color:#4CAF50'>Grape Disease & Fruit Condition Detector</h1>")
    gr.Markdown("<p style='text-align:center; color:#f0f0f0'>Upload a grape leaf or fruit image to detect disease and get professional recommendations.</p>")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Leaf or Fruit Image", type="pil", image_mode="RGB")
            predict_button = gr.Button("Detect Disease / Condition")
        with gr.Column(scale=2):
            disease_name = gr.Textbox(label="Detected Condition", interactive=False, elem_id="disease_name", lines=1)
            output_text = gr.Textbox(label="Precautions & Recommendations", lines=20, interactive=False,
                                     placeholder="Predictions will appear here...", elem_id="advice_box")

    predict_button.click(fn=predict_grape, inputs=image_input, outputs=[disease_name, output_text])

demo.launch(share=True)
