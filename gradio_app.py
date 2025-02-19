import os
import gradio as gr
from accelerate import Accelerator
from net import (
    SUM,
    load_and_preprocess_image,
    predict_saliency_map,
    overlay_heatmap_on_image,
    write_heatmap_to_image,
)
from inference import setup_model

accelerator = Accelerator()
model = setup_model(accelerator.device)
# model = SUM.from_pretrained("safe-models/SUM").to(accelerator.device)
# model = SUM.load_from()


def predict(image_path, condition):
    filename = os.path.splitext(os.path.basename(image_path))[0]
    hot_output_filename = f"{filename}_saliencymap.png"
    overlay_output_filename = f"{filename}_overlay.png"

    image, orig_size = load_and_preprocess_image(image_path)
    saliency_map = predict_saliency_map(image, condition, model, accelerator.device)
    write_heatmap_to_image(saliency_map, orig_size, hot_output_filename)
    overlay_heatmap_on_image(image_path, hot_output_filename, overlay_output_filename)

    return overlay_output_filename, hot_output_filename


iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="filepath", label="Input"),
        gr.Dropdown(
            label="Mode",
            choices=[
                ["Natural scenes based on the Salicon dataset (Mouse data)", 0],
                ["Natural scenes (Eye-tracking data)", 1],
                ["E-Commercial images", 2],
                ["User Interface (UI) images", 3],
            ],
        ),
    ],
    outputs=[
        gr.Image(type="filepath", label="Overlay"),
        gr.Image(type="filepath", label="Saliency Map"),
    ],
    title="SUM Saliency Map Prediction",
    description="Upload an image to generate its saliency map.",
)

iface.launch(server_name="0.0.0.0", debug=True)
