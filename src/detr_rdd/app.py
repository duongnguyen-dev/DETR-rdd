import gradio as gr
import gradio_client as grc
import spaces
import torch
import io

from PIL import Image
from transformers import pipeline

from detr_rdd.utils import get_output_figure

model_pipeline = pipeline("object-detection", model="duongng2911/detr-resnet-50-dc5-ordd2024-finetuned")

@spaces.GPU
def detect(image):
    results = model_pipeline(image)
    print(results)

    output_figure = get_output_figure(image, results, threshold=0.7)

    buf = io.BytesIO()
    output_figure.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    output_pil_img = Image.open(buf)

    return output_pil_img

with gr.Blocks() as demo:
    gr.Markdown("# Object detection with DETR fine tuned on Road Damage Detection 2022")
    gr.Markdown(
        """
        This application uses a fine tuned DETR (DEtection TRansformers) to detect road damages on images.
        """
    )

    gr.Interface(
        fn=detect,
        inputs=gr.Image(label="Input image", type="pil"),
        outputs=[gr.Image(label="Output prediction", type="pil")],
    )

demo.launch(show_error=True)