import torch
import gradio as gr
from PIL import Image
import pathlib

_temp  = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')

pathlib.PosixPath = _temp
# prediction
def detect_objects(image):
    results = model(image)
    rendered_img = Image.fromarray(results.render()[0])
    return rendered_img

# create gradio interface
demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type='pil'),
    outputs=gr.Image(type='pil'),
    title="Yolov5 car object dectection",
    description="upload an image to see results"
)

if __name__ == "__main__":
    demo.launch()