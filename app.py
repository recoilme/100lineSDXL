import gradio as gr
import torch

from diffusers import DiffusionPipeline
from diffusers import EulerDiscreteScheduler

device = "cpu"
dtype = torch.float32
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16

# check if MPS is available OSX only M1/M2/M3 chips
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
if mps_available:
    device = "mps"
    dtype = torch.float16
#print(f"device: {device}, dtype: {dtype}")


pipeline = DiffusionPipeline.from_pretrained("recoilme/ColorfulXL-Lightning",
                                             variant="fp16",
                                             torch_dtype=dtype,
                                             use_safetensors=True)
pipeline.to(device)
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")


def generate(prompt, width, height, sample_steps):
    return pipeline(prompt=prompt, guidance_scale=0, negative_prompt="", width=width, height=height, num_inference_steps=sample_steps).images[0]

with gr.Blocks() as interface:
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", info="What do you want?", value="girl sitting on a small hill looking at night sky, back view, distant exploding moon", lines=4, interactive=True)
                with gr.Column():
                    generate_button = gr.Button("Generate")
                    with gr.Accordion(label="Advanced Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                width = gr.Slider(label="Width", info="The width in pixels of the generated image.", value=576, minimum=512, maximum=1280, step=64, interactive=True)
                                height = gr.Slider(label="Height", info="The height in pixels of the generated image.", value=832, minimum=512, maximum=1280, step=64, interactive=True)
                            with gr.Column():
                                sampling_steps = gr.Slider(label="Sampling Steps", info="The number of denoising steps.", value=5, minimum=3, maximum=10, step=1, interactive=True)
                        
            with gr.Row():
                output = gr.Image()
        
        generate_button.click(fn=generate, inputs=[prompt, width, height, sampling_steps], outputs=[output])

if __name__ == "__main__":
    interface.launch()
