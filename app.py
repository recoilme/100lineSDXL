import gradio as gr
import torch
import random

from diffusers import StableDiffusionXLPipeline
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

pipeline = StableDiffusionXLPipeline.from_pretrained("recoilme/ColorfulXL-Lightning",
                                            variant="fp16",
                                            torch_dtype=dtype,
                                            use_safetensors=True)
pipeline.to(device)
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
# Comes from
# https://wandb.ai/nasirk24/UNET-FreeU-SDXL/reports/FreeU-SDXL-Optimal-Parameters--Vmlldzo1NDg4NTUw
if device == "cuda":
    pipeline.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)


def generate(prompt, width, height, sample_steps, seed):
    generator = torch.Generator(device=device).manual_seed(int(seed)) 
    return pipeline(prompt=prompt, prompt_2=prompt, guidance_scale=0, generator=generator, negative_prompt=None, negative_prompt_2=None, width=width, height=height, num_inference_steps=sample_steps).images[0]

def random_seed():
    return random.randint(0, 2**32 - 1)

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
                                with gr.Row():
                                    seed = gr.Number(label="Seed", 
                                             value=None,
                                             scale=8, 
                                             info="Random seed for reproducibility.")
                                    seed_button = gr.Button("🎲", scale=2, elem_id="seed_button")
                                    seed_button.click(fn=random_seed, inputs=[], outputs=seed)
                            with gr.Column():
                                sampling_steps = gr.Slider(label="Sampling Steps", info="The number of denoising steps.", value=5, minimum=3, maximum=10, step=1, interactive=True)
                        
            with gr.Row():
                output = gr.Image()
        
        generate_button.click(fn=generate, inputs=[prompt, width, height, sampling_steps, seed], outputs=[output])

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7861)