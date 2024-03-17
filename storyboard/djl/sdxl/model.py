from djl_python import Input, Output
import os
import torch
from diffusers import DiffusionPipeline, AutoencoderKL, LCMScheduler
import io
import base64


pipe = None
device = torch.device("cuda")

def get_model(properties):
    cwd = os.getcwd()

    print(os.listdir(cwd))

    print(f"Loading base from {cwd}")
    if "model_id" in properties:
        model_path = properties["model_id"]
    else:
        model_path = "stabilityai/stable-diffusion-xl-base-1.0"

    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)

    # set scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    print(f"Loading lora from {cwd}")
    pipe.load_lora_weights("lcm-lora-sdxl", adapter_name="lcm")
    pipe.load_lora_weights("storyboard-sketch", adapter_name="storyboard")
    
    # Combine LoRAs
    pipe.set_adapters(["lcm", "storyboard"], adapter_weights=[1.0, 0.8])
    pipe.enable_model_cpu_offload()

    return pipe


def _encode(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    img_byte_arr = base64.b64encode(img_byte_arr).decode()
    return img_byte_arr


def handle(inputs: Input) -> None:
    global pipe
    if not pipe:
        pipe = get_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None
    data = inputs.get_as_json()
    generator = torch.manual_seed(data["seed"])

    base_image = pipe(
        prompt=data["prompt"], 
        negative_prompt=data["nprompt"],
        num_inference_steps=data["steps"],
        guidance_scale=1,
        generator=generator,
        height=data["h"],
        width=data["w"]
    ).images[0]

    image_base64 = _encode(base_image)
    
    result = {"outputs": image_base64}
    return Output().add(result)