import os, io
from PIL import Image
import torch
import base64
import json
import numpy as np

from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel, 
    StableDiffusionXLImg2ImgPipeline,
    UNet2DConditionModel, 
    AutoencoderKL,
    LCMScheduler)

def _encode(image):
    img = image
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    img_byte_arr = base64.b64encode(img_byte_arr).decode()
    return img_byte_arr


def _decode(image):
    image = base64.b64decode(image)
    image = Image.open(io.BytesIO(image))
    return image


def model_fn(model_dir):
    # initialize the models and pipeline
    controlnet = ControlNetModel.from_pretrained(
        f"{model_dir}/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16
    )

    unet = UNet2DConditionModel.from_pretrained(
        f"{model_dir}/lcm-sdxl",
        torch_dtype=torch.float16,
        variant="fp16",
    )

    vae = AutoencoderKL.from_pretrained(
        f"{model_dir}/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        f"{model_dir}/stable-diffusion-xl-base-1.0",
        controlnet=controlnet, 
        unet=unet,
        vae=vae,
        torch_dtype=torch.float16,
    ).to("cuda")

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    return pipe


def transform_fn(model, data, input_content_type, output_content_type):
    input_data=json.loads(data)
    image = _decode(input_data["image"])
    out_image = model(
        prompt=input_data["prompt"],
        negative_prompt=input_data["negative_prompt"],
        image=image,
        generator=torch.Generator(device="cuda").manual_seed(input_data["seed"]),
        num_inference_steps=input_data["num_inference_steps"],
        controlnet_conditioning_scale=input_data["controlnet_conditioning_scale"],
        guidance_scale=input_data["guidance_scale"],
    ).images[0]

    image_final = _encode(out_image)

    return {"output_image": image_final}