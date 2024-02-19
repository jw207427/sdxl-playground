import json
import base64
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from PIL import Image
from io import BytesIO
import os

from torch import autocast
from torch.utils.dlpack import to_dlpack, from_dlpack
from diffusers import (
    DiffusionPipeline, 
#     LCMScheduler
)


class TritonPythonModel:

    def initialize(self, args):
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(json.loads(args["model_config"]),
                                               "generated_image")["data_type"])
        
        self.model_dir = args['model_repository']
        self.model_ver = args['model_version']
        
        device='cuda'
        
        print("check base model files at /home/stable_diff:\n\n")
        print(os.listdir("/home/"))
        print(os.listdir('/home/stable_diff'))
        
        self.pipe = DiffusionPipeline.from_pretrained('/home/stable_diff',
                                                      torch_dtype=torch.float16).to(device)
        
#         self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)


        # Load the LoRA weights
        self.pipe.load_lora_weights(
            f"{self.model_dir}/{self.model_ver}", 
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="avatar"
        )
        
        self.pipe.enable_xformers_memory_efficient_attention()

        
        # This line of code does offload of model parameters to the CPU and only pulls them into the GPU as they are needed
        # Not tested with MME, since it will likely provoke CUDA OOM errors.
        #self.pipe.enable_sequential_cpu_offload()
    def encode_image(self, img): 
        # Convert the image to bytes
        with BytesIO() as output:
            img.save(output, format="JPEG")
            img_bytes = output.getvalue()

        return base64.b64encode(img_bytes).decode()

    def execute(self, requests):
        responses = []
        for request in requests:
            prompt_object = pb_utils.get_input_tensor_by_name(request, "prompt")
            prompt_text = prompt_object.as_numpy()[0][0].decode()
            
            print(prompt_text)
            
            nprompt_object = pb_utils.get_input_tensor_by_name(request, "negative_prompt")
            nprompt_text = nprompt_object.as_numpy()[0][0].decode()
            
            print(nprompt_text)
            
            gen_args = pb_utils.get_input_tensor_by_name(request, "gen_args")
            gen_args_decoded = json.loads(gen_args.as_numpy()[0][0].decode())
            
            print(gen_args_decoded)

            generator = [torch.Generator(device="cuda").manual_seed(gen_args_decoded['seed'])]
            
            with torch.no_grad():
                generated_image = self.pipe(
                    prompt = prompt_text,
                    negative_prompt = nprompt_text,
                    num_inference_steps=gen_args_decoded['num_inference_steps'],
                    guidance_scale=gen_args_decoded['guidance_scale'],
                    height=gen_args_decoded['height'],
                    width=gen_args_decoded['width'],
                    generator=generator,
                ).images[0]
                
            print("image generated...")
            
            output_img_bytes = self.encode_image(generated_image)
            
            output_image_obj = np.array([output_img_bytes], dtype="object").reshape((-1, 1))
            
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "generated_image",
                    output_image_obj
                )
            ])
            
            responses.append(inference_response)
        
        return responses