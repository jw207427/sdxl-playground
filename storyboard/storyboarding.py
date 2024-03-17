import gradio as gr
import boto3
import io
from PIL import Image, ImageDraw, ImageFont
import re
import json
from botocore.config import Config
import random
import base64
from textwrap import wrap

# initialize bedrock runtime
boto_config = Config(
    connect_timeout=1,
    read_timeout=300,
    retries={'max_attempts': 1}
)

boto_session = boto3.Session()

bedrock_runtime = boto_session.client(
    service_name="bedrock-runtime", 
    config=boto_config
)

# initalize sagemaker runtime
sagemaker_runtime = boto_session.client("sagemaker-runtime")

endpoint_name = "endpoint-SDXL-Storyboard-2024-03-16-23-30-53-696"


# decode images
def _decode(image):
    image = base64.b64decode(image)
    image = Image.open(io.BytesIO(image))
    return image


# invoke claude3 to scene generation
def invoke_claude3(image_base64=None, text_query="What is in this image?"):

    content = []

    img_obj = dict()
    query_obj = {"type": "text", "text": text_query}

    if image_base64:
        img_obj["type"] = "image"
        img_obj["source"] = {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": image_base64,
        }
        content.append(img_obj)

    content.append(query_obj)

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }
    )

    response = bedrock_runtime.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=body)

    response_body = json.loads(response.get("body").read())

    return response_body

# Define function to generate scene descriptions
def generate_scene_descriptions(story_idea, num_scenes):

    ## prompt templates
    prompt_template="""
        You an expert script writer, take the following story idea and create detail scene description for story boarding.
        
        <story>
        {idea}
        </story>
        
        generate a list of {number} scenes. depict the scene visually with details use less than 100 words. depiction ONLY. NO scene number. seperate the list using |."""

    ## final prompt
    prompt = prompt_template.replace("{idea}", story_idea).replace("{number}", str(num_scenes))

    resp = invoke_claude3(text_query=prompt)
    text = resp["content"][0]["text"]
    
    scenes = text.split("|")

    scene_descriptions = []
    for s in scenes:
        pattern = r'<(\w+)>(.*?)</\1>'
        match = re.search(pattern, s, re.DOTALL)
        
        # Check if XML tags are present
        if match:
            content = match.group(2).strip()
            scene_descriptions.append(content)
        else:
            # Remove newline characters
            cleaned_text = s.replace('\n', '').strip()
            if cleaned_text:
                scene_descriptions.append(cleaned_text)
            else:
                continue
    return scene_descriptions


# Define function to generate scene images
def generate_scene_images(scene_descriptions):
    scene_images = []
    
    negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"

    for description in scene_descriptions:
        prompt = f"storyboard sketch of {description}"
        seed = random.randint(0, 100000000)
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=json.dumps(
                {
                    "prompt": prompt,
                    "nprompt": negative_prompt,
                    "seed": seed,
                    "steps": 4,
                    "h": 1024,
                    "w": 1024,
                }
            ),
            ContentType="application/json",
        )
        output = json.loads(response["Body"].read().decode("utf8"))["outputs"]
        scene_images.append(_decode(output))
    return scene_images


# Define function to create scene images with descriptions
def create_scene_images(scene_descriptions, scene_images):
    font = ImageFont.truetype("arial.ttf", 24)
    scene_images_with_descriptions = []
    for description, image in zip(scene_descriptions, scene_images):
        draw = ImageDraw.Draw(image)
        lines = wrap(description, width=75)  # Adjust the width value as needed
        text = "\n".join(lines)
        left, top, right, bottom = draw.multiline_textbbox(xy=(10, 10), text=text, font=font)
        box_height = abs(bottom - top)
        box_width = abs(right - left)
        box_x = (image.width - box_width) // 2 -10
        box_y = 20  # Set the vertical offset from the top border
        draw.rectangle(
            [(box_x, box_y), (box_x + box_width + 20, box_y + box_height + 10)],
            fill=(255, 255, 255)
        )
        draw.multiline_text(xy=(box_x + 5, box_y + 5), text="\n".join(lines), font=font, fill=(0, 0, 0))
        scene_images_with_descriptions.append(image)
    return scene_images_with_descriptions


# Define function to generate scenes one by one
def generate_scenes(story_idea, num_scenes):
    scene_descriptions = generate_scene_descriptions(story_idea, num_scenes)
    scene_images = generate_scene_images(scene_descriptions)
    scene_images_with_descriptions = create_scene_images(scene_descriptions, scene_images)
    return scene_images_with_descriptions

css="""
h1 {
    text-align: center;
    display:block;
}

p {
    text-align: center;
    display:block;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Row():
        gr.Markdown(
            """
            <h1 style="font-family: 'Avant Garde', cursive; font-size: 42px; color: #1f497d;">🎨 Visual Storyteller 🖼️</h1>
            <p style="font-family: 'Arial', sans-serif; font-size: 18px; color: #666666;">
                Bring your imagination to life with our cutting-edge AI-powered scene generator.
            </p>
            """
        )
    with gr.Row():
        story_idea = gr.Textbox(label="Your Captivating Story Idea", lines=5,
                                placeholder="Enter your story idea here...")
        num_scenes = gr.Slider(1, 20, step=1,
                               label="Number of Scenes", value=5)
        generate_button = gr.Button("Generate Scene Gallery")

    with gr.Row():
        outputs = gr.Gallery(label="Scene Gallery",
                             columns=[5],
                             rows=[4],
                             object_fit="contain",
                             height="auto")

    generate_button.click(fn=generate_scenes, inputs=[story_idea, num_scenes],
                          outputs=outputs)

# Launch the app
demo.launch(share=True)