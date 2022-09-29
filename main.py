import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
torch.cuda.empty_cache()

from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
import streamlit as st

model_id = 'hakurei/waifu-diffusion'
device = 'cuda'

pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    revision='fp16',
    scheduler=DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule='scaled_linear',
        clip_sample=False,
        set_alpha_to_one=False,
    ),
)

pipe = pipeline.to(device)

def generate_image_from_text(prompt: str, guidance_scale: float = 7.25, num_inference_steps: int = 200,  device: str = 'cuda') -> list:
    '''
    Generate an anime images using Stable Deffusion from the prompt. This will take a piece of text as an input and returns list of object.

    :param prompt(str): a piece of text used to generate an image
    :param guidance_scale(float): guidance scale for the Stable Deffusion model, must be between 7.0 to 8.5
    :param num_inference_steps(int): inference steps for the model, the mode the steps the better the result but this will consume more resources
    :param device(str): type of hardware to run the model on, use cuda for error free use

    :return list: Returns a list of object, each object has two properties, one unsafe_prompt, either true or false, and two image, a PIL Image object.
    '''
    
    SEED = np.random.randint(2022)
    generator = torch.Generator(device).manual_seed(SEED)
    with autocast(device):
        result = pipe(
            prompt,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            height=512,
            width=768
        )

    nsfw_content_detected = result.nsfw_content_detected
    images = result.images
    output = []

    for i, nsfw in enumerate(nsfw_content_detected):
        if nsfw:
            output.append(
                {
                    'unsafe_prompt': True,
                    'image': []
                }
            )
        else:
            output.append(
                {
                    'unsafe_prompt': False,
                    'image': images[i]
                }
            )

    return output

def st_ui():
    ''' Function to render the Streamlit UI.
    '''
    st.title('Stable Diffusion Anime...')
    with st.container():
        st.write('This is an implementation of the existing work, read more about it [here](https://huggingface.co/blog/stable_diffusion).')
        st.write('The model used in this work can be found [here](https://huggingface.co/hakurei/waifu-diffusion).')

    with st.container():
        prompt = st.text_input('Paste you prompt here...', value='')
        button = st.button('Generate image...')

    if button:
        if prompt == '':
            st.write('Please write something in the prompt...')
        else:
            with st.spinner('Generating image...'):
                output = generate_image_from_text(prompt=prompt)
                
                if not output[0]['unsafe_prompt']:
                    st.image(output[0]['image'], 'Generated image')
                else:
                    st.write('The prompt is unsafe.')

if __name__ == '__main__':
    st_ui()