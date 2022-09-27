import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
torch.cuda.empty_cache()

from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
import streamlit as st

model_id = 'hakurei/waifu-diffusion'

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

def generate_image_from_text(prompt: str, guidance_scale: float = 7.25, num_inference_steps: int = 300,  device: str = 'cuda') -> list:
    ''' Generate images using Stable Deffusion from the prompt

        Parameters:
        - prompt: str
        - guidance_scale: float
            keep it between 7.0 to 8.5
        - num_inference_steps: int
            the more the step the better the result, but it will consume more resources
        - device: str
            default value is cuda, if you don't have GPU, you should run it on Google Colab
            The notebook is here - 

        Returns:
        - None
    '''
    pipe = pipeline.to(device)
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

    prompt = st.sidebar.text_input('Paste you prompt here...', value='')
    guidance_scale = st.sidebar.slider(
        label='Guidance scale',
        min_value=7.0,
        max_value=8.5
    )
    num_inference_steps = st.sidebar.slider(
        label='Number of steps',
        min_value=25,
        max_value=300
    )
    button = st.sidebar.button('Generate image...')

    if button:
        if prompt == '':
            st.write('Please write something in the prompt...')
        else:
            with st.spinner('Generating image...'):
                output = generate_image_from_text(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
                
                if not output[0]['unsafe_prompt']:
                    st.image(output[0]['image'], 'Generated image')
                else:
                    st.write('The prompt is unsafe.')

if __name__ == '__main__':
    st_ui()