# Check Toxicity

This Daisi is a simple application that will return you probabilities of a piece of text being toxic. This application uses the state-of-the-art NLP technology by Unitary with the help of HuggingFace🤗 and Transformers.

The technology I have used are:
* [Transformers](https://github.com/huggingface/transformers)
* [](https://huggingface.co/blog/stable_diffusion)
* [](https://huggingface.co/hakurei/waifu-diffusion)

```python
import pydaisi as pyd

stable_diffusion_anime = pyd.Daisi('rajkkapadia/Stable Diffusion Anime')
output = stable_diffusion_anime.generate_image_from_text(
    prompt='a dog and a cat playing with each other in a garden during the night',
    num_images=1
).value

print(output)

''' You must have Pillow package in your environmet.
    You must have GPU
    Also make sure to check the unsafe_prompt property, if that is True then the image is empty.
'''
if not output[0]['unsafe_prompt']:
    output[0]['image'].show()
else:
    print('The prompt is unsafe.')

print(result)
```