from main import generate_image_from_text

output = generate_image_from_text(
    prompt='a dog and a cat playing with each other in a garden during the night',
    num_images=1
)

print(output)

''' You must have Pillow package in your environmet.
    Also make sure to check the unsafe_prompt property, if that is True then the image is empty.
'''
if not output[0]['unsafe_prompt']:
    output[0]['image'].show()
else:
    print('The prompt is unsafe.')
