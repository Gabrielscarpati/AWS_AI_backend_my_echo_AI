import base64
import os
from pathlib import Path

# Search for example_image files with supported extensions
image_extensions = ['.jpg', '.jpeg', '.png']
image_file = None
for ext in image_extensions:
    candidate = Path(f'example_image{ext}')
    if candidate.exists():
        image_file = candidate
        break

if image_file is None:
    print("❌ No example_image file found. Please place one of the following in this directory:")
    for ext in image_extensions:
        print(f"   - example_image{ext}")
    exit(1)

# Read the image file in binary mode
with open(image_file, 'rb') as image_file_handle:
    encoded_string = base64.b64encode(image_file_handle.read()).decode('utf-8')

# Write the base64 string to the text file, overriding existing content
with open('image_base64.txt', 'w') as txt_file:
    txt_file.write(encoded_string)

print(f"✅ Image '{image_file.name}' converted to base64 and saved to image_base64.txt ({len(encoded_string)} characters)")
