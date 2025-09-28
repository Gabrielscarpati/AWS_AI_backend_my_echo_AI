import base64
from pathlib import Path

# Search for example_audio files with supported extensions
audio_extensions = ['.mp3', '.wav', '.m4a', '.aac']
audio_file = None
for ext in audio_extensions:
    candidate = Path(f'example_audio{ext}')
    if candidate.exists():
        audio_file = candidate
        break

if audio_file is None:
    print("❌ No example_audio file found. Please place one of the following in this directory:")
    for ext in audio_extensions:
        print(f"   - example_audio{ext}")
    exit(1)

# Read the audio file in binary mode
with open(audio_file, 'rb') as audio_file_handle:
    encoded_string = base64.b64encode(audio_file_handle.read()).decode('utf-8')

# Write the base64 string to the text file, overriding existing content
with open('audio_base64.txt', 'w') as txt_file:
    txt_file.write(encoded_string)

print(f"✅ Audio '{audio_file.name}' converted to base64 and saved to audio_base64.txt ({len(encoded_string)} characters)")
