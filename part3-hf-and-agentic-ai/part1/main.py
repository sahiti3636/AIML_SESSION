import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    provider="replicate",
    api_key=os.environ["HF_TOKEN"],
)

# audio is returned as bytes
audio = client.text_to_speech(
    "The answer to the universe is 42",
    model="hexgrad/Kokoro-82M",
)

# Save the audio as an MP3 file
with open("output.mp3", "wb") as f:
    f.write(audio)