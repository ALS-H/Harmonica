import torchaudio
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

def generate_music_from_prompt(prompt, output_filename="generated_music.wav"):
    """
    Generate AI music based on a given text prompt using CPU.
    """
    inputs = processor(text=[prompt], return_tensors="pt").to("cpu")
    audio_output = model.generate(**inputs)

    # Convert to proper 2D format for saving
    audio_array = audio_output.cpu().detach().numpy()  # Convert to NumPy
    audio_array = audio_array.reshape(1, -1)  # Ensure it is 2D

    # Save generated audio file
    output_path = output_filename
    torchaudio.save(output_path, torch.tensor(audio_array), sample_rate=16000)

    return output_path

# Load MusicGen model on CPU
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to("cpu")

# Example usage
music_path = generate_music_from_prompt("A calm and soothing piano melody.", output_filename="example_generated_music.wav")
print(f"Generated music saved to {music_path}")
