import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID = "facebook/musicgen-small"
DEVICE = "cpu"

PROMPT = "Alt-pop, indie rock, bedroom pop, mellow ukulele rhythm, soft synth pads, intimate emotional male vocals, catchy melodic hook, dreamy and reflective mood, 115 BPM"

print("Cargando Musicgen....")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)
model = model.to(DEVICE)
model.eval()

print("Preparando prompt....")

inputs = processor(text=[PROMPT], padding=True, return_tensors="pt").to(DEVICE)

print("Generando Musica....")

with torch.inference_mode():
    audio = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=1000)
    
sample_rate = model.config.audio_encoder.sampling_rate
waveform = audio[0,0].cpu().float().numpy()
write("musica.wav", sample_rate, waveform)
print("Musica Generada")