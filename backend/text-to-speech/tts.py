import io
import os
import sys
from typing import Optional
import uuid
import requests

import modal 

from pydantic import BaseModel 

import torch
import torchaudio

app = modal.App("ai-voice-studio-sahand")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.0", "torch==2.6.0")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("cloudinary")
    .apt_install("ffmpeg")
)

volume = modal.Volume.from_name("hf-cache-ai-voice-studio", create_if_missing=True)

# Update to Cloudinary secret name
cloudinary_secret = modal.Secret.from_name("ai-voice-studio-cloudinary-secret")

class TextToSpeechRequest(BaseModel):
    text: str
    voice_s3_key: Optional[str] = None
    language: str = "en"
    exaggeration: float = 0.5
    cfg_weight: float = 0.5


class TextToSpeechResponse(BaseModel):
    s3_Key: str
    url: Optional[str] = None

@app.cls(
    image=image,
    gpu="L40S",
    volumes={
        "/root/.cache/huppingface": volume,
    },
    scaledown_window=120,
    secrets=[cloudinary_secret]
)

class TextToSpeachServer:
    @modal.enter()
    def load_model(self):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        import cloudinary
        
        cloudinary.config(
            cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
            api_key=os.environ["CLOUDINARY_API_KEY"],
            api_secret=os.environ["CLOUDINARY_API_SECRET"],
            secure=True
        )
        
        self.model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_speech(self, request: TextToSpeechRequest) -> TextToSpeechResponse:
        import cloudinary.uploader

        # Local temp storage for clone prompt
        prompt_temp_path = None
        
        with torch.no_grad():
            if request.voice_s3_key:
                # voice_s3_key might now be a full URL from Cloudinary (returned by frontend)
                voice_input = request.voice_s3_key
                prompt_temp_path = f"/tmp/prompt_{uuid.uuid4()}.wav"
                
                print(f"Fetching voice sample: {voice_input}...")
                if voice_input.startswith("http"):
                    r = requests.get(voice_input)
                    with open(prompt_temp_path, "wb") as f:
                        f.write(r.content)
                else:
                    # Logic for internal public_id if needed, 
                    # but we'll assume frontend sends URL for simplicity
                    raise ValueError("voice_s3_key must be a full URL")

                wav = self.model.generate(
                    request.text, 
                    audio_prompt_path=prompt_temp_path,
                    language_id=request.language,
                    exaggeration=request.exaggeration,
                    cfg_weight=request.cfg_weight
                )
            else:
                wav = self.model.generate(
                    request.text,
                    language_id=request.language,
                    exaggeration=request.exaggeration,
                    cfg_weight=request.cfg_weight
                )
            wav_cpu = wav.cpu()

        # Save generated audio to buffer
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav_cpu, self.model.sr, format="wav")
        buffer.seek(0)

        # Upload to Cloudinary
        print("Uploading generated audio to Cloudinary...")
        upload_result = cloudinary.uploader.upload(
            buffer,
            resource_type="auto",
            folder="ai-voice-studio/generated"
        )
        
        # Cleanup temp file
        if prompt_temp_path and os.path.exists(prompt_temp_path):
            os.remove(prompt_temp_path)

        print(f"Uploaded to Cloudinary: {upload_result['secure_url']}")
        return TextToSpeechResponse(
            s3_Key=upload_result['public_id'],
            url=upload_result['secure_url']
        )
