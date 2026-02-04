from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid
import os
import io
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

# Load environment variables from frontend .env
load_dotenv("../frontend/.env")

cloudinary.config(
  cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
  api_key = os.getenv("CLOUDINARY_API_KEY"),
  api_secret = os.getenv("CLOUDINARY_API_SECRET"),
  secure = True
)

# Try importing Chatterbox, else fallback to gTTS
try:
    import torch
    import torchaudio
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    TTS_ENGINE = "chatterbox"
except ImportError:
    print("Chatterbox TTS not found. Falling back to gTTS.")
    from gtts import gTTS
    TTS_ENGINE = "gtts"

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure audio directory exists
AUDIO_DIR = "generated_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Mount the audio directory to serve files statically
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

class TextToSpeechRequest(BaseModel):
    text: str
    voice_s3_key: Optional[str] = None
    language: str = "en"
    exaggeration: float = 0.5
    cfg_weight: float = 0.5

class TextToSpeechResponse(BaseModel):
    s3_Key: str
    url: Optional[str] = None

model = None

def load_model():
    global model
    if TTS_ENGINE == "chatterbox":
        try:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            
            print(f"Loading Chatterbox model on {device}...")
            model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            print("Chatterbox Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load Chatterbox model: {e}")
    else:
        print("Using gTTS engine.")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/text-to-speech", response_model=TextToSpeechResponse)
async def generate_speech(request: TextToSpeechRequest):
    try:
        filename = f"{uuid.uuid4()}.wav"
        filepath = os.path.join(AUDIO_DIR, filename)

        if TTS_ENGINE == "chatterbox" and model is not None:
            with torch.no_grad():
                audio_prompt_path = None
                
                if request.voice_s3_key:
                    # Download voice sample (it might be a Cloudinary URL or public_id)
                    try:
                        voice_input = request.voice_s3_key
                        
                        # Local cache path
                        safe_filename = "".join([c if c.isalnum() else "_" for c in voice_input])[-50:]
                        local_voice_path = os.path.join(AUDIO_DIR, f"voice_{safe_filename}")
                        
                        if not os.path.exists(local_voice_path):
                            print(f"Fetching voice sample: {voice_input}...")
                            if voice_input.startswith("http"):
                                import requests
                                r = requests.get(voice_input)
                                with open(local_voice_path, "wb") as f:
                                    f.write(r.content)
                            else:
                                # Assume it's a public_id and try to get the URL
                                # Or just use it directly if we have a way.
                                # For now, frontend sends the public_id in s3Key field.
                                # Let's assume it might be a full URL now if we update frontend actions.
                                pass 
                            print(f"Saved voice sample to {local_voice_path}")
                        
                        audio_prompt_path = local_voice_path
                        
                    except Exception as e:
                        print(f"Failed to download voice from S3: {e}")
                        # Fallback to no cloning? Or raise error?
                        # Let's log and continue without cloning if possible, or maybe just fail.
                        # For now, let's try to proceed without cloning if download fails to avoid crash
                        pass

                if audio_prompt_path and os.path.exists(audio_prompt_path):
                     wav = model.generate(
                        request.text,
                        audio_prompt_path=audio_prompt_path,
                        language_id=request.language,
                        exaggeration=request.exaggeration,
                        cfg_weight=request.cfg_weight
                    )
                else:
                    wav = model.generate(
                        request.text,
                        language_id=request.language,
                        exaggeration=request.exaggeration,
                        cfg_weight=request.cfg_weight
                    )
                    
                wav_cpu = wav.cpu()
                torchaudio.save(filepath, wav_cpu, model.sr, format="wav")

        elif TTS_ENGINE == "gtts":
            # gTTS generation
            lang = request.language if request.language in ['en', 'es', 'fr', 'de'] else 'en'
            tts = gTTS(text=request.text, lang=lang)
            filename = f"{uuid.uuid4()}.mp3"
            filepath = os.path.join(AUDIO_DIR, filename)
            tts.save(filepath)
        
        else:
             raise HTTPException(status_code=503, detail="No TTS engine available")

        # Upload to Cloudinary
        try:
            print(f"Uploading {filepath} to Cloudinary...")
            upload_result = cloudinary.uploader.upload(
                filepath, 
                resource_type="auto",
                folder="ai-voice-studio/generated"
            )
            print(f"Uploaded to Cloudinary: {upload_result['secure_url']}")
            
            # Return both key (for DB) and URL (for direct playback)
            return TextToSpeechResponse(
                s3_Key=upload_result['public_id'],
                url=upload_result['secure_url']
            )
        except Exception as e:
            print(f"Cloudinary upload failed: {e}")
            # Fallback to local path if upload fails (works for local testing)
            return TextToSpeechResponse(s3_Key=f"audio/{filename}")

    except Exception as e:
        print(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
